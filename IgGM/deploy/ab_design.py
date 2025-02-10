# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import logging
import time

import torch
from torch import nn
from torch.distributions import Categorical

from IgGM.model import DesignModel, PPIModel
from IgGM.protein import ProtStruct
from IgGM.protein.data_transform import get_asym_ids
from IgGM.utils import to_device, IGSO3Buffer, replace_with_mask
from .base_designer import BaseDesigner
from ..model.arch.core.diffuser import Diffuser


class AbDesigner(BaseDesigner):
    """The antibody & antigen multimer structure predictor.
    """

    def __init__(self, ppi_path, design_path, buffer_path, config):
        super().__init__()
        logging.info('Restoring the pre-trained IgGM-PPI-SeqPT model ...')
        self.plm_featurizer = PPIModel.restore(ppi_path)
        config.c_s = self.plm_featurizer.c_s
        config.c_p = self.plm_featurizer.c_z
        self.config = config
        self.igso3_buffer = IGSO3Buffer()
        self.igso3_buffer.load(buffer_path)
        self.diffuser = Diffuser(igso3_buffer=self.igso3_buffer)
        self.buffer_path = buffer_path
        logging.info('Restoring the pre-trained IgGM design model ...')
        self.model = DesignModel.restore(design_path, config)
        self.idxs_step = self._get_idxs_step()
        self.eval()

    def _build_inputs(self, chains):
        """Build an input dict for model inference."""
        # Separamos las cadenas de anticuerpo y antígeno
        antibody_chains = [chain for chain in chains if chain["id"] in {"H", "L"}]
        antigen_chains = [chain for chain in chains if chain["id"] in {"A", "B"}]

        # Determinamos el ID del ligando (anticuerpo)
        if len(antibody_chains) == 1:
            ligand_id = antibody_chains[0]["id"]
        elif len(antibody_chains) == 2:
            # Nos aseguramos de que "H" esté primero y "L" después
            antibody_chains.sort(key=lambda x: x["id"])
            ligand_id = ':'.join([chain["id"] for chain in antibody_chains])
        else:
            raise ValueError("El anticuerpo debe tener 1 o 2 cadenas (H o H y L)")

        # Determinamos el ID del receptor (antígeno)
        if len(antigen_chains) == 0:
            raise ValueError("Se debe proporcionar al menos una cadena para el antígeno")
        elif len(antigen_chains) == 1:
            receptor_id = antigen_chains[0]["id"]
        elif len(antigen_chains) == 2:
            receptor_id = ':'.join(sorted([chain["id"] for chain in antigen_chains]))
        else:
            raise ValueError("Sólo se soportan 1 o 2 cadenas para el antígeno")

        # Construimos el input base utilizando el método de la clase base
        inputs = super()._build_inputs(chains)
        inputs["base"]["ligand_id"] = ligand_id
        inputs["base"]["receptor_id"] = receptor_id

        complex_id = ':'.join([ligand_id, receptor_id])
        prot_data = self.init_prot_data(inputs, complex_id)  # Inicializa los datos a partir del ruido
        prot_data["base"]["complex_id"] = complex_id

        epitope = prot_data[complex_id]['epitope']
        if epitope is None:
            logging.info('No se proporcionó información de epítopo, la posición se determinará por el modelo')

        return prot_data

    def forward(self, inputs, chunk_size=None):
        """Run the antibody & antigen multimer structure predictor."""
        start = time.time()
        inputs = to_device(inputs, device=self.device)
        complex_id = inputs["base"]["complex_id"]
        idxs_step = self.idxs_step[:-1]  # se omite tau_{0} (fijo a 0)
        # Actualiza las secuencias y estructuras a través de múltiples iteraciones
        prot_data_curr = inputs[complex_id]  # datos del muestreo

        inputs_addi = None  # se inicializará al final de la primera iteración

        for idx_step in idxs_step:
            # Se utiliza la secuencia y estructura de la iteración anterior
            aa_seqs_pred, cord_tns_pred, inputs_addi = \
                self.__sample_cm_ss2ss(prot_data_curr, idx_step, inputs_addi, chunk_size=chunk_size)

            prot_data_curr['seq'] = aa_seqs_pred
            prot_data_curr['cord'] = cord_tns_pred
            prot_data_curr['cmsk'] = ProtStruct.get_cmsk_vld(aa_seqs_pred, self.device)

        logging.info('Start ab design model in %.2f second', time.time() - start)
        return prot_data_curr

    @torch.no_grad()
    def infer(self, chains, *args, **kwargs):
        # Permitimos IDs: "H", "L", "A" y "B", y entre 2 y 4 cadenas en total
        assert all(x["id"] in {'H', 'L', 'A', 'B'} for x in chains), 'Chain ID must be "H", "L", "A" or "B"'
        assert len(chains) in (2, 3, 4), 'FASTA file should contain 2, 3 or 4 chains'
        inputs = self._build_inputs(chains)
        outputs = self.forward(inputs, *args, **kwargs)
        return inputs, outputs

    def infer_pdb(self, chains, filename, *args, **kwargs):
        inputs, outputs = self.infer(chains, *args, **kwargs)
        complex_id = inputs["base"]["complex_id"]
        raw_seqs = {}
        for chain_id in complex_id.split(":"):
            raw_seq = inputs[chain_id]["base"]["seq"]
            raw_seqs[chain_id] = raw_seq

        self._output_to_fasta(inputs, outputs, filename[:-4] + ".fasta")
        self._output_to_pdb(inputs, outputs, filename)

    def __sample_cm_ss2ss(self, prot_data_curr, idx_step, inputs_addi, chunk_size=None):
        """Sample amino-acid sequences & backbone structures w/ CM."""
        # Realiza la pasada forward
        inputs = self.__build_inputs_cm(prot_data_curr, idx_step)
        outputs = self.model(inputs, inputs_addi=inputs_addi, chunk_size=chunk_size)

        inputs_addi = self.build_inputs_addi(outputs)  # para la siguiente iteración

        # Construye las secuencias y estructuras predichas
        prob_tns = nn.functional.softmax(outputs['1d'].permute(0, 2, 1), dim=2)
        distr = Categorical(probs=prob_tns)
        aa_seqs_pred = self.sample_seqs_from_distr(distr)
        pmsk_vec = inputs['pmsk']
        aa_seqs_pred = [replace_with_mask(inputs['seq-o'], aa_seq_pred, pmsk_vec)
                        for aa_seq_pred in aa_seqs_pred]
        cord_tns_pred = self.calc_cords_from_param(aa_seqs_pred, outputs['3d']['param'][-1])
        pmsk_vec_ligand = inputs['pmsk-ligand']
        cord_tns_pred = torch.where(pmsk_vec_ligand.view(1, -1, 1, 1).to(torch.bool),
                                     cord_tns_pred, inputs['cord-o'])

        return aa_seqs_pred[0], cord_tns_pred[0], inputs_addi

    def __build_inputs_cm(self, prot_data_curr, idx_step):
        """Build inputs for CM-based sampling."""
        # Inicializa: perturba aleatoriamente las secuencias y/o estructuras
        prot_data_pert = self.diffuser.run(prot_data_curr, idx_step)
        # Construye un diccionario de tensores de entrada
        inputs = self.model.featurize(self.plm_featurizer, prot_data_pert)
        ic_feat = torch.zeros_like(prot_data_curr['asym_id'])
        ag_len = len(prot_data_curr['epitope'])
        ic_feat[:, -ag_len:] = prot_data_curr['epitope']

        inputs['ic_feat'] = ic_feat.unsqueeze(-1).type_as(inputs['sfea-i'])
        return inputs
