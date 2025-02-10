# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import argparse
import os
import sys

import torch
import tqdm

from IgGM.protein import cal_ppi
from IgGM.protein.parser import parse_fasta, PdbParser

sys.path.append('.')

from IgGM.deploy import AbDesigner
from IgGM.utils import setup
from IgGM.model.pretrain import esm_ppi_650m_ab, antibody_design_trunk, IGSO3Buffer_trunk


def parse_args():
    parser = argparse.ArgumentParser(
        description='Antibody sequence and structure co-design w/ IgGM'
    )
    parser.add_argument(
        '--fasta', '-f', type=str, required=True,
        help='Path to input antibody FASTA file (with design region marked)'
    )
    parser.add_argument(
        '--antigen', '-ag', type=str, required=True,
        help='Path to input antigen PDB file'
    )
    parser.add_argument(
        '--output', type=str, default='outputs',
        help='Output directory for PDB files, default is "outputs"'
    )
    parser.add_argument(
        '--epitope', default=None, nargs='+', type=int,
        help='Epitope residues in antigen chain(s); e.g., 1 2 3 4 55'
    )
    parser.add_argument(
        '--device', '-d', type=str, default=None, help='Inference device'
    )
    parser.add_argument(
        '--steps', '-s', type=int, default=10, help='Number of sampling steps'
    )
    parser.add_argument(
        '--chunk_size', '-cs', type=int, default=64,
        help='Chunk size for long chain inference'
    )
    parser.add_argument(
        '--num_samples', '-ns', type=int, default=1,
        help='Number of samples for each input'
    )
    args = parser.parse_args()
    return args


def predict(args):
    """Predict antibody & antigen sequence and structures w/ pre-trained IgGM-Ag models."""
    fasta_path = args.fasta
    pdb_path = args.antigen

    sequences, ids, _ = parse_fasta(fasta_path)
    # Permitimos 1, 2, 3 o 4 cadenas en el FASTA
    assert len(sequences) in (1, 2, 3, 4), "Must be 1, 2, 3 or 4 chains in FASTA file"

    # Separamos las cadenas de anticuerpo (IDs "H" y "L")
    antibody_chains = [
        {"sequence": seq, "id": seq_id}
        for seq, seq_id in zip(sequences, ids) if seq_id in {"H", "L"}
    ]
    # El resto se consideran antígeno (IDs "A" y/o "B")
    antigen_ids = [seq_id for seq_id in ids if seq_id not in {"H", "L"}]

    antigen_chains = []
    # Para cada cadena de antígeno, se carga la estructura correspondiente del PDB
    for antigen_id in antigen_ids:
        aa_seq, atom_cord, atom_cmsk, _, _ = PdbParser.load(pdb_path, chain_id=antigen_id)
        if args.epitope is None:
            try:
                # Se puede calcular el epítopo considerando todas las cadenas de antígeno juntas
                epitope = cal_ppi(pdb_path, antigen_ids)
            except Exception as e:
                epitope = None
        else:
            epitope = torch.zeros(len(aa_seq))
            for i in args.epitope:
                epitope[i] = 1
        antigen_chains.append({
            "sequence": aa_seq,
            "cord": atom_cord,
            "cmsk": atom_cmsk,
            "epitope": epitope,
            "id": antigen_id
        })

    # Combina las cadenas de anticuerpo y antígeno
    chains = antibody_chains + antigen_chains

    _, basename = os.path.split(fasta_path)
    name = basename.split(".")[0]

    batches = [
        {
            "name": name,
            "chains": chains,
            "output": f"{args.output}/{name}_{i}.pdb",
        }
        for i in range(args.num_samples)
    ]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Predicción de estructuras y diseño de secuencias para anticuerpo & antígeno
    designer = AbDesigner(
        ppi_path=esm_ppi_650m_ab(),
        design_path=antibody_design_trunk(),
        buffer_path=IGSO3Buffer_trunk(),
        config=args,
    )
    designer.to(device)

    chunk_size = args.chunk_size
    print(f"# Inference samples: {len(batches)}")
    for task in tqdm.tqdm(batches):
        designer.infer_pdb(task["chains"], filename=task["output"], chunk_size=chunk_size)


def main():
    args = parse_args()
    setup(True)
    predict(args)


if __name__ == '__main__':
    main()
