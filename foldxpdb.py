#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FoldX BuildModel: batch-generate mutant PDBs from RepairPDB.
Paths in config.py; mutation list naming: foldx_txt/individual_list_{ID}.txt
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


def run_foldx_buildmodel(pdb_base, mutation, id_name, repair_dir, mut_dir, foldx_exe, txt_dir):
    task_dir = os.path.join(mut_dir, str(id_name))
    os.makedirs(task_dir, exist_ok=True)
    log_path = os.path.join(task_dir, f"{id_name}_log.txt")

    repaired_pdb = os.path.join(repair_dir, f"{pdb_base}.pdb")
    if not os.path.exists(repaired_pdb):
        error_msg = f"Repaired PDB not found: {repaired_pdb}"
        print(error_msg)
        with open(log_path, "w") as log_file:
            log_file.write(error_msg + "\n")
        return

    mut_pdb_copy = os.path.join(task_dir, os.path.basename(repaired_pdb))
    shutil.copy(repaired_pdb, mut_pdb_copy)

    mutant_file_src = os.path.join(txt_dir, f"individual_list_{id_name}.txt")
    if not os.path.exists(mutant_file_src):
        error_msg = f"Pre-existing mutant file not found: {mutant_file_src}"
        print(error_msg)
        with open(log_path, "w") as log_file:
            log_file.write(error_msg + "\n")
        return

    mutant_file = os.path.join(task_dir, os.path.basename(mutant_file_src))
    shutil.copy(mutant_file_src, mutant_file)

    cmd = [
        foldx_exe,
        "--command=BuildModel",
        f"--pdb={os.path.basename(repaired_pdb)}",
        f"--mutant-file={os.path.basename(mutant_file)}",
        "--numberOfRuns=3",
        "--out-pdb=true",
        "--order=_USERDEFINED",
    ]
    try:
        result = subprocess.run(cmd, cwd=task_dir, check=True, capture_output=True, text=True)
        with open(log_path, "w") as log_file:
            log_file.write(f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\n")
            log_file.write(f"Used mutant file: {mutant_file_src}\n")

        # FoldX output convention: {pdb_base}_1_*.pdb; take first sorted and rename to {ID}.pdb
        output_pattern = os.path.join(task_dir, f"{pdb_base}_1_*.pdb")
        output_files = glob.glob(output_pattern)
        if output_files:
            first_output = sorted(output_files)[0]
            new_name = os.path.join(mut_dir, f"{id_name}.pdb")
            shutil.move(first_output, new_name)
            print(f"Mutated PDB generated and renamed: {new_name}")
            with open(log_path, "a") as log_file:
                log_file.write(f"Success: Renamed {first_output} to {new_name}\n")
        else:
            error_msg = f"Output mutated PDB not found for {id_name} (pattern: {output_pattern})"
            print(error_msg)
            with open(log_path, "a") as log_file:
                log_file.write(error_msg + "\n")
    except subprocess.CalledProcessError as e:
        error_msg = f"Error running BuildModel for {id_name}: STDOUT={e.stdout} STDERR={e.stderr}"
        print(error_msg)
        with open(log_path, "w") as log_file:
            log_file.write(error_msg + "\n")
            log_file.write(f"Used mutant file: {mutant_file_src}\n")


def generate_mutations_batch(csv_path, repair_dir, mut_dir, foldx_exe, txt_dir, num_processes=16, limit=None):
    os.makedirs(mut_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(limit)

    with Pool(processes=num_processes) as pool:
        args = [
            (row["PDB"], row.get("Mutation", ""), row["ID"], repair_dir, mut_dir, foldx_exe, txt_dir)
            for _, row in df.iterrows()
        ]
        pool.starmap(run_foldx_buildmodel, args)


def parse_args():
    p = argparse.ArgumentParser(description="FoldX BuildModel batch runner")
    p.add_argument("--dataset", default=cfg.DATASET, help="Dataset name; defaults to config.DATASET")
    p.add_argument("--csv", default=None, help="Mutation table CSV; default data/{DATASET}/{DATASET}.csv")
    p.add_argument("--repair-dir", default=None, help="RepairPDB directory; default data/{DATASET}/pdb/repairpdb")
    p.add_argument("--mut-dir", default=None, help="Mutant output directory; default data/{DATASET}/pdb/mut")
    p.add_argument("--txt-dir", default=None, help="individual_list_*.txt directory; default data/{DATASET}/foldx_txt")
    p.add_argument("--foldx", default=cfg.FOLDX_EXE, help="Path to FoldX executable")
    p.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    p.add_argument("--limit", type=int, default=None, help="Process only first N rows (for debugging)")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.join(cfg.PROJECT_ROOT, args.dataset)
    csv_path = args.csv or os.path.join(data_dir, f"{args.dataset}.csv")
    repair_dir = args.repair_dir or os.path.join(data_dir, "pdb", "repairpdb")
    mut_dir = args.mut_dir or os.path.join(data_dir, "pdb", "mut")
    txt_dir = args.txt_dir or os.path.join(data_dir, "foldx_txt")
    foldx_exe = args.foldx

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.isdir(repair_dir):
        raise FileNotFoundError(f"repair directory not found: {repair_dir}")
    if not os.path.isdir(txt_dir):
        raise FileNotFoundError(f"foldx_txt directory not found: {txt_dir}")
    if not os.path.isfile(foldx_exe):
        raise FileNotFoundError(
            f"FoldX not found: {foldx_exe}\n"
            f"Install FoldX under bin/foldx, or set FOLDX_EXE / use --foldx"
        )

    print(f"CSV       : {csv_path}")
    print(f"REPAIR_DIR: {repair_dir}")
    print(f"MUT_DIR   : {mut_dir}")
    print(f"TXT_DIR   : {txt_dir}")
    print(f"FOLDX_EXE : {foldx_exe}")
    print(f"workers   : {args.workers}")

    generate_mutations_batch(
        csv_path,
        repair_dir,
        mut_dir,
        foldx_exe,
        txt_dir,
        num_processes=args.workers,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
