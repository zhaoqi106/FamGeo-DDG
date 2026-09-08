#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FoldX RepairPDB: remove HETATM then energy-minimize.
Paths (see config.py): data/{DATASET}/pdb/wt -> repairpdb; executable bin/foldx or FOLDX_EXE.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Select

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


def normalize_id(x: str) -> str:
    """Strip whitespace, lowercase, drop .pdb suffix; return pdb id without extension."""
    if x is None:
        return ""
    s = str(x).strip().lower()
    if s.endswith(".pdb"):
        s = s[:-4]
    return s


def load_pdb_ids_from_csv(csv_path: str, col: str) -> list:
    df = pd.read_csv(csv_path)
    if col not in df.columns:
        raise ValueError(f"Column not found in CSV: {col}; existing columns: {list(df.columns)}")

    ids = []
    for v in df[col].dropna().tolist():
        pid = normalize_id(v)
        if pid:
            ids.append(pid)
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def find_file_case_insensitive(directory: str, filename: str):
    if not os.path.isdir(directory):
        return None

    direct_path = os.path.join(directory, filename)
    if os.path.isfile(direct_path):
        return direct_path

    filename_lower = filename.lower()
    for item in os.listdir(directory):
        if item.lower() == filename_lower:
            full_path = os.path.join(directory, item)
            if os.path.isfile(full_path):
                return full_path
    return None


class NonHetSelect(Select):
    def accept_residue(self, residue):
        return residue.id[0] == " "


def clean_to_file(input_pdb: str, output_pdb: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("PDB", input_pdb)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, NonHetSelect())


def run_foldx_repair(foldx_exe: str, workdir: str, pdb_filename: str):
    cmd = [foldx_exe, "--command=RepairPDB", f"--pdb={pdb_filename}"]
    try:
        subprocess.run(cmd, cwd=workdir, check=True, timeout=6000)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"FoldX timed out (>6000s): {pdb_filename}")


def find_repair_output(workdir: str, cleaned_basename: str) -> str:
    # FoldX output convention: {basename}_Repair.pdb; case may vary
    root = cleaned_basename[:-4] if cleaned_basename.lower().endswith(".pdb") else cleaned_basename
    expected = os.path.join(workdir, f"{root}_Repair.pdb")
    if os.path.isfile(expected):
        return expected

    candidates = [f for f in os.listdir(workdir) if f.lower().endswith("_repair.pdb")]
    if len(candidates) == 1:
        return os.path.join(workdir, candidates[0])

    raise FileNotFoundError(
        f"Repair output not found. Expected: {os.path.basename(expected)}; candidates: {candidates}"
    )


def process_single_pdb(args):
    pid, input_dir, output_dir, foldx_exe, total, index = args

    in_pdb = find_file_case_insensitive(input_dir, f"{pid}.pdb")
    out_pdb = os.path.join(output_dir, f"{pid.upper()}.pdb")

    if in_pdb is None:
        msg = f"[{index}/{total}] MISSING: {os.path.join(input_dir, pid)}.pdb"
        print(msg)
        return (pid, "MISSING", msg)

    if os.path.isfile(out_pdb):
        msg = f"[{index}/{total}] SKIP (exists): {out_pdb}"
        print(msg)
        return (pid, "SKIP", msg)

    try:
        with tempfile.TemporaryDirectory(prefix="foldx_repair_") as tmpdir:
            cleaned_basename = f"{pid}.pdb"
            cleaned_path = os.path.join(tmpdir, cleaned_basename)
            clean_to_file(in_pdb, cleaned_path)
            run_foldx_repair(foldx_exe, tmpdir, cleaned_basename)
            repaired_tmp = find_repair_output(tmpdir, cleaned_basename)
            shutil.move(repaired_tmp, out_pdb)

        msg = f"[{index}/{total}] OK: {pid} -> {out_pdb}"
        print(msg)
        return (pid, "OK", msg)

    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.strip() if e.stderr else "No stderr"
        msg = f"[{index}/{total}] FAIL(FoldX): {pid} | stderr: {stderr_msg}"
        print(msg)
        return (pid, "FAIL", msg)

    except Exception as e:
        msg = f"[{index}/{total}] FAIL: {pid} | {e}"
        print(msg)
        return (pid, "FAIL", msg)


def parse_args():
    p = argparse.ArgumentParser(description="FoldX RepairPDB batch runner")
    p.add_argument("--dataset", default=cfg.DATASET, help="Dataset name; defaults to config.DATASET")
    p.add_argument("--csv", default=None, help="CSV with PDB column; default data/{DATASET}/{DATASET}.csv")
    p.add_argument("--csv-col", default="PDB", help="PDB column name in CSV")
    p.add_argument("--input-dir", default=None, help="Raw WT PDB directory; default data/{DATASET}/pdb/wt")
    p.add_argument("--output-dir", default=None, help="Repair output directory; default data/{DATASET}/pdb/repairpdb")
    p.add_argument("--foldx", default=cfg.FOLDX_EXE, help="Path to FoldX executable")
    p.add_argument("--workers", type=int, default=10, help="Number of parallel workers")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.join(cfg.PROJECT_ROOT, args.dataset)
    csv_path = args.csv or os.path.join(data_dir, f"{args.dataset}.csv")
    input_dir = args.input_dir or os.path.join(data_dir, "pdb", "wt")
    output_dir = args.output_dir or os.path.join(data_dir, "pdb", "repairpdb")
    foldx_exe = args.foldx

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not os.path.isfile(foldx_exe):
        raise FileNotFoundError(
            f"FoldX not found: {foldx_exe}\n"
            f"Install FoldX under bin/foldx, or set FOLDX_EXE / use --foldx"
        )

    os.makedirs(output_dir, exist_ok=True)

    pdb_ids = load_pdb_ids_from_csv(csv_path, args.csv_col)
    total = len(pdb_ids)

    print(f"\nProcessing {total} PDBs, workers={args.workers}")
    print(f"CSV       : {csv_path}")
    print(f"INPUT_DIR : {input_dir}")
    print(f"OUTPUT_DIR: {output_dir}")
    print(f"FOLDX_EXE : {foldx_exe}\n")

    args_list = [
        (pid, input_dir, output_dir, foldx_exe, total, i + 1)
        for i, pid in enumerate(pdb_ids)
    ]

    with Pool(processes=args.workers) as pool:
        results = pool.map(process_single_pdb, args_list)

    ok = sum(1 for _, status, _ in results if status == "OK")
    skip = sum(1 for _, status, _ in results if status == "SKIP")
    missing = sum(1 for _, status, _ in results if status == "MISSING")
    failed = sum(1 for _, status, _ in results if status == "FAIL")

    log_path = os.path.join(output_dir, "repair_log.txt")
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"INPUT_DIR={input_dir}\n")
        log.write(f"CSV_PATH={csv_path}\n")
        log.write(f"OUTPUT_DIR={output_dir}\n")
        log.write(f"FOLDX_EXE={foldx_exe}\n")
        log.write(f"PROCESSES={args.workers}\n\n")
        for _, _, msg in results:
            log.write(msg + "\n")
        log.write("\n")
        log.write(f"TOTAL={total}, OK={ok}, SKIP={skip}, MISSING={missing}, FAILED={failed}\n")

    print("\n==== Summary ====")
    print("TOTAL  :", total)
    print("OK     :", ok)
    print("SKIP   :", skip)
    print("MISSING:", missing)
    print("FAILED :", failed)
    print("LOG    :", log_path)


if __name__ == "__main__":
    main()
