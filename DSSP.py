import os
import copy
import argparse
import pandas as pd
import numpy as np
import signal
import ctypes
import tempfile
import subprocess
from multiprocessing import Pool

from Bio.PDB import PDBParser, is_aa
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.DSSP import make_dssp_dict, residue_max_acc

# Global Pool, for signal-handler terminate
_POOL = None


def _set_pdeathsig(sig=signal.SIGTERM):
    """Linux-only: send sig to this process when the parent dies, to avoid leftover workers."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, int(sig))
    except Exception:
        pass


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _set_pdeathsig(signal.SIGTERM)
    try:
        signal.signal(signal.SIGHUP, signal.SIG_DFL)
    except Exception:
        pass


def _shutdown_handler(signum, frame):
    global _POOL
    print(f"[SIGNAL] Received {signum}, terminating pool...", flush=True)
    if _POOL is not None:
        try:
            _POOL.terminate()
        except Exception:
            pass
        try:
            _POOL.join()
        except Exception:
            pass
    os._exit(128 + int(signum))


def normalize_icode(icode) -> str:
    """None/''/whitespace -> ' '; keep single char; otherwise try to compress to one char, fallback ' '."""
    if icode is None:
        return " "
    if isinstance(icode, str):
        if icode == "" or icode.strip() == "":
            return " "
        if len(icode) == 1:
            return icode
        s = icode.strip()
        return s[0] if s else " "
    s = str(icode)
    if s.strip() == "" or s == "None":
        return " "
    s2 = s.strip()
    return s2[0] if s2 else " "


def parse_resid_to_resseq_icode(resid):
    """Parse resid in a DSSP key to (resseq:int, icode:str); on failure return (None, None)."""
    icode = " "
    resseq = None

    if isinstance(resid, tuple):
        for x in resid:
            if isinstance(x, int):
                resseq = x
                break
        for x in resid:
            if isinstance(x, str) and len(x) == 1 and x.isalpha():
                icode = x
                break
        if resseq is None:
            for x in resid:
                if isinstance(x, str):
                    s = x.strip()
                    if s.lstrip("-").isdigit():
                        resseq = int(s)
                        break
        if resseq is None:
            return None, None
        return resseq, normalize_icode(icode)

    if isinstance(resid, int):
        return resid, " "

    s = str(resid).strip()
    digits = "".join(ch for ch in s if ch.isdigit() or ch == "-")
    if digits and digits != "-":
        try:
            resseq = int(digits)
            if s and s[-1].isalpha():
                icode = s[-1]
            return resseq, normalize_icode(icode)
        except Exception:
            return None, None

    return None, None


def run_mkdssp_to_file(pdb_path: str, mkdssp_exe: str, out_dssp_path: str) -> None:
    """mkdssp <pdb> <out.dssp>"""
    proc = subprocess.run(
        [mkdssp_exe, pdb_path, out_dssp_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"mkdssp failed (rc={proc.returncode})\n"
            f"cmd: {mkdssp_exe} {pdb_path} {out_dssp_path}\n"
            f"stderr:\n{proc.stderr}"
        )


def _angle_to_value(x: float, missing_value: float = 0.0) -> float:
    """DSSP/biopython use 360 for missing angles; map to missing_value (default 0.0) to avoid NaN in CSV."""
    try:
        v = float(x)
    except Exception:
        return float(missing_value)
    if abs(v - 360.0) < 1e-6:
        return float(missing_value)
    return float(v)


def parse_dssp_tuple(dssp_data):
    """
    Parse make_dssp_dict() dssp_data into unified fields.
    Current env 14-tuple layout:
      (aa, ss, acc, phi, psi, dssp_index,
       NH_O_1_relidx, NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy,
       NH_O_2_relidx, NH_O_2_energy, O_NH_2_relidx, O_NH_2_energy)
    Compatible with: len=15 with index first; len=13 without index.
    """
    if not isinstance(dssp_data, (tuple, list)):
        raise ValueError(f"Bad dssp_data type: {type(dssp_data)}")
    n = len(dssp_data)
    if n < 13:
        raise ValueError(f"dssp_data too short (len={n}): {dssp_data}")

    if n == 14 and isinstance(dssp_data[0], str) and isinstance(dssp_data[5], int):
        aa = str(dssp_data[0]).strip()
        ss = str(dssp_data[1]).strip()
        acc = float(dssp_data[2])
        phi = _angle_to_value(dssp_data[3])
        psi = _angle_to_value(dssp_data[4])

        NH_O_1_energy = float(dssp_data[7])
        O_NH_1_energy = float(dssp_data[9])
        NH_O_2_energy = float(dssp_data[11])
        O_NH_2_energy = float(dssp_data[13])

    elif n == 15 and isinstance(dssp_data[0], int) and isinstance(dssp_data[1], str):
        aa = str(dssp_data[1]).strip()
        ss = str(dssp_data[2]).strip()
        acc = float(dssp_data[3])
        phi = _angle_to_value(dssp_data[4])
        psi = _angle_to_value(dssp_data[5])

        NH_O_1_energy = float(dssp_data[8])
        O_NH_1_energy = float(dssp_data[10])
        NH_O_2_energy = float(dssp_data[12])
        O_NH_2_energy = float(dssp_data[14])

    elif n == 13 and isinstance(dssp_data[0], str) and isinstance(dssp_data[2], (int, float)):
        aa = str(dssp_data[0]).strip()
        ss = str(dssp_data[1]).strip()
        acc = float(dssp_data[2])
        phi = _angle_to_value(dssp_data[3])
        psi = _angle_to_value(dssp_data[4])

        NH_O_1_energy = float(dssp_data[6])
        O_NH_1_energy = float(dssp_data[8])
        NH_O_2_energy = float(dssp_data[10])
        O_NH_2_energy = float(dssp_data[12])

    else:
        aa = str(dssp_data[0]).strip()
        ss = str(dssp_data[1]).strip()
        acc = float(dssp_data[2])
        phi = _angle_to_value(dssp_data[3])
        psi = _angle_to_value(dssp_data[4])

        pair_energies = []
        for j in range(5, n - 1, 2):
            try:
                pair_energies.append(float(dssp_data[j + 1]))
            except Exception:
                pair_energies.append(0.0)
        while len(pair_energies) < 4:
            pair_energies.append(0.0)
        NH_O_1_energy, O_NH_1_energy, NH_O_2_energy, O_NH_2_energy = pair_energies[:4]

    aa_u = aa.upper() if aa else "-"
    max_acc = float(residue_max_acc.get(aa_u, 0.0))
    rasa = (acc / max_acc) if max_acc > 1e-6 else 0.0

    return aa_u, ss, acc, rasa, phi, psi, NH_O_1_energy, O_NH_1_energy, NH_O_2_energy, O_NH_2_energy


def _sanitize_df_numeric(df: pd.DataFrame, pdb_id: str = "", verbose: bool = True) -> pd.DataFrame:
    """Force numeric columns to be finite: coerce -> Inf/NaN->0, then clip to conservative ranges."""
    if df is None or df.empty:
        return df

    numeric_cols = [
        "ss_H", "ss_E", "ss_C",
        "ACC", "RASA", "phi", "psi",
        "NH_O_1_energy", "O_NH_1_energy", "NH_O_2_energy", "O_NH_2_energy",
        "ASA_complex", "ASA_chain", "Delta_ASA", "ASA_ratio",
    ]

    nan_before = 0
    inf_before = 0
    for c in numeric_cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            nan_before += int(s.isna().sum())
            inf_before += int(np.isinf(s.to_numpy(dtype=float, copy=False)).sum())

    for c in numeric_cols:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[np.isinf(df[c].to_numpy(dtype=float, copy=False)), c] = np.nan

    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    # Clip thresholds: RASA relaxed to 1.5 (some impls may be >1); angles [-180,180]; H-bond energy [-50,50]
    if "RASA" in df.columns:
        df["RASA"] = df["RASA"].clip(lower=0.0, upper=1.5)
    if "ASA_ratio" in df.columns:
        df["ASA_ratio"] = df["ASA_ratio"].clip(lower=0.0, upper=1.0)
    if "ACC" in df.columns:
        df["ACC"] = df["ACC"].clip(lower=0.0, upper=1e4)
    if "ASA_complex" in df.columns:
        df["ASA_complex"] = df["ASA_complex"].clip(lower=0.0, upper=1e4)
    if "ASA_chain" in df.columns:
        df["ASA_chain"] = df["ASA_chain"].clip(lower=0.0, upper=1e4)
    if "Delta_ASA" in df.columns:
        df["Delta_ASA"] = df["Delta_ASA"].clip(lower=-1e4, upper=1e4)

    if "phi" in df.columns:
        df["phi"] = df["phi"].clip(lower=-180.0, upper=180.0)
    if "psi" in df.columns:
        df["psi"] = df["psi"].clip(lower=-180.0, upper=180.0)

    for c in ["NH_O_1_energy", "O_NH_1_energy", "NH_O_2_energy", "O_NH_2_energy"]:
        if c in df.columns:
            df[c] = df[c].clip(lower=-50.0, upper=50.0)

    nan_after = 0
    inf_after = 0
    for c in numeric_cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            nan_after += int(s.isna().sum())
            inf_after += int(np.isinf(s.to_numpy(dtype=float, copy=False)).sum())

    if verbose and (nan_before > 0 or inf_before > 0 or nan_after > 0 or inf_after > 0):
        print(f"[SANITIZE] pdb={pdb_id} nan_before={nan_before} inf_before={inf_before} "
              f"nan_after={nan_after} inf_after={inf_after}", flush=True)

    return df


def compute_dssp_with_chain_asa(
    pdb_path,
    mkdssp_exe,
    probe_radius=1.4,
    n_points=960,
    debug_bad_keys=False,
    keep_tmp_dssp=False,
    sanitize=True,
    sanitize_verbose=True,
):
    """Build DSSP+ASA DataFrame for a single PDB (one residue per row)."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]
    pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]

    tmp_dir = tempfile.gettempdir()
    tmp_dssp = os.path.join(tmp_dir, f"{pdb_id}.{os.getpid()}.dssp")
    run_mkdssp_to_file(pdb_path, mkdssp_exe, tmp_dssp)

    dssp_dict, dssp_keys = make_dssp_dict(tmp_dssp)

    if (not keep_tmp_dssp) and os.path.isfile(tmp_dssp):
        try:
            os.remove(tmp_dssp)
        except Exception:
            pass

    sr_complex = ShrakeRupley(probe_radius=probe_radius, n_points=n_points)
    struct_complex = copy.deepcopy(structure)
    model_complex = struct_complex[0]
    sr_complex.compute(model_complex, level="R")

    asa_complex = {}  # (chain_id, resseq, icode) -> asa
    for chain in model_complex:
        chain_id = chain.id
        for residue in chain:
            if not is_aa(residue):
                continue
            res_id = residue.id
            resseq = int(res_id[1])
            icode = normalize_icode(res_id[2])
            asa_complex[(chain_id, resseq, icode)] = float(getattr(residue, "sasa", 0.0))

    # Per-chain ASA: detach other chains then compute separately
    asa_chain = {}
    for chain in model:
        chain_id = chain.id
        struct_single = copy.deepcopy(structure)
        model_single = struct_single[0]
        for other in list(model_single):
            if other.id != chain_id:
                model_single.detach_child(other.id)

        sr_single = ShrakeRupley(probe_radius=probe_radius, n_points=n_points)
        sr_single.compute(model_single, level="R")

        chain_single = model_single[chain_id]
        for residue in chain_single:
            if not is_aa(residue):
                continue
            res_id = residue.id
            resseq = int(res_id[1])
            icode = normalize_icode(res_id[2])
            asa_chain[(chain_id, resseq, icode)] = float(getattr(residue, "sasa", 0.0))

    rows = []
    for key in dssp_keys:
        chain_id = key[0]
        resid = key[1]

        resseq, icode = parse_resid_to_resseq_icode(resid)
        if resseq is None:
            if debug_bad_keys:
                print(f"[WARN] skip bad DSSP resid: pdb={pdb_id} key={key} resid={resid} type(resid)={type(resid)}")
            continue
        icode = normalize_icode(icode)

        dssp_data = dssp_dict[key]
        try:
            aa, ss, acc, rasa, phi, psi, NH_O_1_energy, O_NH_1_energy, NH_O_2_energy, O_NH_2_energy = parse_dssp_tuple(dssp_data)
        except Exception as e:
            if debug_bad_keys:
                print(f"[WARN] bad dssp_data: pdb={pdb_id} key={key} len={len(dssp_data)} data={dssp_data} err={e}")
            continue

        asa_c = asa_complex.get((chain_id, resseq, icode), 0.0)
        asa_u = asa_chain.get((chain_id, resseq, icode), asa_c)
        delta_asa = asa_u - asa_c
        asa_ratio = asa_c / asa_u if asa_u > 1e-6 else 0.0

        # ss one-hot: only H / E / other(C)
        if ss == "H":
            ss_H, ss_E, ss_C = 1.0, 0.0, 0.0
        elif ss == "E":
            ss_H, ss_E, ss_C = 0.0, 1.0, 0.0
        else:
            ss_H, ss_E, ss_C = 0.0, 0.0, 1.0

        rows.append(
            {
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "resseq": resseq,
                "icode": icode,
                "aa": aa,
                "ss": ss,
                "ss_H": ss_H,
                "ss_E": ss_E,
                "ss_C": ss_C,
                "ACC": acc,
                "RASA": rasa,
                "phi": phi,
                "psi": psi,
                "NH_O_1_energy": NH_O_1_energy,
                "O_NH_1_energy": O_NH_1_energy,
                "NH_O_2_energy": NH_O_2_energy,
                "O_NH_2_energy": O_NH_2_energy,
                "ASA_complex": asa_c,
                "ASA_chain": asa_u,
                "Delta_ASA": delta_asa,
                "ASA_ratio": asa_ratio,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["chain_id", "resseq", "icode"], inplace=True)

    if sanitize:
        df = _sanitize_df_numeric(df, pdb_id=pdb_id, verbose=sanitize_verbose)

    return df


def _process_one_pdb(args):
    pdb_path, out_csv, mkdssp_exe, skip_existing, debug_bad_keys, keep_tmp_dssp, sanitize, sanitize_verbose = args
    pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]

    if skip_existing and os.path.isfile(out_csv):
        print(f"[SKIP] {pdb_id} -> already exists {out_csv}")
        return

    print(f"[DSSP+ASA] Processing {pdb_id} ...")
    try:
        df = compute_dssp_with_chain_asa(
            pdb_path,
            mkdssp_exe=mkdssp_exe,
            debug_bad_keys=debug_bad_keys,
            keep_tmp_dssp=keep_tmp_dssp,
            sanitize=sanitize,
            sanitize_verbose=sanitize_verbose,
        )
        df.to_csv(out_csv, index=False)
    except Exception as e:
        print(f"[ERROR] Failed processing {pdb_id}: {e}")
    else:
        print(f"[OK] Wrote {out_csv}")


def main():
    global _POOL

    import config as cfg

    parser = argparse.ArgumentParser(
        description="Generate DSSP+ASA CSV per PDB (mkdssp->file->parse, safe multiprocessing)."
    )
    parser.add_argument(
        "--pdb_dir",
        type=str,
        default=cfg.MUT_PDB_DIR,
        required=False,
        help="PDB input directory (default data/{DATASET}/pdb/mut)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=cfg.DSSP_MUT_DIR,
        required=False,
        help="CSV output directory (default data/{DATASET}/DSSP/mut)",
    )
    parser.add_argument(
        "--mkdssp_exe",
        type=str,
        default=cfg.MKDSSP_EXE,
        help="Path to mkdssp executable (default mkdssp on PATH, or env MKDSSP_EXE)",
    )
    parser.add_argument("--workers", type=int, default=40, help="Number of parallel workers")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if output CSV already exists")
    parser.add_argument("--debug_bad_keys", action="store_true", help="Print abnormal DSSP key/resid/dssp_data entries")
    parser.add_argument("--keep_tmp_dssp", action="store_true", help="Keep temporary .dssp files (deleted by default)")
    parser.add_argument("--no_sanitize", action="store_true", help="Disable numeric sanitization (not recommended)")
    parser.add_argument("--sanitize_quiet", action="store_true", help="Do not print [SANITIZE] stats during cleaning")
    args = parser.parse_args()

    pdb_dir = args.pdb_dir
    out_dir = args.out_dir
    mkdssp_exe = args.mkdssp_exe
    workers = args.workers

    # When MKDSSP_EXE=mkdssp, resolve via PATH
    resolved_mkdssp = mkdssp_exe
    if not os.path.isfile(resolved_mkdssp):
        import shutil
        resolved_mkdssp = shutil.which(mkdssp_exe) or mkdssp_exe
    if not os.path.isfile(resolved_mkdssp):
        raise FileNotFoundError(
            f"mkdssp not found: {mkdssp_exe}\n"
            f"Install DSSP/mkdssp and add it to PATH, or set env MKDSSP_EXE"
        )
    mkdssp_exe = resolved_mkdssp
    if not os.path.isdir(pdb_dir):
        raise NotADirectoryError(f"pdb_dir not found: {pdb_dir}")

    os.makedirs(out_dir, exist_ok=True)

    pdb_files = [f for f in os.listdir(pdb_dir) if f.lower().endswith(".pdb")]
    pdb_files.sort()
    if not pdb_files:
        print(f"[WARN] No PDB files found in {pdb_dir}")
        return

    sanitize = (not args.no_sanitize)
    sanitize_verbose = (not args.sanitize_quiet)

    tasks = []
    for fname in pdb_files:
        pdb_path = os.path.join(pdb_dir, fname)
        pdb_id = os.path.splitext(fname)[0]
        out_csv = os.path.join(out_dir, f"{pdb_id}.csv")
        tasks.append((pdb_path, out_csv, mkdssp_exe, args.skip_existing, args.debug_bad_keys, args.keep_tmp_dssp, sanitize, sanitize_verbose))

    print(f"[INFO] Found {len(pdb_files)} PDB files in {pdb_dir}")
    print(f"[INFO] Output dir: {out_dir}")
    print(f"[INFO] mkdssp: {mkdssp_exe}")
    print(f"[INFO] workers: {workers}")
    print(f"[INFO] sanitize: {sanitize} (verbose={sanitize_verbose})")

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)
    try:
        signal.signal(signal.SIGHUP, _shutdown_handler)
    except Exception:
        pass

    _POOL = Pool(processes=workers, initializer=init_worker)
    try:
        _POOL.map(_process_one_pdb, tasks)
        _POOL.close()
        _POOL.join()
    except KeyboardInterrupt:
        print("[MAIN] KeyboardInterrupt -> terminate", flush=True)
        _POOL.terminate()
        _POOL.join()
    except Exception as e:
        print(f"[MAIN] Exception: {e} -> terminate", flush=True)
        _POOL.terminate()
        _POOL.join()
        raise
    finally:
        _POOL = None


if __name__ == "__main__":
    main()
