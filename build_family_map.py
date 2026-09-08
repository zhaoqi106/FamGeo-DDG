import os
from collections import defaultdict
from typing import List, Optional, Tuple

import pandas as pd
from Bio.PDB import PDBParser

import config

# Family map: k-mer Jaccard clustering of complex receptor/ligand chain sequence signatures -> pdb_family_map.csv
KMER_K = 3
SIM_THRESHOLD = 0.9
OUT_NAME = "pdb_family_map.csv"


AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
    "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}


def parse_chain_set_semicolon(x) -> List[str]:
    """'A;B;C' -> ['A','B','C'], dedupe while preserving order."""
    if x is None:
        return []
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return []
    parts = [p.strip() for p in s.split(";")]
    out, seen = [], set()
    for p in parts:
        if not p:
            continue
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def get_receptor_ligand_chains_from_row(row: pd.Series) -> Tuple[List[str], List[str]]:
    """Heavy+Light=receptor; Antigen_chain1/2=ligand."""
    heavy = parse_chain_set_semicolon(row.get("Heavy_chain", ""))
    light = parse_chain_set_semicolon(row.get("Light_chain", ""))
    ag1 = parse_chain_set_semicolon(row.get("Antigen_chain1", ""))
    ag2 = parse_chain_set_semicolon(row.get("Antigen_chain2", ""))
    receptor = sorted(set(heavy + light))
    ligand = sorted(set(ag1 + ag2))
    return receptor, ligand


def pdb_file_path(pdb_id: str) -> Optional[str]:
    """Look up {pdb_id}.pdb / .PDB under WT_PDB_DIR."""
    p1 = os.path.join(config.WT_PDB_DIR, f"{pdb_id}.pdb")
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(config.WT_PDB_DIR, f"{pdb_id}.PDB")
    if os.path.exists(p2):
        return p2
    return None


def extract_chain_seq(structure, chain_id: str) -> str:
    """Convert standard protein residues (ATOM, blank hetflag) to one-letter sequence."""
    model = next(structure.get_models())
    if chain_id not in model:
        return ""

    chain = model[chain_id]
    residues = []
    for res in chain.get_residues():
        hetflag, resseq, icode = res.id
        if hetflag.strip():
            continue
        aa3 = res.get_resname().strip().upper()
        aa1 = AA3_TO_AA1.get(aa3)
        if aa1 is None:
            continue
        residues.append((resseq, str(icode).strip(), aa1))

    residues.sort(key=lambda x: (x[0], x[1]))
    return "".join(x[2] for x in residues)


def build_signatures_for_pdb(pdb_id: str, row: pd.Series, parser: PDBParser, cache: dict):
    """complex_sig = receptor_sig + '||' + ligand_sig; chain sequences joined by '|'."""
    if pdb_id not in cache:
        p = pdb_file_path(pdb_id)
        cache[pdb_id] = None if p is None else parser.get_structure(pdb_id, p)

    structure = cache[pdb_id]
    if structure is None:
        return "", "", ""

    receptor_chains, ligand_chains = get_receptor_ligand_chains_from_row(row)

    rec_parts = [extract_chain_seq(structure, ch) for ch in receptor_chains]
    lig_parts = [extract_chain_seq(structure, ch) for ch in ligand_chains]

    receptor_sig = "|".join(rec_parts)
    ligand_sig = "|".join(lig_parts)
    complex_sig = receptor_sig + "||" + ligand_sig
    return receptor_sig, ligand_sig, complex_sig


def kmers(s: str, k: int) -> set:
    s = (s or "").strip()
    if len(s) < k:
        return set()
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


def main():
    print("=== Build family map for strict family-level split ===")
    print("DATASET:", config.DATASET)
    print("CSV_PATH:", config.CSV_PATH)
    print("WT_PDB_DIR:", config.WT_PDB_DIR)
    print("OUT_DIR:", config.DATA_DIR)
    print("k-mer k:", KMER_K, "| similarity threshold:", SIM_THRESHOLD)
    print()

    df = pd.read_csv(config.CSV_PATH)
    if "PDB" not in df.columns:
        raise ValueError("CSV missing PDB column.")
    required_cols = ["Heavy_chain", "Light_chain", "Antigen_chain1", "Antigen_chain2"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV missing column {col}.")

    # Take first row per PDB, assuming chain-set annotations are consistent
    pdb_df = df.groupby("PDB").head(1).reset_index(drop=True)
    pdb_ids = pdb_df["PDB"].astype(str).str.strip().tolist()
    print("Unique PDB count:", len(pdb_ids))

    parser = PDBParser(QUIET=True)
    cache = {}

    rows = []
    missing_pdb_files = []
    empty_sig = 0

    for _, row in pdb_df.iterrows():
        pdb_id = str(row["PDB"]).strip()

        if pdb_file_path(pdb_id) is None:
            missing_pdb_files.append(pdb_id)
            rec_sig, lig_sig, sig = "", "", ""
        else:
            rec_sig, lig_sig, sig = build_signatures_for_pdb(pdb_id, row, parser, cache)

        if sig == "" or sig == "||":
            empty_sig += 1

        receptor_chains, ligand_chains = get_receptor_ligand_chains_from_row(row)
        rows.append(
            {
                "PDB": pdb_id,
                "receptor_chains": ";".join(receptor_chains),
                "ligand_chains": ";".join(ligand_chains),
                "receptor_sig": rec_sig,
                "ligand_sig": lig_sig,
                "complex_sig": sig,
            }
        )

    sig_df = pd.DataFrame(rows)
    print("Missing PDB files:", len(missing_pdb_files))
    if missing_pdb_files:
        print("  examples:", missing_pdb_files[:10])
    print("Empty signature rows:", empty_sig)
    print()

    sig_df["km"] = sig_df["complex_sig"].apply(lambda s: kmers(s, KMER_K))
    km_list = sig_df["km"].tolist()
    n = len(sig_df)

    uf = UnionFind(n)

    # Merge identical signatures first, then merge by Jaccard >= SIM_THRESHOLD
    sig_to_idx = defaultdict(list)
    for i, s in enumerate(sig_df["complex_sig"].tolist()):
        sig_to_idx[s].append(i)
    for idxs in sig_to_idx.values():
        if len(idxs) >= 2:
            base = idxs[0]
            for j in idxs[1:]:
                uf.union(base, j)

    valid = [
        i for i, s in enumerate(sig_df["complex_sig"].tolist())
        if s and s != "||" and len(km_list[i]) > 0
    ]
    print("Similarity compare count (valid PDB):", len(valid))

    for ii in range(len(valid)):
        i = valid[ii]
        for jj in range(ii + 1, len(valid)):
            j = valid[jj]
            sim = jaccard(km_list[i], km_list[j])
            if sim >= SIM_THRESHOLD:
                uf.union(i, j)

    root_to_fid = {}
    family_id = []
    for i in range(n):
        r = uf.find(i)
        if r not in root_to_fid:
            root_to_fid[r] = len(root_to_fid)
        family_id.append(root_to_fid[r])

    sig_df["family_id"] = family_id
    fam_sizes = sig_df["family_id"].value_counts().to_dict()
    sig_df["family_size"] = sig_df["family_id"].map(fam_sizes)

    out_path = os.path.join(config.DATA_DIR, OUT_NAME)
    sig_df.drop(columns=["km"], inplace=True)
    sig_df.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("Family count:", len(fam_sizes))
    print("Top family sizes:", sorted(fam_sizes.values(), reverse=True)[:10])

    if missing_pdb_files:
        miss_path = os.path.join(config.DATA_DIR, "missing_pdb_files.txt")
        with open(miss_path, "w", encoding="utf-8") as f:
            for pdb_id in missing_pdb_files:
                f.write(pdb_id + "\n")
        print("Saved missing list:", miss_path)


if __name__ == "__main__":
    main()
