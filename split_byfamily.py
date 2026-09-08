import os
import numpy as np
import pandas as pd
from collections import defaultdict

import config

FAMILY_MAP_NAME = "pdb_family_map.csv"
SPLIT_SEED = getattr(config, "SPLIT_SEED", 42)
N_FOLDS = getattr(config, "N_FOLDS", 5)

OUT_DIR = os.path.join(config.DATA_DIR, "Fam")

BALANCE_BY_Y_MEAN = False
BALANCE_BY_BIG = False
BIG_DDG_THRESHOLD = 8.0  # Match extreme-label filter below: drop |ddG| >= 8


def greedy_assign_families(fam_sizes, n_folds, seed):
    """
    fam_sizes: family_id -> sample count.
    Greedy bin packing: largest families first, assign to fold with fewest samples so far;
    return fold -> set(family_id).
    """
    rng = np.random.RandomState(seed)

    items = list(fam_sizes.items())
    rng.shuffle(items)  # Shuffle same-size items to avoid bias
    items.sort(key=lambda x: x[1], reverse=True)

    fold_groups = [set() for _ in range(n_folds)]
    fold_counts = [0 for _ in range(n_folds)]

    for fam, sz in items:
        j = int(np.argmin(fold_counts))
        fold_groups[j].add(fam)
        fold_counts[j] += int(sz)

    return fold_groups, fold_counts


def greedy_assign_families_balance_mean(fam_sizes, fam_ymean, n_folds, seed):
    """Balance total sample count and y_mean together: count dominates, y_mean as light regularizer."""
    rng = np.random.RandomState(seed)

    items = list(fam_sizes.items())
    rng.shuffle(items)
    items.sort(key=lambda x: x[1], reverse=True)

    fold_groups = [set() for _ in range(n_folds)]
    fold_counts = np.zeros(n_folds, dtype=float)
    fold_y_sum = np.zeros(n_folds, dtype=float)

    for fam, sz in items:
        y = float(fam_ymean.get(fam, 0.0))
        overall_mean = float(np.sum([fam_ymean[f]*fam_sizes[f] for f in fam_sizes]) / max(1, sum(fam_sizes.values())))
        scores = fold_counts + 0.1 * np.abs((fold_y_sum + y*sz) / (fold_counts + sz + 1e-9) - overall_mean)

        j = int(np.argmin(scores))
        fold_groups[j].add(fam)
        fold_counts[j] += sz
        fold_y_sum[j] += y * sz

    return fold_groups, fold_counts.tolist()


def main():
    print("=== Make folds by FAMILY_ID ===")
    print("DATASET:", config.DATASET)
    print("CSV_PATH:", config.CSV_PATH)
    print("DATA_DIR:", config.DATA_DIR)
    print("N_FOLDS:", N_FOLDS)
    print("SPLIT_SEED:", SPLIT_SEED)
    print("OUT_DIR:", OUT_DIR)
    print()

    df = pd.read_csv(config.CSV_PATH)
    if "PDB" not in df.columns:
        raise ValueError("Sample CSV missing PDB column")

    # Filter extreme labels: keep ddG in (-8, 8)
    if "ddG" not in df.columns:
        raise ValueError("This split requires sample CSV to contain a ddG column (for extreme-value filtering)")
    before_n = len(df)
    df["ddG"] = df["ddG"].astype(float)
    df = df[(df["ddG"] < 8.0) & (df["ddG"] > -8.0)].reset_index(drop=True)
    removed = before_n - len(df)
    print(f"[FILTER] removed_extreme_ddG={removed} (before={before_n} after={len(df)}) range=[-8, 8]")

    fam_path = os.path.join(config.DATA_DIR, FAMILY_MAP_NAME)
    if not os.path.exists(fam_path):
        raise FileNotFoundError(f"family_map not found: {fam_path} (run build_pdb_family_map.py first)")

    fam_df = pd.read_csv(fam_path)
    if "PDB" not in fam_df.columns or "family_id" not in fam_df.columns:
        raise ValueError("pdb_family_map.csv must contain PDB and family_id columns")

    df["PDB"] = df["PDB"].astype(str).str.strip()
    fam_df["PDB"] = fam_df["PDB"].astype(str).str.strip()

    df = df.merge(fam_df[["PDB", "family_id"]], on="PDB", how="left")
    if df["family_id"].isna().any():
        missing = df.loc[df["family_id"].isna(), "PDB"].astype(str).unique().tolist()
        raise ValueError(f"Some PDBs lack family_id mapping (missing in pdb_family_map.csv); examples: {missing[:10]}")

    df["family_id"] = df["family_id"].astype(int)

    fam_sizes = df["family_id"].value_counts().to_dict()
    print("Family count:", len(fam_sizes))
    print("Top family sizes:", sorted(fam_sizes.values(), reverse=True)[:10])

    fam_ymean = {}
    if BALANCE_BY_Y_MEAN:
        if "ddG" not in df.columns:
            raise ValueError("BALANCE_BY_Y_MEAN=True requires sample CSV to contain a ddG column")
        fam_ymean = df.groupby("family_id")["ddG"].mean().to_dict()

    # family-disjoint: the same family must not span folds
    if BALANCE_BY_Y_MEAN:
        fold_groups, fold_counts = greedy_assign_families_balance_mean(fam_sizes, fam_ymean, N_FOLDS, SPLIT_SEED)
    else:
        fold_groups, fold_counts = greedy_assign_families(fam_sizes, N_FOLDS, SPLIT_SEED)

    os.makedirs(OUT_DIR, exist_ok=True)

    for k in range(N_FOLDS):
        val_fams = fold_groups[k]
        is_val = df["family_id"].isin(val_fams)
        val_df = df[is_val].drop(columns=["family_id"]).reset_index(drop=True)
        train_df = df[~is_val].drop(columns=["family_id"]).reset_index(drop=True)

        # PDB may appear across folds, but families must not overlap
        assert set(df.loc[is_val, "family_id"]) & set(df.loc[~is_val, "family_id"]) == set()

        train_path = os.path.join(OUT_DIR, f"train_fold_{k}.csv")
        val_path   = os.path.join(OUT_DIR, f"val_fold_{k}.csv")
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        print(f"fold {k}: train={len(train_df)} val={len(val_df)} "
              f"val_families={len(val_fams)} fold_target_count={int(fold_counts[k])}")

    fam_list_path = os.path.join(OUT_DIR, "fold_families.txt")
    with open(fam_list_path, "w") as f:
        for k in range(N_FOLDS):
            fams = sorted(list(fold_groups[k]))
            f.write(f"fold {k}: {len(fams)} families\n")
            f.write(",".join(map(str, fams)) + "\n\n")
    print("\nSaved fold family lists to:", fam_list_path)
    print("Saved split CSVs to:", OUT_DIR)

if __name__ == "__main__":
    main()
