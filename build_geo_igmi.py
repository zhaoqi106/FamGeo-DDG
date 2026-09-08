#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GeoPPI/IGMI-style 5-fold SSCV (S1131/S4169/S645/M1101) -> data/{DATASET}/Struc/

Rules:
1. GeoPPI S3 antibody-antigen PDBs use antigen-only ECOD H-group.
2. Other PDBs use IGMI ecod_dict.ecod complete whole-PDB signature.
3. Effective signature containing NO_H_NAME -> singleton PDB cluster.
4. Fully named identical effective signatures -> merge.
5. PDB absent from ecod_dict -> singleton PDB cluster.
6. Intact clusters allocated with deterministic LPT: largest first -> currently smallest fold.
7. train/val CSVs preserve ONLY the original input columns.
"""

import argparse
import csv
import hashlib
import os
import pickle
from collections import OrderedDict, defaultdict


ANTIGEN_H = {
    "1AK4": "Retrovirus capsid protein N-terminal domain",
    "1BJ1": "Cystine-knot cytokines",
    "1CZ8": "Cystine-knot cytokines",
    "1DQJ": "Lysozyme-like",
    "1DVF": "Immunoglobulin-related",
    "1FFW": "Class I glutamine amidotransferase-like",
    "1JRH": "Immunoglobulin-related",
    "1JTG": "a+b domain in beta-lactamase/transpeptidase-like proteins",
    "1KTZ": "Snake toxin-like",
    "1MHP": "HAD domain-related",
    "1MLC": "Lysozyme-like",
    "1N8Z": "Leucine-rich repeats",
    "1VFB": "Lysozyme-like",
    "1YY9": "Leucine-rich repeats",
    "2JEL": "HPr-like",
    "2NY7": "gp120 inner domain",
    "2NYY": 'Metalloproteases ("zincins") catalytic domain',
    "2NZ9": 'Metalloproteases ("zincins") catalytic domain',
    "3BDY": "Cystine-knot cytokines",
    "3BE1": "Leucine-rich repeats",
    "3BN9": "RIFT-related",
    "3HFM": "Lysozyme-like",
    "3K2M": "Immunoglobulin-related",
    "3NGB": "gp120 inner domain",
    "3NPS": "RIFT-related",
    "1T83": "Immunoglobulin-related",
    "3WJJ": "Immunoglobulin-related",
}

S3_OVERRIDE_BY_DATASET = {
    "S1131": {
        "1AK4", "1FFW", "1JTG", "1KTZ",
    },
    "S4169": {
        "1AK4", "1BJ1", "1CZ8", "1DQJ", "1DVF", "1FFW", "1JRH",
        "1JTG", "1KTZ", "1MHP", "1MLC", "1N8Z", "1VFB", "1YY9",
        "2JEL", "2NYY", "2NZ9", "3BN9", "3HFM", "3NGB", "3NPS",
    },
    "S645": {
        "1AK4", "1BJ1", "1CZ8", "1DQJ", "1DVF", "1FFW", "1JRH",
        "1JTG", "1KTZ", "1MHP", "1MLC", "1N8Z", "1VFB", "1YY9",
        "2JEL", "2NYY", "2NZ9", "3BDY", "3BE1", "3BN9", "3HFM",
        "3K2M", "3NGB", "3NPS",
    },
    "M1101": {
        "1AK4", "1BJ1", "1CZ8", "1DQJ", "1DVF", "1FFW", "1JRH",
        "1JTG", "1KTZ", "1MHP", "1MLC", "1N8Z", "1T83", "1VFB",
        "1YY9", "2JEL", "2NY7", "2NYY", "2NZ9", "3BDY", "3BE1",
        "3BN9", "3HFM", "3K2M", "3NGB", "3NPS", "3WJJ",
    },
}

PURE_ANTIGEN_ONLY = {"S645", "M1101"}
DEFAULT_DATASETS = ["S1131", "S4169", "S645", "M1101"]


def normalize_pdb(raw):
    # Normalize Excel scientific notation / HM_ / reverse_ prefixes
    s = str(raw).strip().strip('"').strip("'")
    if not s:
        return s
    if s.upper() in {"1.00E+96", "1E+96", "1.0E+96", "1.000E+96"}:
        return "1E96"
    parts = s.split("_")
    if parts[0].upper() == "HM" and len(parts) >= 2:
        return parts[1].upper()
    if parts[0].lower() == "reverse" and len(parts) >= 2:
        return parts[1].upper()
    return parts[0].upper()


def detect_pdb_column(fieldnames):
    for c in ("PDB", "protein", "pdb", "pdbID", "pdb_id"):
        if c in fieldnames:
            return c
    raise ValueError(f"Cannot find PDB column. Columns: {fieldnames}")


def canonical_signature(names):
    return str(sorted([str(x) for x in names], key=str.lower))


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError(f"No CSV header: {path}")
        return list(r.fieldnames), list(r)


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_ecod_reverse(path):
    with open(path, "rb") as f:
        ecod_dict = pickle.load(f)
    if not isinstance(ecod_dict, dict):
        raise TypeError(f"{path} is not a dict")

    pdb_to_sig = {}
    for sig, pdbs in ecod_dict.items():
        for raw in pdbs:
            pdb = normalize_pdb(raw)
            if pdb in pdb_to_sig and pdb_to_sig[pdb] != sig:
                raise RuntimeError(f"PDB {pdb} occurs under multiple ECOD signatures")
            pdb_to_sig[pdb] = sig
    return ecod_dict, pdb_to_sig


def build_pdb_rows(rows, pdb_col):
    out = OrderedDict()
    for i, row in enumerate(rows):
        pdb = normalize_pdb(row[pdb_col])
        out.setdefault(pdb, []).append(i)
    return out


def build_clusters(dataset, pdb_rows, pdb_to_sig):
    overrides = S3_OVERRIDE_BY_DATASET[dataset]
    dataset_pdbs = set(pdb_rows)

    if dataset in PURE_ANTIGEN_ONLY:
        missing = sorted(dataset_pdbs - overrides)
        unexpected = sorted(overrides - dataset_pdbs)
        if missing:
            raise RuntimeError(f"{dataset}: PDBs not covered by GeoPPI S3: {missing}")
        if unexpected:
            raise RuntimeError(f"{dataset}: expected S3 PDBs absent from CSV: {unexpected}")
    else:
        missing = sorted(overrides - dataset_pdbs)
        if missing:
            raise RuntimeError(f"{dataset}: expected S3 override PDBs absent: {missing}")

    pdb_meta = OrderedDict()
    for pdb in pdb_rows:
        if pdb in overrides:
            if pdb not in ANTIGEN_H:
                raise RuntimeError(f"Missing antigen H-group for {pdb}")
            eff = canonical_signature([ANTIGEN_H[pdb]])
            pdb_meta[pdb] = {
                "source": "GeoPPI_S3_antigen_only",
                "original_signature": pdb_to_sig.get(pdb, ""),
                "effective_signature": eff,
            }
        elif pdb in pdb_to_sig:
            sig = pdb_to_sig[pdb]
            pdb_meta[pdb] = {
                "source": "IGMI_whole_PDB_signature",
                "original_signature": sig,
                "effective_signature": sig,
            }
        else:
            pdb_meta[pdb] = {
                "source": "singleton_not_in_ecod_dict",
                "original_signature": "",
                "effective_signature": "",
            }

    clusters = OrderedDict()
    cluster_meta = {}
    merge_candidates = OrderedDict()

    for pdb, meta in pdb_meta.items():
        sig = meta["effective_signature"]
        if not sig:
            cid = f"PDB::{pdb}"
            rule = "singleton_not_in_ecod_dict"
        elif "NO_H_NAME" in sig:
            cid = f"PDB::{pdb}"
            rule = "singleton_due_to_NO_H_NAME"
        else:
            merge_candidates.setdefault(sig, []).append(pdb)
            continue

        clusters[cid] = list(pdb_rows[pdb])
        cluster_meta[cid] = {"rule": rule, "signature": sig, "pdbs": [pdb]}
        meta["cluster_id"] = cid
        meta["cluster_rule"] = rule

    for sig, pdbs in merge_candidates.items():
        cid = "SIG::" + hashlib.sha1(sig.encode("utf-8")).hexdigest()[:12]
        ids = []
        for pdb in pdbs:
            ids.extend(pdb_rows[pdb])
        clusters[cid] = ids
        cluster_meta[cid] = {
            "rule": "merged_exact_effective_signature",
            "signature": sig,
            "pdbs": list(pdbs),
        }
        for pdb in pdbs:
            pdb_meta[pdb]["cluster_id"] = cid
            pdb_meta[pdb]["cluster_rule"] = "merged_exact_effective_signature"

    assigned = [i for ids in clusters.values() for i in ids]
    expected = sum(len(v) for v in pdb_rows.values())
    if len(assigned) != expected or len(set(assigned)) != expected:
        raise RuntimeError(f"{dataset}: invalid cluster row assignment")

    return clusters, cluster_meta, pdb_meta


def balanced_lpt(clusters, n_folds):
    ordered = sorted(clusters.items(), key=lambda x: (-len(x[1]), x[0]))
    fold_rows = [[] for _ in range(n_folds)]
    fold_clusters = [[] for _ in range(n_folds)]
    loads = [0] * n_folds
    cluster_to_fold = {}

    for cid, ids in ordered:
        fold = min(range(n_folds), key=lambda x: (loads[x], x))
        fold_rows[fold].extend(ids)
        fold_clusters[fold].append(cid)
        loads[fold] += len(ids)
        cluster_to_fold[cid] = fold

    return fold_rows, fold_clusters, cluster_to_fold


def process_dataset(dataset, base_dir, pdb_to_sig, n_folds):
    input_csv = os.path.join(base_dir, dataset, f"{dataset}.csv")
    out_dir = os.path.join(base_dir, dataset, "Struc")

    fieldnames, rows = read_csv(input_csv)
    pdb_col = detect_pdb_column(fieldnames)
    pdb_rows = build_pdb_rows(rows, pdb_col)
    clusters, cluster_meta, pdb_meta = build_clusters(dataset, pdb_rows, pdb_to_sig)
    fold_rows, fold_clusters, cluster_to_fold = balanced_lpt(clusters, n_folds)

    row_to_cluster = {}
    row_to_fold = {}
    for cid, ids in clusters.items():
        fold = cluster_to_fold[cid]
        for i in ids:
            row_to_cluster[i] = cid
            row_to_fold[i] = fold

    cluster_folds = defaultdict(set)
    pdb_folds = defaultdict(set)
    for i, row in enumerate(rows):
        pdb = normalize_pdb(row[pdb_col])
        cluster_folds[row_to_cluster[i]].add(row_to_fold[i])
        pdb_folds[pdb].add(row_to_fold[i])

    bad_clusters = {k: v for k, v in cluster_folds.items() if len(v) != 1}
    bad_pdbs = {k: v for k, v in pdb_folds.items() if len(v) != 1}
    if bad_clusters or bad_pdbs:
        raise RuntimeError(
            f"{dataset}: leakage detected, clusters={bad_clusters}, PDBs={bad_pdbs}"
        )

    os.makedirs(out_dir, exist_ok=True)

    for val_fold in range(n_folds):
        val_ids = sorted(fold_rows[val_fold])
        train_ids = sorted(
            i
            for f in range(n_folds)
            if f != val_fold
            for i in fold_rows[f]
        )
        if set(val_ids) & set(train_ids):
            raise RuntimeError(f"{dataset} fold {val_fold}: train/val overlap")
        if len(val_ids) + len(train_ids) != len(rows):
            raise RuntimeError(f"{dataset} fold {val_fold}: train+val count mismatch")

        write_csv(
            os.path.join(out_dir, f"val_fold_{val_fold}.csv"),
            fieldnames,
            [rows[i] for i in val_ids],
        )
        write_csv(
            os.path.join(out_dir, f"train_fold_{val_fold}.csv"),
            fieldnames,
            [rows[i] for i in train_ids],
        )

    fold_summary = []
    for fold, ids in enumerate(fold_rows):
        pdbs = {normalize_pdb(rows[i][pdb_col]) for i in ids}
        fold_summary.append({
            "fold": fold,
            "val_samples": len(ids),
            "train_samples": len(rows) - len(ids),
            "pdbs": len(pdbs),
            "clusters": len(fold_clusters[fold]),
            "sample_fraction": f"{len(ids) / len(rows):.8f}",
        })

    write_csv(
        os.path.join(out_dir, "fold_summary.csv"),
        ["fold", "val_samples", "train_samples", "pdbs", "clusters", "sample_fraction"],
        fold_summary,
    )

    cluster_summary = []
    for cid, ids in sorted(clusters.items(), key=lambda x: (-len(x[1]), x[0])):
        meta = cluster_meta[cid]
        s3_pdbs = [
            p for p in meta["pdbs"]
            if pdb_meta[p]["source"] == "GeoPPI_S3_antigen_only"
        ]
        cluster_summary.append({
            "cluster_id": cid,
            "cluster_rule": meta["rule"],
            "effective_ecod_signature": meta["signature"],
            "pdb_count": len(meta["pdbs"]),
            "pdbs": "|".join(meta["pdbs"]),
            "S3_override_pdb_count": len(s3_pdbs),
            "S3_override_pdbs": "|".join(s3_pdbs),
            "samples": len(ids),
            "fold": cluster_to_fold[cid],
        })

    write_csv(
        os.path.join(out_dir, "cluster_summary.csv"),
        [
            "cluster_id", "cluster_rule", "effective_ecod_signature",
            "pdb_count", "pdbs", "S3_override_pdb_count",
            "S3_override_pdbs", "samples", "fold",
        ],
        cluster_summary,
    )

    pdb_assignment = []
    for pdb, ids in pdb_rows.items():
        meta = pdb_meta[pdb]
        cid = meta["cluster_id"]
        pdb_assignment.append({
            "pdb": pdb,
            "samples": len(ids),
            "ecod_source": meta["source"],
            "original_ecod_signature": meta["original_signature"],
            "effective_ecod_signature": meta["effective_signature"],
            "cluster_rule": meta["cluster_rule"],
            "cluster_id": cid,
            "cluster_pdb_count": len(cluster_meta[cid]["pdbs"]),
            "cluster_samples": len(clusters[cid]),
            "fold": cluster_to_fold[cid],
        })

    write_csv(
        os.path.join(out_dir, "pdb_assignment.csv"),
        [
            "pdb", "samples", "ecod_source", "original_ecod_signature",
            "effective_ecod_signature", "cluster_rule", "cluster_id",
            "cluster_pdb_count", "cluster_samples", "fold",
        ],
        pdb_assignment,
    )

    s3_count = sum(m["source"] == "GeoPPI_S3_antigen_only" for m in pdb_meta.values())
    multi_count = sum(len(m["pdbs"]) > 1 for m in cluster_meta.values())
    val_counts = [len(x) for x in fold_rows]
    train_counts = [len(rows) - x for x in val_counts]
    pdb_counts = [len({normalize_pdb(rows[i][pdb_col]) for i in ids}) for ids in fold_rows]
    cluster_counts = [len(x) for x in fold_clusters]

    summary = [
        "=" * 96,
        f"{dataset} GeoPPI/IGMI final {n_folds}-fold split",
        "=" * 96,
        f"samples                 : {len(rows)}",
        f"normalized PDBs         : {len(pdb_rows)}",
        f"GeoPPI S3 overrides     : {s3_count}",
        f"clusters                : {len(clusters)}",
        f"multi-PDB clusters      : {multi_count}",
        "val sample counts       : " + ", ".join(map(str, val_counts)),
        "train sample counts     : " + ", ".join(map(str, train_counts)),
        "val PDB counts          : " + ", ".join(map(str, pdb_counts)),
        "val cluster counts      : " + ", ".join(map(str, cluster_counts)),
        f"cross-fold clusters     : {len(bad_clusters)}",
        f"cross-fold PDBs         : {len(bad_pdbs)}",
        f"output                  : {out_dir}",
    ]
    print("\n".join(summary))
    with open(os.path.join(out_dir, "split_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")

    return {
        "dataset": dataset,
        "samples": len(rows),
        "pdbs": len(pdb_rows),
        "clusters": len(clusters),
        "val_counts": val_counts,
        "output": out_dir,
    }


def main():
    import config as cfg

    p = argparse.ArgumentParser()
    p.add_argument(
        "--base-dir",
        default=cfg.PROJECT_ROOT,
        help="Data root directory (default ./data or env MUT_DATA_ROOT)",
    )
    p.add_argument(
        "--ecod-dict",
        default=cfg.ECOD_DICT_PATH,
        help="ECOD dictionary path (default data/ecod_dict.ecod or env ECOD_DICT)",
    )
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args()

    datasets = [x.upper() for x in args.datasets]
    bad = [x for x in datasets if x not in S3_OVERRIDE_BY_DATASET]
    if bad:
        raise ValueError(f"Unsupported datasets: {bad}")

    ecod_dict, pdb_to_sig = load_ecod_reverse(args.ecod_dict)
    print(f"Loaded ECOD dict: {args.ecod_dict}")
    print(f"ECOD signatures : {len(ecod_dict)}")

    results = []
    for dataset in datasets:
        print("\n" + "=" * 96)
        results.append(process_dataset(dataset, args.base_dir, pdb_to_sig, args.folds))

    print("\n" + "=" * 96)
    print("FINAL SUMMARY")
    print("=" * 96)
    for r in results:
        print(
            f"{r['dataset']}: samples={r['samples']} PDBs={r['pdbs']} "
            f"clusters={r['clusters']} val={r['val_counts']} -> {r['output']}"
        )


if __name__ == "__main__":
    main()
