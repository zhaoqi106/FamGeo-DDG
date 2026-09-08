# Preprocess: build precomputed large graphs for training-time dynamic cropping (Base+DSSP+DeltaESM)

import os
import argparse
import logging
import multiprocessing
import traceback
import warnings

import pandas as pd
import torch
from tqdm import tqdm

import config as cfg
from config import CSV_PATH, DATA_DIR, DATASET, FUSION_GRAPH_DIR
from graph_utils import build_local_fusion_graph

warnings.filterwarnings("ignore")

OUTPUT_DIR = FUSION_GRAPH_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, "graph_generation.log")
FAILED_CSV = os.path.join(OUTPUT_DIR, "failed_samples.csv")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    encoding="utf-8",
)

print(f"[INFO] DATASET: {DATASET}")
print(f"[INFO] CSV_PATH: {CSV_PATH}")
print(f"[INFO] Output dir: {OUTPUT_DIR}")
print(f"[INFO] Log file: {LOG_FILE}")
print(f"[INFO] Failed samples CSV: {FAILED_CSV}")
print(
    "[INFO] Graph feature scheme: Base(41) + "
    f"DSSP({'14' if getattr(cfg, 'USE_DSSP', True) else '0'}) + "
    f"DeltaESM({getattr(cfg, 'ESM_DIM', 2560) if getattr(cfg, 'USE_ESM', True) else 0})"
)


def load_metadata(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = ["PDB", "ID", "Mutation"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df


def process_one_sample(row: dict):
    pdb_id = str(row["PDB"]).strip()
    sample_id = str(row["ID"]).strip()
    mutation_str = str(row["Mutation"]).strip()
    save_path = os.path.join(OUTPUT_DIR, f"{sample_id}.pt")

    if os.path.exists(save_path):
        logging.info(f"SUCCESS | Exists | {sample_id} | {pdb_id}")
        return {
            "status": "Exists",
            "ID": sample_id,
            "PDB": pdb_id,
            "Mutation": mutation_str,
            "Error": "",
        }

    try:
        graph_data, status = build_local_fusion_graph(
            pdb_id=pdb_id,
            sample_id=sample_id,
            mutation_str=mutation_str,
            center_dist=25.0,
            edge_cutoff=25.0,
        )

        if graph_data is None:
            error_msg = f"Build Failed: {status}"
            logging.error(
                f"FAIL | {sample_id} | {pdb_id} | Mutation={mutation_str} | {error_msg}"
            )
            return {
                "status": "Error",
                "ID": sample_id,
                "PDB": pdb_id,
                "Mutation": mutation_str,
                "Error": error_msg,
            }

        ddg_value = row.get("ddG", row.get("ddg", None))
        if ddg_value is not None:
            graph_data.y = torch.tensor([float(ddg_value)], dtype=torch.float32)
        else:
            graph_data.y = None

        graph_data.sample_id = sample_id
        graph_data.pdb = pdb_id
        graph_data.mutation_str = mutation_str
        torch.save(graph_data, save_path)

        logging.info(f"SUCCESS | {sample_id} | {pdb_id} | Nodes={graph_data.x.size(0)}")
        return {
            "status": "Success",
            "ID": sample_id,
            "PDB": pdb_id,
            "Mutation": mutation_str,
            "Error": "",
        }

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        tb = traceback.format_exc()
        logging.error(
            f"FAIL | {sample_id} | {pdb_id} | Mutation={mutation_str} | "
            f"ErrorType={error_type} | Message={error_msg}\nTraceback:\n{tb}"
        )
        return {
            "status": "Error",
            "ID": sample_id,
            "PDB": pdb_id,
            "Mutation": mutation_str,
            "Error": f"{error_type}: {error_msg}",
        }


def main():
    parser = argparse.ArgumentParser(
        description="Generate precomputed mutation-centered graphs for training-time dynamic cropping."
    )
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers; 0 for serial")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples for testing")
    args = parser.parse_args()

    print(f"=== Start generating precomputed graphs | DATASET={DATASET} ===")
    print("[INFO] These graphs will be dynamically cropped in main.py by EFFECTIVE_RADIUS / EFFECTIVE_EDGE_CUTOFF")

    multiprocessing.set_start_method("spawn", force=True)

    df = load_metadata(CSV_PATH)
    samples = df.to_dict(orient="records")

    if args.limit > 0:
        samples = samples[: args.limit]

    print(f"Total samples: {len(samples)}")
    print(f"Building graphs (workers={args.workers})...")

    results = {"Success": 0, "Exists": 0, "Error": 0}
    failed_list = []

    if args.workers == 0:
        for s in tqdm(samples):
            res = process_one_sample(s)
            results[res["status"]] += 1
            if res["status"] == "Error":
                failed_list.append(res)
    else:
        with multiprocessing.Pool(args.workers) as pool:
            for res in tqdm(pool.imap_unordered(process_one_sample, samples), total=len(samples)):
                results[res["status"]] += 1
                if res["status"] == "Error":
                    failed_list.append(res)

    print("\n=== Generation finished ===")
    print(f"Successfully generated: {results['Success']}")
    print(f"Skipped (already exists): {results['Exists']}")
    print(f"Failed: {results['Error']}")

    if failed_list:
        pd.DataFrame(failed_list).to_csv(FAILED_CSV, index=False)
        print(f"Failed sample details saved: {FAILED_CSV}")

    print(f"Detailed log: {LOG_FILE}")
    logging.info(
        f"SUMMARY | Success={results['Success']} | Exists={results['Exists']} | Error={results['Error']}"
    )


if __name__ == "__main__":
    main()
