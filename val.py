import os
import re
import sys
import glob
import warnings
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from config import (
    GPU_ID,
    NODE_IN_DIM,
    ESM_DIM,
    BATCH_SIZE,
    HIDDEN_DIM,
    N_LAYERS,
    DROPOUT,
    DATALOADER_WORKERS,
    PIN_MEMORY,
    USE_ESM_IN_MODEL,
    USE_MUTATION_LOCAL_POOLING,
    LOCAL_POOL_FALLBACK,
    USE_STRUCTURE_ATTENTION,
)
from egnn_model import LocalMutationGNN

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

FIXED_EFFECTIVE_RADIUS = 15.0
FIXED_EFFECTIVE_EDGE_CUTOFF = 10.0


def infer_device(device: str) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ExternalDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        graph_dir: str,
        effective_radius: float = FIXED_EFFECTIVE_RADIUS,
        effective_edge_cutoff: float = FIXED_EFFECTIVE_EDGE_CUTOFF,
    ):
        self.df = df.reset_index(drop=True)
        self.graph_dir = graph_dir
        self.effective_radius = float(effective_radius)
        self.effective_edge_cutoff = float(effective_edge_cutoff)

        self.mask_dssp = getattr(cfg, "MASK_DSSP", False)
        self.mask_esm = getattr(cfg, "MASK_ESM", False)
        self.mask_base_only = getattr(cfg, "MASK_BASE_ONLY", False)

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _get_sample_id(row: pd.Series) -> str:
        for col in ["graph_id", "ID", "PDB"]:
            if col in row.index and pd.notna(row[col]):
                return str(row[col]).strip()
        raise KeyError("CSV missing sample ID column (graph_id / ID / PDB)")

    @staticmethod
    def _get_label(row: pd.Series) -> Optional[float]:
        for col in ["ddG", "ddg", "label"]:
            if col in row.index and pd.notna(row[col]):
                return float(row[col])
        return None

    def _apply_fixed_subgraph_crop(self, data):
        if not (
            hasattr(data, "node_dist_to_mut")
            and hasattr(data, "edge_dist")
            and data.node_dist_to_mut is not None
            and data.edge_dist is not None
        ):
            return data

        node_mask = data.node_dist_to_mut <= self.effective_radius
        if int(node_mask.sum()) == 0:
            min_idx = int(torch.argmin(data.node_dist_to_mut).item())
            node_mask = torch.zeros_like(data.node_dist_to_mut, dtype=torch.bool)
            node_mask[min_idx] = True

        old_to_new = torch.full((data.x.size(0),), -1, dtype=torch.long)
        kept_nodes = torch.nonzero(node_mask, as_tuple=False).view(-1)
        old_to_new[kept_nodes] = torch.arange(kept_nodes.numel(), dtype=torch.long)

        row_idx, col_idx = data.edge_index
        edge_mask = (
            (data.edge_dist <= self.effective_edge_cutoff)
            & node_mask[row_idx]
            & node_mask[col_idx]
        )

        data.x = data.x[node_mask]
        data.pos = data.pos[node_mask]
        data.node_dist_to_mut = data.node_dist_to_mut[node_mask]

        if int(edge_mask.sum()) > 0:
            kept_row = row_idx[edge_mask]
            kept_col = col_idx[edge_mask]
            data.edge_index = torch.stack(
                [old_to_new[kept_row], old_to_new[kept_col]], dim=0
            )
            data.edge_dist = data.edge_dist[edge_mask]
        else:
            data.edge_index = torch.zeros((2, 0), dtype=torch.long)
            data.edge_dist = torch.zeros((0,), dtype=torch.float32)

        return data

    def _apply_feature_ablation_mask(self, data):
        if not hasattr(data, "x") or data.x is None:
            return data

        x = data.x.clone()

        # Base = 0:41, DSSP = 41:55, DeltaESM = 55:
        if self.mask_dssp and x.size(1) >= 55:
            x[:, 41:55] = 0.0
        if self.mask_esm and x.size(1) > 55:
            x[:, 55:] = 0.0
        if self.mask_base_only and x.size(1) > 41:
            x[:, 41:] = 0.0

        data.x = x
        return data

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sample_id = self._get_sample_id(row)

        pt_path = os.path.join(self.graph_dir, f"{sample_id}.pt")
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Missing graph file: {pt_path}")

        # PyTorch>=2.6 defaults weights_only=True, which cannot deserialize PyG Data
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        data = self._apply_fixed_subgraph_crop(data)
        data = self._apply_feature_ablation_mask(data)

        y = self._get_label(row)
        if y is not None:
            data.y = torch.tensor([y], dtype=torch.float32)
            data.has_label = torch.tensor([1], dtype=torch.long)
        else:
            data.y = torch.tensor([0.0], dtype=torch.float32)
            data.has_label = torch.tensor([0], dtype=torch.long)

        data.sample_id = sample_id
        if "PDB" in row.index and pd.notna(row["PDB"]):
            data.pdb = str(row["PDB"]).strip()

        return data


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    if len(y_true) < 2 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
        if np.isnan(pearson) or np.isinf(pearson):
            pearson = 0.0

    return {"RMSE": rmse, "MAE": mae, "Pearson": pearson}


@torch.no_grad()
def evaluate_external(
    model: LocalMutationGNN,
    loader: DataLoader,
    ddg_mean: float,
    ddg_std: float,
    device: torch.device,
):
    model.eval()

    sample_ids = []
    y_all = []
    pred_all = []
    has_label_all = []

    for batch in loader:
        batch = batch.to(device)
        y_raw = batch.y.view(-1)

        pred_norm = model(batch)
        pred_raw = pred_norm * ddg_std + ddg_mean

        y_all.append(y_raw.detach().cpu().numpy())
        pred_all.append(pred_raw.detach().cpu().numpy())

        if hasattr(batch, "has_label") and batch.has_label is not None:
            has_label_all.append(batch.has_label.view(-1).detach().cpu().numpy())
        else:
            has_label_all.append(np.ones(int(y_raw.size(0)), dtype=np.int64))

        batch_sample_ids = getattr(batch, "sample_id", None)
        if isinstance(batch_sample_ids, (list, tuple)):
            sample_ids.extend([str(x) for x in batch_sample_ids])
        else:
            sample_ids.extend([""] * int(y_raw.size(0)))

    y_all = np.concatenate(y_all, axis=0).reshape(-1)
    pred_all = np.concatenate(pred_all, axis=0).reshape(-1)
    has_label_all = np.concatenate(has_label_all, axis=0).reshape(-1).astype(bool)

    pred_df = pd.DataFrame(
        {
            "ID": sample_ids,
            "True_ddG": y_all,
            "Pred_ddG": pred_all,
            "AbsErr": np.abs(y_all - pred_all),
            "HasLabel": has_label_all.astype(int),
        }
    )

    if has_label_all.any():
        metrics = calc_metrics(y_all[has_label_all], pred_all[has_label_all])
    else:
        metrics = {"RMSE": np.nan, "MAE": np.nan, "Pearson": np.nan}

    return metrics, pred_df


def build_model(device: torch.device) -> LocalMutationGNN:
    model = LocalMutationGNN(
        node_in_dim=NODE_IN_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=N_LAYERS,
        dropout=DROPOUT,
        esm_dim=ESM_DIM,
        use_esm_in_model=USE_ESM_IN_MODEL,
        use_mutation_local_pooling=USE_MUTATION_LOCAL_POOLING,
        local_pool_fallback=LOCAL_POOL_FALLBACK,
        use_structure_attention=USE_STRUCTURE_ATTENTION,
    ).to(device)
    return model


def infer_ddg_stats_from_train_csv(train_csv: str) -> Tuple[float, float]:
    if not os.path.isfile(train_csv):
        raise FileNotFoundError(f"Training-fold CSV for denormalization not found: {train_csv}")

    train_df = pd.read_csv(train_csv)
    if "ddG" in train_df.columns:
        y_train = train_df["ddG"].astype(float).values
    else:
        y_train = train_df.iloc[:, -1].astype(float).values

    ddg_mean = float(np.mean(y_train))
    ddg_std = float(np.std(y_train) + 1e-8)
    return ddg_mean, ddg_std


def resolve_train_fold_csv(
    source_dataset: Optional[str],
    split_type: Optional[str],
    fold: Optional[int],
    project_root: Optional[str] = None,
) -> Optional[str]:
    if source_dataset is None or split_type is None or fold is None:
        return None
    root = project_root or cfg.PROJECT_ROOT
    return os.path.join(
        root,
        str(source_dataset),
        str(split_type),
        f"train_fold_{int(fold)}.csv",
    )


def load_checkpoint(
    model: LocalMutationGNN,
    model_path: str,
    device: torch.device,
    train_csv: Optional[str] = None,
) -> Tuple[dict, float, float]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
        state_dict = checkpoint
    else:
        raise KeyError("Model parameters not found in checkpoint (model_state_dict / model / plain state_dict)")

    # Compatibility: drop leftover classification-head params from older weights
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith("cls_head.")
    }

    load_msg = model.load_state_dict(filtered_state_dict, strict=False)

    missing = [k for k in load_msg.missing_keys if not k.startswith("cls_head.")]
    unexpected = [k for k in load_msg.unexpected_keys if not k.startswith("cls_head.")]

    if missing:
        raise RuntimeError(f"Missing non-cls_head parameters when loading checkpoint: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected non-cls_head parameters when loading checkpoint: {unexpected}")

    has_stats = (
        isinstance(checkpoint, dict)
        and ("ddg_mean" in checkpoint)
        and ("ddg_std" in checkpoint)
    )

    if has_stats:
        ddg_mean = float(checkpoint["ddg_mean"])
        ddg_std = float(checkpoint["ddg_std"])
        stats_source = "checkpoint"
    elif train_csv is not None:
        ddg_mean, ddg_std = infer_ddg_stats_from_train_csv(train_csv)
        stats_source = f"train_csv:{train_csv}"
    else:
        raise RuntimeError(
            "checkpoint missing ddg_mean/ddg_std, and neither train_csv nor "
            "--source_dataset + --split_type was provided. Cannot correctly denormalize predictions."
        )

    if abs(ddg_std) < 1e-12:
        ddg_std = 1.0

    print(f"ddG stats source: {stats_source}")
    return checkpoint, ddg_mean, ddg_std


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def parse_run_fold(model_filename: str) -> Tuple[Optional[int], Optional[int]]:
    base = os.path.basename(model_filename)
    m = re.search(r"run(\d+)_fold(\d+)_best\.pt$", base)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"fold[_-]?(\d+)_best\.pt$", base)
    if m:
        return None, int(m.group(1))
    return None, None


def load_model_and_validate(
    model_path: str,
    csv_path: str,
    graph_dir: str,
    output_dir: Optional[str] = None,
    device: str = "cuda",
    train_csv: Optional[str] = None,
    source_dataset: Optional[str] = None,
    split_type: Optional[str] = None,
) -> Dict[str, float]:
    dev = infer_device(device)
    print(f"Using device: {dev}")
    print(f"Fixed subgraph crop: radius={FIXED_EFFECTIVE_RADIUS}, edge_cutoff={FIXED_EFFECTIVE_EDGE_CUTOFF}")
    print("Validation mode: pure regression (final synced version)")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not os.path.isdir(graph_dir):
        raise FileNotFoundError(f"Graph data directory not found: {graph_dir}")

    df = pd.read_csv(csv_path)
    print(f"Validation samples: {len(df)}")

    _, fold_id = parse_run_fold(os.path.basename(model_path))
    resolved_train_csv = train_csv
    if resolved_train_csv is None:
        resolved_train_csv = resolve_train_fold_csv(
            source_dataset=source_dataset,
            split_type=split_type,
            fold=fold_id,
        )

    model = build_model(dev)
    checkpoint, ddg_mean, ddg_std = load_checkpoint(
        model,
        model_path,
        dev,
        train_csv=resolved_train_csv,
    )
    print(f"ddG stats: mean={ddg_mean:.4f}, std={ddg_std:.4f}")

    if isinstance(checkpoint, dict):
        if "epoch" in checkpoint:
            print(f"Training epoch: {checkpoint['epoch']}")
        if "run_idx" in checkpoint:
            print(f"run_idx: {checkpoint['run_idx']}")
        if "fold" in checkpoint:
            print(f"fold: {checkpoint['fold']}")
        if "ablation_mode" in checkpoint:
            print(f"ablation_mode: {checkpoint['ablation_mode']}")

    dataset = ExternalDataset(df=df, graph_dir=graph_dir)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=DATALOADER_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    print("\nStarting evaluation...")
    metrics, predictions_df = evaluate_external(
        model=model,
        loader=loader,
        ddg_mean=ddg_mean,
        ddg_std=ddg_std,
        device=dev,
    )

    print("\n" + "=" * 68)
    print("External validation results (pure regression)")
    print("=" * 68)
    if np.isnan(metrics["RMSE"]):
        print("Current CSV has no valid labels; exported predictions only, skipped RMSE/MAE/Pearson.")
    else:
        print(f"RMSE:           {metrics['RMSE']:.4f}")
        print(f"MAE:            {metrics['MAE']:.4f}")
        print(f"Pearson correlation: {metrics['Pearson']:.4f}")
    print("=" * 68)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        pred_csv_path = os.path.join(output_dir, "external_predictions.csv")
        metrics_txt_path = os.path.join(output_dir, "external_metrics.txt")

        predictions_df.to_csv(pred_csv_path, index=False)
        print(f"Predictions saved to: {pred_csv_path}")

        with open(metrics_txt_path, "w", encoding="utf-8") as f:
            f.write("External validation results (pure regression)\n")
            f.write("=" * 68 + "\n")
            f.write(f"Model path: {model_path}\n")
            f.write(f"Validation CSV: {csv_path}\n")
            f.write(f"Graph data dir: {graph_dir}\n")
            f.write(f"Samples: {len(df)}\n")
            f.write(
                f"Fixed subgraph crop: radius={FIXED_EFFECTIVE_RADIUS}, "
                f"edge_cutoff={FIXED_EFFECTIVE_EDGE_CUTOFF}\n"
            )
            f.write("Validation mode: pure regression (final synced version)\n")
            f.write("-" * 68 + "\n")

            if np.isnan(metrics["RMSE"]):
                f.write("Current CSV has no valid labels; exported predictions only, skipped RMSE/MAE/Pearson.\n")
            else:
                f.write(f"RMSE: {metrics['RMSE']:.4f}\n")
                f.write(f"MAE: {metrics['MAE']:.4f}\n")
                f.write(f"Pearson: {metrics['Pearson']:.4f}\n")

            f.write("=" * 68 + "\n")

        print(f"Metrics saved to: {metrics_txt_path}")

    return {
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "Pearson": metrics["Pearson"],
    }


def find_model_paths(model_dir: str, pattern: str) -> List[str]:
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model_paths = sorted(glob.glob(os.path.join(model_dir, pattern)))
    if not model_paths:
        raise FileNotFoundError(
            f"No weight files matching {pattern} found under {model_dir}"
        )
    return model_paths


def validate_all_models(
    model_dir: str,
    csv_path: str,
    graph_dir: str,
    output_dir: str,
    device: str = "cuda",
    pattern: str = "run*_fold*_best.pt",
    source_dataset: Optional[str] = None,
    split_type: Optional[str] = None,
):
    model_paths = find_model_paths(model_dir, pattern)
    print(f"Found {len(model_paths)} weight files")

    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []

    for idx, model_path in enumerate(model_paths, start=1):
        model_name = os.path.basename(model_path)
        run_id, fold_id = parse_run_fold(model_name)
        model_output_dir = os.path.join(
            output_dir,
            sanitize_name(os.path.splitext(model_name)[0]),
        )

        print("\n" + "#" * 80)
        print(f"[{idx}/{len(model_paths)}] Current weights: {model_name}")
        print(f"Output dir: {model_output_dir}")
        print("#" * 80)

        try:
            metrics = load_model_and_validate(
                model_path=model_path,
                csv_path=csv_path,
                graph_dir=graph_dir,
                output_dir=model_output_dir,
                device=device,
                source_dataset=source_dataset,
                split_type=split_type,
            )
            summary_rows.append(
                {
                    "model_name": model_name,
                    "model_path": model_path,
                    "run": run_id,
                    "fold": fold_id,
                    "RMSE": metrics["RMSE"],
                    "MAE": metrics["MAE"],
                    "Pearson": metrics["Pearson"],
                    "status": "ok",
                }
            )
        except Exception as e:
            print(f"[ERROR] {model_name} evaluation failed: {e}")
            summary_rows.append(
                {
                    "model_name": model_name,
                    "model_path": model_path,
                    "run": run_id,
                    "fold": fold_id,
                    "RMSE": np.nan,
                    "MAE": np.nan,
                    "Pearson": np.nan,
                    "status": f"failed: {e}",
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty and "RMSE" in summary_df.columns:
        ok_df = summary_df[summary_df["status"] == "ok"].copy()
        if not ok_df.empty:
            ok_df = ok_df.sort_values(
                by=["RMSE", "MAE", "Pearson"],
                ascending=[True, True, False],
            )
            summary_df = pd.concat(
                [ok_df, summary_df[summary_df["status"] != "ok"]],
                axis=0,
                ignore_index=True,
            )

    summary_csv = os.path.join(output_dir, "all_model_metrics_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print("\n" + "=" * 80)
    print(f"All model evaluations finished; summary saved to: {summary_csv}")
    print("=" * 80)

    return summary_df


def main():
    import argparse

    # Default: evaluate VAL_DATASET with weights trained on TRAIN_DATASET
    train_dataset = os.environ.get("TRAIN_DATASET", "S1131")
    val_dataset = os.environ.get("VAL_DATASET", "ATLAS")
    train_run_root = os.path.join(cfg.PROJECT_ROOT, f"{train_dataset}_{cfg.RUN_TAG}")
    val_data_dir = os.path.join(cfg.PROJECT_ROOT, val_dataset)

    parser = argparse.ArgumentParser(
        description="External validation batch eval script (final pure-regression synced version; supports multiple weights)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a single model weight file; if set, evaluate only this model",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.path.join(train_run_root, "model"),
        help="Model directory for batch reading run*_fold*_best.pt",
    )
    parser.add_argument(
        "--model_pattern",
        type=str,
        default="run*_fold*_best.pt",
        help="Glob pattern for batch-matching model files",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=os.path.join(val_data_dir, f"{val_dataset}.csv"),
        help="Validation set CSV path",
    )
    parser.add_argument(
        "--graph_dir",
        type=str,
        default=os.path.join(val_data_dir, "graph", "fusion"),
        help="Validation set graph data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(val_data_dir, f"{train_dataset}_results"),
        help="Output directory (one subdirectory per model, plus overall summary table)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device",
    )
    parser.add_argument(
        "--source_dataset",
        type=str,
        default=None,
        help="Source training dataset of the weights, e.g. M1101 / S1131; used to locate train_fold_*.csv when checkpoint lacks ddg stats",
    )
    parser.add_argument(
        "--split_type",
        type=str,
        default=None,
        help="Training split type, e.g. Struc / Mpb / Fam",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default=None,
        help="Explicitly specify a training-fold CSV (useful for single-model eval); batch eval prefers auto-match via source_dataset+split_type",
    )

    args = parser.parse_args()

    print("=" * 68)
    print("External validation (final pure-regression synced version; supports batch weights)")
    print("=" * 68)
    print(f"Single model path: {args.model_path}")
    print(f"Model dir:         {args.model_dir}")
    print(f"Match pattern:     {args.model_pattern}")
    print(f"Validation CSV:    {args.csv_path}")
    print(f"Graph data dir:    {args.graph_dir}")
    print(f"Output dir:        {args.output_dir}")
    print(f"Device:            {args.device}")
    print(f"Source dataset:    {args.source_dataset}")
    print(f"Split type:        {args.split_type}")
    print(f"Train CSV:         {args.train_csv}")
    print("=" * 68)

    if args.model_path is not None and str(args.model_path).strip() != "":
        single_output_dir = os.path.join(
            args.output_dir,
            sanitize_name(os.path.splitext(os.path.basename(args.model_path))[0]),
        )
        metrics = load_model_and_validate(
            model_path=args.model_path,
            csv_path=args.csv_path,
            graph_dir=args.graph_dir,
            output_dir=single_output_dir,
            device=args.device,
            train_csv=args.train_csv,
            source_dataset=args.source_dataset,
            split_type=args.split_type,
        )
        print("\nEvaluation finished!")
        print(metrics)
    else:
        validate_all_models(
            model_dir=args.model_dir,
            csv_path=args.csv_path,
            graph_dir=args.graph_dir,
            output_dir=args.output_dir,
            device=args.device,
            pattern=args.model_pattern,
            source_dataset=args.source_dataset,
            split_type=args.split_type,
        )
        print("\nBatch evaluation finished!")


if __name__ == "__main__":
    main()
