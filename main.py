import os
import sys

# CUDA_VISIBLE_DEVICES must be set before importing torch, else it falls onto physical GPU 0
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU_ID)

import time
import random
import warnings
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from config import (
    GPU_ID, DATASET, DATA_DIR, N_LAYERS,
    FUSION_GRAPH_DIR, TRAIN_FOLD_CSV, VAL_FOLD_CSV, N_FOLDS,
    NODE_IN_DIM, ESM_DIM,
    BATCH_SIZE, GRAD_ACCUM_STEPS, LR, EPOCHS, PATIENCE,
    HIDDEN_DIM, DROPOUT, WEIGHT_DECAY,
    SEED, DATALOADER_WORKERS, PIN_MEMORY, DEVICE,
    LOSS_ALPHA_MAX, ALPHA_WARMUP_EPOCHS,
    SCORE_LAMBDA, USE_STRUCTURE_ATTENTION,
    ENABLE_EARLY_STOP, TYPETIMES, LOG_DIR, SAVE_MODEL_WEIGHTS,
    USE_ESM_IN_MODEL, USE_MUTATION_LOCAL_POOLING, LOCAL_POOL_FALLBACK,
    EFFECTIVE_RADIUS, EFFECTIVE_EDGE_CUTOFF,
)

from egnn_model import LocalMutationGNN, mixed_ddg_loss

warnings.filterwarnings("ignore")


def _device():
    return torch.device(DEVICE) if isinstance(DEVICE, str) else DEVICE


DEV = _device()


def _resolve_amp():
    """Return (whether AMP is enabled, autocast dtype). BF16 needs CUDA + hardware support."""
    want_bf16 = bool(getattr(cfg, "USE_BF16", False))
    if not want_bf16:
        return False, torch.float32
    if DEV.type != "cuda" or not torch.cuda.is_available():
        print("[AMP] USE_BF16=True but not on CUDA; falling back to FP32")
        return False, torch.float32
    if not torch.cuda.is_bf16_supported():
        print("[AMP] Current GPU does not support BF16; falling back to FP32")
        return False, torch.float32
    print("[AMP] Enabling BF16 mixed-precision training")
    return True, torch.bfloat16


USE_AMP, AMP_DTYPE = _resolve_amp()


def amp_context():
    return torch.autocast(
        device_type=DEV.type,
        dtype=AMP_DTYPE,
        enabled=USE_AMP,
    )


def _print_gpu_binding():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if DEV.type != "cuda" or not torch.cuda.is_available():
        print(f"[GPU] CUDA_VISIBLE_DEVICES={visible} | CUDA not in use")
        return
    # After setting CUDA_VISIBLE_DEVICES, the process always sees cuda:0 mapping to physical GPU_ID
    print(
        f"[GPU] Physical GPU_ID={GPU_ID} | CUDA_VISIBLE_DEVICES={visible} | "
        f"logical device=cuda:0 | name={torch.cuda.get_device_name(0)}"
    )


_print_gpu_binding()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_best_model(
    model: nn.Module,
    run_idx: int,
    fold: int,
    save_dir: str,
    split_type: str,
    ddg_mean: float = 0.0,
    ddg_std: float = 1.0,
    best_epoch: int = 0,
):
    """Save best weights per run/fold (includes ddG denorm stats for val.py)."""
    if not SAVE_MODEL_WEIGHTS:
        return

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"run{run_idx}_fold{fold}_best.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "run_idx": run_idx,
            "fold": fold,
            "epoch": int(best_epoch),
            "dataset": DATASET,
            "split_type": split_type,
            "ablation_mode": getattr(cfg, "ABLATION_MODE", "unknown"),
            "ddg_mean": float(ddg_mean),
            "ddg_std": float(ddg_std),
        },
        save_path,
    )
    print(f"[Save] Best model -> {save_path}")


class FusionDataset(Dataset):
    """Crop precomputed graphs by radius/edge cutoff and zero feature blocks per ablation flags."""

    def __init__(
        self,
        df: pd.DataFrame,
        graph_dir: str,
        effective_radius: float = EFFECTIVE_RADIUS,
        effective_edge_cutoff: float = EFFECTIVE_EDGE_CUTOFF,
    ):
        self.df = df.reset_index(drop=True)
        self.graph_dir = graph_dir
        self.mask_dssp = getattr(cfg, "MASK_DSSP", False)
        self.mask_esm = getattr(cfg, "MASK_ESM", False)
        self.mask_base_only = getattr(cfg, "MASK_BASE_ONLY", False)
        self.effective_radius = float(effective_radius)
        self.effective_edge_cutoff = float(effective_edge_cutoff)

    def __len__(self):
        return len(self.df)

    def _crop_subgraph(self, data):
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
            node_mask[min_idx] = True

        old_to_new = torch.full((data.x.size(0),), -1, dtype=torch.long)
        kept_idx = torch.nonzero(node_mask, as_tuple=False).view(-1)
        old_to_new[kept_idx] = torch.arange(kept_idx.numel(), dtype=torch.long)

        row_idx, col_idx = data.edge_index
        edge_mask = (
            (data.edge_dist <= self.effective_edge_cutoff)
            & node_mask[row_idx]
            & node_mask[col_idx]
        )

        data.x = data.x[node_mask]
        data.pos = data.pos[node_mask]
        data.node_dist_to_mut = data.node_dist_to_mut[node_mask]

        if edge_mask.any():
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

    def _apply_feature_mask(self, data):
        if not hasattr(data, "x") or data.x is None:
            return data

        x = data.x.clone()

        # Base = 0:41, DSSP = 41:55, DeltaESM = 55:
        if self.mask_dssp:
            x[:, 41:55] = 0.0
        if self.mask_esm:
            x[:, 55:] = 0.0
        if self.mask_base_only:
            x[:, 41:] = 0.0

        data.x = x
        return data

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        gid = str(row.get("graph_id", row.get("ID", row.get("PDB", idx))))
        graph_path = os.path.join(self.graph_dir, f"{gid}.pt")
        data = torch.load(graph_path, map_location="cpu", weights_only=False)

        data = self._crop_subgraph(data)
        data = self._apply_feature_mask(data)

        if "ddG" in row.index:
            y = float(row["ddG"])
        elif "ddg" in row.index:
            y = float(row["ddg"])
        else:
            y = float(row.iloc[-1])

        data.y = torch.tensor([y], dtype=torch.float32)
        data.sample_id = str(row.get("ID", gid))
        if "PDB" in row.index:
            data.pdb = str(row["PDB"]).strip()

        return data


def calc_rmse_mae_p(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    p = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) >= 2 else 0.0
    if np.isnan(p) or np.isinf(p):
        p = 0.0
    return rmse, mae, p


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, ddg_mean: float, ddg_std: float):
    model.eval()
    ys_raw, preds_raw = [], []

    for batch in loader:
        batch = batch.to(DEV)
        y_raw = batch.y.view(-1)

        with amp_context():
            pred_norm = model(batch)
        pred_raw = pred_norm.float() * ddg_std + ddg_mean

        ys_raw.append(y_raw.detach().cpu().numpy())
        preds_raw.append(pred_raw.detach().cpu().numpy())

    y_true = np.concatenate(ys_raw).reshape(-1)
    y_pred = np.concatenate(preds_raw).reshape(-1)
    rmse, mae, p = calc_rmse_mae_p(y_true, y_pred)

    return rmse, mae, p


def save_fold_predictions(
    model: nn.Module,
    loader: DataLoader,
    val_df: pd.DataFrame,
    ddg_mean: float,
    ddg_std: float,
    save_path: str,
):
    """Save predictions in val_df original order, aligned with validation CSV row order."""
    model.eval()
    rows = []
    all_true, all_pred = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEV)
            y_raw = batch.y.view(-1)
            with amp_context():
                pred_norm = model(batch)
            pred_raw = pred_norm.float() * ddg_std + ddg_mean

            all_true.append(y_raw.detach().cpu().numpy())
            all_pred.append(pred_raw.detach().cpu().numpy())

    y_true_all = np.concatenate(all_true).reshape(-1)
    y_pred_all = np.concatenate(all_pred).reshape(-1)

    for idx, row in val_df.iterrows():
        sample_id = str(row.get("ID", row.get("PDB", idx)))
        rows.append(
            {
                "ID": sample_id,
                "true_ddG": float(y_true_all[idx]),
                "pred_ddG": float(y_pred_all[idx]),
            }
        )

    pd.DataFrame(rows).to_csv(save_path, index=False)


def run_fold(
    fold: int,
    run_idx: int,
    split_type: str,
    split_log_dir: str,
    log_file=None,
) -> Dict[str, Any]:
    def log_print(msg: str):
        print(msg)
        if log_file is not None:
            log_file.write(msg + "\n")
            log_file.flush()

    train_csv = TRAIN_FOLD_CSV.format(split_type=split_type, fold=fold)
    val_csv = VAL_FOLD_CSV.format(split_type=split_type, fold=fold)

    for csv_path in (train_csv, val_csv):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Missing 5-fold split file: {csv_path}")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    y_train = (
        train_df["ddG"].astype(float).values
        if "ddG" in train_df.columns
        else train_df.iloc[:, -1].astype(float).values
    )
    ddg_mean = float(np.mean(y_train))
    ddg_std = float(np.std(y_train) + 1e-8)

    log_print(
        f"Split={split_type} | Fold {fold} | "
        f"Train {len(train_df)} | Val {len(val_df)} | "
        f"mean={ddg_mean:.4f} std={ddg_std:.4f} | "
        f"Radius={EFFECTIVE_RADIUS} EdgeCutoff={EFFECTIVE_EDGE_CUTOFF}"
    )

    train_set = FusionDataset(train_df, FUSION_GRAPH_DIR)
    val_set = FusionDataset(val_df, FUSION_GRAPH_DIR)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=DATALOADER_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=DATALOADER_WORKERS,
        pin_memory=PIN_MEMORY,
    )

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
    ).to(DEV)

    optimizer = AdamW(
        model.parameters(),
        lr=float(LR),
        weight_decay=float(WEIGHT_DECAY),
    )

    best_score = -1e9
    best_epoch = 0
    best_state = None
    patience_counter = 0

    best_rmse = best_mae = best_p = 0.0

    warmup_epochs = (
        ALPHA_WARMUP_EPOCHS
        if ALPHA_WARMUP_EPOCHS is not None
        else max(1, int(EPOCHS) * 3 // 10)
    )

    for epoch in range(1, int(EPOCHS) + 1):
        if epoch <= int(warmup_epochs):
            current_alpha = float(LOSS_ALPHA_MAX) * (epoch - 1) / max(int(warmup_epochs) - 1, 1)
        else:
            current_alpha = float(LOSS_ALPHA_MAX)

        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        n_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            batch = batch.to(DEV)
            y_raw = batch.y.view(-1)
            y_norm = (y_raw - ddg_mean) / ddg_std

            with amp_context():
                pred_norm = model(batch)
                loss, huber_loss, pearson_loss = mixed_ddg_loss(
                    pred_norm, y_norm, alpha=current_alpha
                )

            loss = loss / max(1, int(GRAD_ACCUM_STEPS))
            loss.backward()

            if step % int(GRAD_ACCUM_STEPS) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(loss.detach().float().cpu().item())
            n_steps += 1

        if n_steps > 0 and (n_steps % int(GRAD_ACCUM_STEPS) != 0):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        val_rmse, val_mae, val_p = evaluate(
            model, val_loader, ddg_mean, ddg_std
        )

        score = val_p - float(SCORE_LAMBDA) * val_rmse
        dt = time.time() - t0

        log_print(
            f"Epoch {epoch:03d} | Time {dt:.1f}s | "
            f"Loss {epoch_loss / max(n_steps, 1):.4f} | "
            f"RMSE {val_rmse:.4f} | MAE {val_mae:.4f} | P {val_p:.4f}"
        )

        if score > best_score + 1e-8:
            best_score = float(score)
            best_epoch = int(epoch)
            patience_counter = 0

            best_rmse = float(val_rmse)
            best_mae = float(val_mae)
            best_p = float(val_p)

            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            if ENABLE_EARLY_STOP:
                patience_counter += 1
                if patience_counter >= int(PATIENCE):
                    log_print(f"[EarlyStop] stop at epoch={epoch}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    log_print(
        f"[{split_type} | Fold {fold}] BEST @Epoch {best_epoch:03d} | "
        f"RMSE {best_rmse:.4f} | MAE {best_mae:.4f} | P {best_p:.4f}"
    )

    result_dir = os.path.join(split_log_dir, "predictions")
    os.makedirs(result_dir, exist_ok=True)
    pred_save_path = os.path.join(result_dir, f"run{run_idx}_fold{fold}_predictions.csv")
    save_fold_predictions(model, val_loader, val_df, ddg_mean, ddg_std, pred_save_path)

    model_dir = os.path.join(split_log_dir, "weights")
    save_best_model(
        model,
        run_idx,
        fold,
        model_dir,
        split_type,
        ddg_mean=ddg_mean,
        ddg_std=ddg_std,
        best_epoch=best_epoch,
    )

    return {
        "RMSE": best_rmse,
        "MAE": best_mae,
        "Pearson": best_p,
        "BestEpoch": best_epoch,
    }


def main():
    try:
        num_repeats = int(TYPETIMES) if TYPETIMES is not None else 1
    except Exception:
        num_repeats = 1
    num_repeats = max(1, num_repeats)

    split_types = list(getattr(cfg, "SPLIT_TYPES", [getattr(cfg, "TYPE", "Mpb")]))
    if not split_types:
        raise ValueError("SPLIT_TYPES must not be empty.")

    allowed_split_types = {"Mpb", "Fam", "Struc"}
    invalid_split_types = [x for x in split_types if x not in allowed_split_types]
    if invalid_split_types:
        raise ValueError(
            f"Unknown data splits: {invalid_split_types}; allowed: {sorted(allowed_split_types)}"
        )

    os.makedirs(LOG_DIR, exist_ok=True)

    print(
        f"\n=== Pure regression training start | DATASET={DATASET} | "
        f"SPLITS={split_types} | "
        f"ABLATION_MODE={getattr(cfg, 'ABLATION_MODE', 'unknown')} | "
        f"Repeats={num_repeats} ==="
    )

    all_summaries = []

    for split_type in split_types:
        split_log_dir = os.path.join(LOG_DIR, split_type)
        os.makedirs(split_log_dir, exist_ok=True)

        print(f"\n{'=' * 80}")
        print(f"Starting data split: {split_type}")
        print(f"Split dir: {os.path.join(DATA_DIR, split_type)}")
        print(f"Output dir: {split_log_dir}")
        print(f"{'=' * 80}")

        for run_idx in range(num_repeats):
            run_seed = SEED + run_idx * 137
            set_seed(run_seed)

            run_log_dir = os.path.join(
                split_log_dir,
                f"repeat_{run_idx}_seed_{run_seed}",
            )
            os.makedirs(run_log_dir, exist_ok=True)
            log_path = os.path.join(run_log_dir, f"log_repeat{run_idx}.txt")

            with open(log_path, "w", encoding="utf-8") as log_file:
                def log_print(msg: str):
                    print(msg)
                    log_file.write(msg + "\n")
                    log_file.flush()

                log_print(
                    f"Split={split_type} | "
                    f"Repeat {run_idx + 1}/{num_repeats} | SEED={run_seed}"
                )

                fold_results = []
                for fold in range(int(N_FOLDS)):
                    log_print(f"\n>>> {split_type} | Fold {fold} <<<")
                    fold_result = run_fold(
                        fold=fold,
                        run_idx=run_idx,
                        split_type=split_type,
                        split_log_dir=split_log_dir,
                        log_file=log_file,
                    )
                    fold_results.append(fold_result)

                rmses = [r["RMSE"] for r in fold_results]
                maes = [r["MAE"] for r in fold_results]
                ps = [r["Pearson"] for r in fold_results]

                summary = {
                    "dataset": DATASET,
                    "split_type": split_type,
                    "run_idx": run_idx,
                    "seed": run_seed,
                    "avg_RMSE": float(np.mean(rmses)),
                    "std_RMSE": float(np.std(rmses, ddof=1)),
                    "avg_MAE": float(np.mean(maes)),
                    "std_MAE": float(np.std(maes, ddof=1)),
                    "avg_Pearson": float(np.mean(ps)),
                    "std_Pearson": float(np.std(ps, ddof=1)),
                }
                all_summaries.append(summary)

                log_print("\n========== 5-Fold Summary ==========")
                log_print(f"DATASET = {DATASET}")
                log_print(f"SPLIT = {split_type}")
                log_print(f"SEED = {run_seed}")
                log_print(f"Avg RMSE:    {summary['avg_RMSE']:.4f} (±{summary['std_RMSE']:.4f})")
                log_print(f"Avg MAE:     {summary['avg_MAE']:.4f} (±{summary['std_MAE']:.4f})")
                log_print(f"Avg Pearson: {summary['avg_Pearson']:.4f} (±{summary['std_Pearson']:.4f})")

    summary_path = os.path.join(LOG_DIR, "all_split_summaries.csv")
    pd.DataFrame(all_summaries).to_csv(summary_path, index=False)

    print(f"\n=== All done! Logs saved under {LOG_DIR} ===")
    print(f"All-split summary table: {summary_path}")


if __name__ == "__main__":
    main()
