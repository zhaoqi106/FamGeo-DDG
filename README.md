# Predicting Mutation ΔΔG in Protein Complexes with a Local Geometric Graph Neural Network

Predicts binding free-energy changes (ΔΔG) induced by mutations in protein–protein complexes. The model builds a local geometric graph centered on the mutation site. Node features are **Base(41) + DSSP(14) + DeltaESM(2560) = 2615** dimensions.

This repository provides the full source for **FoldX structure generation, feature construction, training, evaluation, and analysis**. Large data files, ESM weights, and the FoldX binary are **not shipped with the repo**; prepare them as described below.

---

## Contents

1. [Environment setup](#1-environment-setup)
2. [Repository layout](#2-repository-layout)
3. [Data directory convention](#3-data-directory-convention)
4. [Third-party dependencies](#4-third-party-dependencies)
5. [End-to-end pipeline](#5-end-to-end-pipeline)
6. [Training and evaluation](#6-training-and-evaluation)
7. [Ablation switches](#7-ablation-switches)
8. [Environment variables](#8-environment-variables)
9. [Script reference](#9-script-reference)
10. [Reproducibility notes](#10-reproducibility-notes)

---

## 1. Environment setup

Recommended: Python ≥ 3.9 and a CUDA GPU (Ampere or newer can enable BF16; otherwise the code falls back to FP32).

```bash
cd MUT_liaoda
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

Main dependencies: `torch`, `torch-geometric`, `biopython`, `pandas`, `numpy`, `fair-esm`.

---

## 2. Repository layout

```text
MUT_liaoda/                 # CODE_ROOT (repository root)
├── config.py               # Default paths and hyperparameters (relative paths)
├── egnn_model.py           # Model and loss
├── graph_utils.py          # Graph construction / features
├── data_process.py         # Precompute fusion graphs (.pt)
├── main.py                 # Standard 5-fold training
├── val.py                  # External-set evaluation
├── foldxprocess.py         # FoldX RepairPDB (WT repair)
├── foldxpdb.py             # FoldX BuildModel (mutants)
├── DSSP.py                 # DSSP / ASA feature tables
├── split_byfamily.py       # Fam splits
├── build_family_map.py     # PDB family mapping
├── build_geo_igmi.py       # Struc splits (Geo/IGMI-style)
├── requirements.txt
├── README.md
├── bin/                    # Place FoldX and other binaries (not in git)
├── weights/                # Place ESM weights (not in git)
└── data/                   # Default data root (override with MUT_DATA_ROOT)
    ├── ecod_dict.ecod      # Not in git; required only for Struc; see below
    └── {DATASET}/
        ├── {DATASET}.csv
        ├── pdb/wt/
        ├── pdb/repairpdb/
        ├── pdb/mut/
        ├── foldx_txt/
        ├── DSSP/wt/
        ├── graph/fusion/
        ├── Mpb/
        ├── Fam/
        └── Struc/
```

---

## 3. Data directory convention

Default data root:

```text
./data/{DATASET}/
```

Global overrides:

```bash
export MUT_DATA_ROOT=/path/to/your/data
export DATASET=S4169
```

### Common CSV columns

| Column | Meaning |
|--------|---------|
| `ID` | Unique sample ID (mutant PDB / graph filename) |
| `PDB` | Wild-type complex ID |
| `Mutation` | Mutation string |
| `ddG` | Experimental label |
| `Partners` | Receptor_ligand chain definition (needed for family mapping, etc.) |

### Subdirectories

| Path | Contents |
|------|----------|
| `pdb/wt/` | Raw WT PDBs (Repair input) |
| `pdb/repairpdb/` | FoldX-repaired WT structures |
| `pdb/mut/` | Mutant structures `{ID}.pdb` |
| `foldx_txt/` | `individual_list_{ID}.txt` |
| `DSSP/wt/` | DSSP CSVs (used for graph construction) |
| `graph/fusion/` | Precomputed graphs `{ID}.pt` |
| `Mpb/` / `Fam/` / `Struc/` | `train_fold_k.csv` / `val_fold_k.csv` |

Default training / experiment outputs:

```text
./data/{DATASET}_runs/{model,result,log,splits}/
```

---

## 4. Third-party dependencies

### 4.1 FoldX (structure generation; important for review)

FoldX **cannot be redistributed with this repository**. Obtain a license and binary from the official site, then:

```bash
# Place at
./bin/foldx

# Or
export FOLDX_EXE=/absolute/path/to/foldx
```

### 4.2 mkdssp

```bash
# Ensure mkdssp is on PATH, or
export MKDSSP_EXE=/path/to/mkdssp
```

### 4.3 ESM-2 weights

```bash
mkdir -p weights
# Place:
# ./weights/esm2_t36_3B_UR50D.pt

export ESM_MODEL_PATH=/path/to/esm2_t36_3B_UR50D.pt   # optional
```

### 4.4 Optional

| Tool | Environment variables | Script |
|------|----------------------|--------|
| Rosetta pmut_scan | `ROSETTA_EXE`, `ROSETTA_DB` | `rosetta.py` |
| PSI-BLAST / SwissProt | `PSIBLAST_EXE`, `NR_DB_PATH` | `PSSM.py` |
| ECOD dictionary `ecod_dict.ecod` | `ECOD_DICT` | `build_geo_igmi.py` |

#### `ecod_dict.ecod` (required only for `Struc` splits)

This file is **not redistributed with this repository** (no redistribution rights). It comes from the [IGMI](https://github.com/ShiweiWu-545/IGMI) repository under the same name. Obtain it upstream and place it at `./data/ecod_dict.ecod`, or point to it with `ECOD_DICT`. If you already have `Struc` fold CSVs, or only use `Mpb` / `Fam` splits, you do not need this file.

---

## 5. End-to-end pipeline

Example with `DATASET=S4169`.

### A. FoldX RepairPDB (wild-type repair)

```bash
export DATASET=S4169
# Place raw WT PDBs under data/S4169/pdb/wt/
python foldxprocess.py --dataset S4169 --workers 10
```

Output: `data/S4169/pdb/repairpdb/*.pdb`

### B. FoldX BuildModel (mutants)

Prepare mutation lists:

```text
data/S4169/foldx_txt/individual_list_{ID}.txt
```

```bash
python foldxpdb.py --dataset S4169 --workers 16
```

Output: `data/S4169/pdb/mut/{ID}.pdb`

### C. DSSP features

```bash
python DSSP.py \
  --pdb_dir data/S4169/pdb/repairpdb \
  --out_dir data/S4169/DSSP/wt \
  --workers 16 --skip_existing
```

### D. Cross-validation splits

```bash
python build_family_map.py
python split_byfamily.py
# Obtain ecod_dict.ecod from IGMI yourself (not in git); see §4.4
python build_geo_igmi.py --datasets S4169
```

Place (or symlink) fold CSVs under:

```text
data/S4169/Mpb/
data/S4169/Fam/
data/S4169/Struc/
```

### E. Precompute graphs

```bash
export DATASET=S4169
export ESM_DEVICE=cpu
python data_process.py
```

Output: `data/S4169/graph/fusion/{ID}.pt`

### F. Training

```bash
export DATASET=S4169
export GPU_ID=0
export SPLIT_TYPES=Struc   # optional: run a single split type
python main.py
```

---

## 6. Training and evaluation

### Internal 5-fold: `main.py`

- Reads fold CSVs according to `SPLIT_TYPES`
- At training time, dynamically crops a local subgraph around the mutation (radius / edge cutoff in `config.py`)
- Metrics: RMSE / MAE / Pearson

### External evaluation: `val.py`

```bash
export TRAIN_DATASET=S1131
export VAL_DATASET=ATLAS
python val.py \
  --model_dir data/S1131_runs/model \
  --csv_path data/ATLAS/ATLAS.csv \
  --graph_dir data/ATLAS/graph/fusion \
  --output_dir data/ATLAS/S1131_results
```

---

## 7. Ablation switches

Change one line in `config.py`:

```python
ABLATION_MODE = "full"
```

| Mode | Meaning |
|------|---------|
| `full` | Full model (default) |
| `noDSSP` | Drop DSSP |
| `noESM` | Drop DeltaESM |
| `base_only` | Base 41-D features only |
| `no_local_pool` | Disable mutation local pooling |
| `no_pearson` | Huber loss only |
| `no_structure_attention` | Disable structure attention |

Input dimensionality remains 2615; ablations are implemented via zeroing / module switches so comparisons stay fair.

---

## 8. Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `MUT_DATA_ROOT` | `./data` | Data root directory |
| `DATASET` | `S4169` | Current dataset |
| `SPLIT_TYPES` | `Mpb,Fam,Struc` | Split types used by `main.py` |
| `GPU_ID` | `0` | GPU index |
| `ESM_MODEL_PATH` | `./weights/esm2_t36_3B_UR50D.pt` | ESM weights |
| `ESM_DEVICE` | `cpu` | Device for ESM feature extraction |
| `FOLDX_EXE` | `./bin/foldx` | FoldX path |
| `MKDSSP_EXE` | `mkdssp` | mkdssp path |
| `MUT_RUN_ROOT` | `./data/{DATASET}_runs` | Experiment output root |
| `MUT_RUN_TAG` | `runs` | Run directory suffix |
| `ECOD_DICT` | `./data/ecod_dict.ecod` | For Struc; file from [IGMI](https://github.com/ShiweiWu-545/IGMI), not in git |
| `ROSETTA_EXE` / `ROSETTA_DB` | `./bin/...` | Optional |
| `PSIBLAST_EXE` / `NR_DB_PATH` | PATH / `./weights/swissprot` | Optional |

---

## 9. Script reference

### Core for reproduction (suggested for GitHub)

| Script | Role |
|--------|------|
| `config.py` | Paths and hyperparameters |
| `foldxprocess.py` | FoldX RepairPDB |
| `foldxpdb.py` | FoldX BuildModel |
| `DSSP.py` | DSSP/ASA |
| `graph_utils.py` | Graph construction |
| `data_process.py` | Batch graph precomputation |
| `egnn_model.py` | Network and loss |
| `main.py` | Training |
| `val.py` | Evaluation |
| `build_family_map.py` / `split_byfamily.py` / `build_geo_igmi.py` | Split protocols (outputs under `Fam` / `Struc`, etc.) |

### Auxiliary scripts (optional)

`check.py`, `checkpdb.py`, `PSSM.py`, `rosetta.py`, `guiyin*.py`, `tsne.py`, `CD.py`, `yuzhifenxi.py`, etc.: QC and analysis; paths likewise resolve relative to `config`.

---

## 10. Reproducibility notes

1. **Structure generation is scripted (FoldX)**  
   - Repair: `python foldxprocess.py --dataset <NAME>`  
   - Mutants: `python foldxpdb.py --dataset <NAME>`
2. FoldX / ESM are third-party assets; obtain them yourself. Defaults are `./bin` and `./weights`, or override via environment variables.
3. `ecod_dict.ecod` comes from [IGMI](https://github.com/ShiweiWu-545/IGMI). This repo has no redistribution rights and **does not include it**; it is needed only when rebuilding `Struc` splits.
4. Absolute machine paths have been removed; everything resolves via `CODE_ROOT` / `MUT_DATA_ROOT`.
5. Random seeds: `SEED`, `SPLIT_SEED` (see `config.py`).
6. Local graph cropping: `EFFECTIVE_RADIUS=15`, `EFFECTIVE_EDGE_CUTOFF=10` (Å).

### Minimal command list

```bash
pip install -r requirements.txt
export DATASET=S4169
export FOLDX_EXE=./bin/foldx
export ESM_MODEL_PATH=./weights/esm2_t36_3B_UR50D.pt

python foldxprocess.py --dataset S4169
python foldxpdb.py --dataset S4169
python DSSP.py --pdb_dir data/S4169/pdb/repairpdb --out_dir data/S4169/DSSP/wt
python data_process.py
python main.py
```

---

## Citation

If you use this code, please cite the corresponding paper (to be updated upon publication).

## License

Code in this repository is for academic research. FoldX, Rosetta, ESM, and other third-party tools follow their own licenses. `ecod_dict.ecod` belongs to [IGMI](https://github.com/ShiweiWu-545/IGMI) and must not be redistributed with this repository without authorization.
