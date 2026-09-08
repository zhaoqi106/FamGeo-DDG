# Node features = Base(41) + DSSP(14) + DeltaESM(2560) = 2615
# Paths: CODE_ROOT/data/{DATASET}/; override data root with MUT_DATA_ROOT

import os

CODE_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data root: prefer env MUT_DATA_ROOT, else repo-local ./data
PROJECT_ROOT = os.path.abspath(
    os.environ.get("MUT_DATA_ROOT", os.path.join(CODE_ROOT, "data"))
)

DATASET = os.environ.get("DATASET", "S4169").strip() or "S4169"

# Local subgraph crop radius / edge cutoff (Angstrom)
EFFECTIVE_RADIUS = 15
EFFECTIVE_EDGE_CUTOFF = 10

# full | noDSSP | noESM | base_only | no_local_pool | no_pearson | no_structure_attention
ABLATION_MODE = "full"
if ABLATION_MODE == "full":
    MASK_DSSP = False
    MASK_ESM = False
    MASK_BASE_ONLY = False
    USE_STRUCTURE_ATTENTION = True

elif ABLATION_MODE == "noDSSP":
    MASK_DSSP = True
    MASK_ESM = False
    MASK_BASE_ONLY = False
    USE_STRUCTURE_ATTENTION = True

elif ABLATION_MODE == "noESM":
    MASK_DSSP = False
    MASK_ESM = True
    MASK_BASE_ONLY = False
    USE_STRUCTURE_ATTENTION = True

elif ABLATION_MODE == "base_only":
    MASK_DSSP = True
    MASK_ESM = True
    MASK_BASE_ONLY = True
    USE_STRUCTURE_ATTENTION = True

elif ABLATION_MODE == "no_local_pool":
    MASK_DSSP = False
    MASK_ESM = False
    MASK_BASE_ONLY = False
    USE_STRUCTURE_ATTENTION = True

elif ABLATION_MODE == "no_pearson":
    MASK_DSSP = False
    MASK_ESM = False
    MASK_BASE_ONLY = False
    USE_STRUCTURE_ATTENTION = True

elif ABLATION_MODE == "no_structure_attention":
    MASK_DSSP = False
    MASK_ESM = False
    MASK_BASE_ONLY = False
    USE_STRUCTURE_ATTENTION = False

else:
    raise ValueError(f"Unknown ABLATION_MODE: {ABLATION_MODE}")

USE_DSSP = not MASK_DSSP
USE_ESM = not MASK_ESM
USE_ESM_IN_MODEL = not MASK_ESM
USE_MUTATION_LOCAL_POOLING = ABLATION_MODE != "no_local_pool"

ENABLE_MULTITASK = False
LOSS_ALPHA_MAX = 0.0 if ABLATION_MODE == "no_pearson" else 0.1

# Input dim is fixed to the full feature block; ablations zero blocks at train time for fair comparison
NODE_IN_DIM = 41 + 14 + 2560  # Base + DSSP + DeltaESM = 2615

# Split dirs: DATA_DIR/{Mpb,Fam,Struc}/; override with SPLIT_TYPES=Struc etc.
SPLIT_TYPES = ["Mpb", "Fam", "Struc"]
_split_env = os.environ.get("SPLIT_TYPES", "").strip()
if _split_env:
    SPLIT_TYPES = [x.strip() for x in _split_env.split(",") if x.strip()]

TYPE = "Mpb"
DATA_DIR = os.path.join(PROJECT_ROOT, DATASET)
CSV_PATH = os.path.join(DATA_DIR, f"{DATASET}.csv")
FUSION_GRAPH_DIR = os.path.join(DATA_DIR, "graph", "fusion")

N_FOLDS = 5
TRAIN_FOLD_CSV = os.path.join(DATA_DIR, "{split_type}", "train_fold_{fold}.csv")
VAL_FOLD_CSV = os.path.join(DATA_DIR, "{split_type}", "val_fold_{fold}.csv")

WT_PDB_DIR = os.path.join(DATA_DIR, "pdb", "repairpdb")
MUT_PDB_DIR = os.path.join(DATA_DIR, "pdb", "mut")
RAW_WT_PDB_DIR = os.path.join(DATA_DIR, "pdb", "wt")
DSSP_WT_DIR = os.path.join(DATA_DIR, "DSSP", "wt")
DSSP_MUT_DIR = os.path.join(DATA_DIR, "DSSP", "mut")
FOLDX_TXT_DIR = os.path.join(DATA_DIR, "foldx_txt")
PSSM_DIR = os.path.join(DATA_DIR, "PSSM", "mut")

RUN_TAG = os.environ.get("MUT_RUN_TAG", "runs")
RUN_OUTPUT_ROOT = os.path.abspath(
    os.environ.get(
        "MUT_RUN_ROOT",
        os.path.join(PROJECT_ROOT, f"{DATASET}_{RUN_TAG}"),
    )
)

ESM_LOCAL_MODEL_PATH = os.environ.get(
    "ESM_MODEL_PATH",
    os.path.join(CODE_ROOT, "weights", "esm2_t36_3B_UR50D.pt"),
)
ESM_DIM = 2560
ESM_LAYERS = [12, 24, 36]
ESM_FUSION_MODE = "mean"
ESM_REDUCED_DIM = 256
# Default cpu to avoid multi-process graph build GPU OOM; for single-process set ESM_DEVICE=cuda
ESM_DEVICE = os.environ.get("ESM_DEVICE", "cpu").strip() or "cpu"

FOLDX_EXE = os.environ.get(
    "FOLDX_EXE",
    os.path.join(CODE_ROOT, "bin", "foldx"),
)
MKDSSP_EXE = os.environ.get("MKDSSP_EXE", "mkdssp")
ROSETTA_EXE = os.environ.get(
    "ROSETTA_EXE",
    os.path.join(CODE_ROOT, "bin", "pmut_scan_parallel.static.linuxgccrelease"),
)
ROSETTA_DB = os.environ.get(
    "ROSETTA_DB",
    os.path.join(CODE_ROOT, "bin", "rosetta_database"),
)
ROSETTA_TXT_DIR = os.path.join(DATA_DIR, "rosettatxt")
PSIBLAST_EXE = os.environ.get("PSIBLAST_EXE", "psiblast")
NR_DB_PATH = os.environ.get(
    "NR_DB_PATH",
    os.path.join(CODE_ROOT, "weights", "swissprot"),
)
# Used by Struc split; file from IGMI, not checked into the repo
ECOD_DICT_PATH = os.environ.get(
    "ECOD_DICT",
    os.path.join(PROJECT_ROOT, "ecod_dict.ecod"),
)

HIDDEN_DIM = 64
N_LAYERS = 3
DROPOUT = 0.3
LOCAL_POOL_FALLBACK = "global"

BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 4
LR = 1e-4
EPOCHS = 100
PATIENCE = 20
WEIGHT_DECAY = 1e-4

ALPHA_WARMUP_EPOCHS = 10
SCORE_LAMBDA = 0.2

# BF16 forward needs Ampere+; fall back to FP32 if unsupported (weights stay FP32)
USE_BF16 = True

SEED = 42
DATALOADER_WORKERS = 4
PIN_MEMORY = True
DEVICE = "cuda"
GPU_ID = int(os.environ.get("GPU_ID", "0"))

ENABLE_EARLY_STOP = True
LOG_DIR = os.path.join(DATA_DIR, "log")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")
SAVE_MODEL_WEIGHTS = True
TYPETIMES = 1  # How many full 5-Fold repeats per split type
SPLIT_SEED = 30
