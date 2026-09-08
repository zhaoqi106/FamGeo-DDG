# Node features = Base(41) + DSSP(14) + DeltaESM(2560) = 2615

import logging
import os
import re
from typing import Dict, List, Optional, Tuple, Union

import esm
import numpy as np
import pandas as pd
import torch
from Bio.PDB import NeighborSearch, PDBParser, is_aa
from torch_geometric.data import Data

import config as cfg

USE_DSSP = getattr(cfg, "USE_DSSP", True)
USE_ESM = getattr(cfg, "USE_ESM", True)

WT_PDB_DIR = getattr(cfg, "WT_PDB_DIR", "")
MUT_PDB_DIR = getattr(cfg, "MUT_PDB_DIR", "")
DSSP_WT_DIR = getattr(cfg, "DSSP_WT_DIR", "")

ESM_LOCAL_MODEL_PATH = getattr(cfg, "ESM_LOCAL_MODEL_PATH", "")
ESM_DIM = getattr(cfg, "ESM_DIM", 2560)
ESM_LAYERS = getattr(cfg, "ESM_LAYERS", [12, 24, 36])
ESM_FUSION_MODE = getattr(cfg, "ESM_FUSION_MODE", "mean")

logging.basicConfig(level=logging.WARNING, format="[graph_utils] %(message)s")

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
DSSP_FEATURE_DIM = 14
BASE_FEATURE_DIM = 41


def normalize_icode(icode) -> str:
    if icode is None:
        return " "
    s = str(icode)
    if s == "" or s.strip() == "":
        return " "
    return s.strip()


def get_one_hot(residue_symbol: str) -> np.ndarray:
    vec = np.zeros(20, dtype=np.float32)
    if len(residue_symbol) == 3:
        residue_symbol = THREE_TO_ONE.get(residue_symbol, "X")
    if residue_symbol in AA_TO_IDX:
        vec[AA_TO_IDX[residue_symbol]] = 1.0
    return vec


def parse_mutation_str(mut_str: str) -> List[Dict[str, Union[str, int]]]:
    """
    Parse strings like AI23V / AI23AV; multi-site mutations comma-separated.
    Regex: WT_AA + CHAIN + RESSEQ + optional ICODE + MUT_AA
    """
    if mut_str is None:
        return []

    mut_list = []
    mut_str = str(mut_str).replace(" ", "")
    sub_muts = mut_str.split(",")
    pattern = re.compile(r"([A-Z])([A-Z])(\d+)([A-Z]?)([A-Z])")

    for sub in sub_muts:
        match = pattern.fullmatch(sub)
        if match:
            wt_aa, chain_code, resseq, icode, mut_aa = match.groups()
            mut_list.append({
                "chain": chain_code,
                "wt": wt_aa,
                "resseq": int(resseq),
                "icode": normalize_icode(icode),
                "mut": mut_aa,
            })
    return mut_list


_ESM_MODEL = None
_ESM_ALPHABET = None
_ESM_BATCH_CONVERTER = None

# Graph build defaults to CPU to avoid each worker loading ESM-3B on GPU and OOMing.
# For single-process with free GPU: ESM_DEVICE=cuda DATASET=ATLAS python data_process.py --workers 0
ESM_DEVICE = str(getattr(cfg, "ESM_DEVICE", os.environ.get("ESM_DEVICE", "cpu"))).strip().lower() or "cpu"


def _reset_esm_globals():
    global _ESM_MODEL, _ESM_ALPHABET, _ESM_BATCH_CONVERTER
    _ESM_MODEL = None
    _ESM_ALPHABET = None
    _ESM_BATCH_CONVERTER = None


def get_esm_model():
    global _ESM_MODEL, _ESM_ALPHABET, _ESM_BATCH_CONVERTER

    if not USE_ESM:
        return None, None

    # Both model and converter must be ready; else leftover half-init after a prior .cuda() OOM
    if _ESM_MODEL is not None and _ESM_BATCH_CONVERTER is not None:
        return _ESM_MODEL, _ESM_BATCH_CONVERTER

    _reset_esm_globals()

    if not os.path.exists(ESM_LOCAL_MODEL_PATH):
        raise FileNotFoundError(f"ESM model not found: {ESM_LOCAL_MODEL_PATH}")

    # PyTorch>=2.6 defaults weights_only=True; local ESM weights contain argparse.Namespace, so disable
    _orig_torch_load = torch.load

    def _torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)

    torch.load = _torch_load_compat
    try:
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(
            ESM_LOCAL_MODEL_PATH
        )
        model.eval()

        want_cuda = ESM_DEVICE.startswith("cuda") and torch.cuda.is_available()
        if want_cuda:
            try:
                model = model.cuda()
            except torch.cuda.OutOfMemoryError:
                logging.warning(
                    "Failed to move ESM to CUDA (OOM); falling back to CPU. Tip: "
                    "set ESM_DEVICE=cpu or rerun with --workers 0/1."
                )
                torch.cuda.empty_cache()
                model = model.cpu()
        else:
            model = model.cpu()

        converter = alphabet.get_batch_converter()
        _ESM_MODEL = model
        _ESM_ALPHABET = alphabet
        _ESM_BATCH_CONVERTER = converter
        logging.warning(
            f"ESM loaded on {next(model.parameters()).device} "
            f"(ESM_DEVICE={ESM_DEVICE})"
        )
    except Exception:
        _reset_esm_globals()
        raise
    finally:
        torch.load = _orig_torch_load

    return _ESM_MODEL, _ESM_BATCH_CONVERTER


def get_chain_sequence_and_mapping(chain) -> Tuple[str, List[Tuple[int, str]]]:
    seq_chars = []
    mapping = []
    for res in chain:
        if is_aa(res):
            res_name = res.get_resname()
            aa_char = THREE_TO_ONE.get(res_name, "X")
            seq_chars.append(aa_char)
            res_id = res.get_id()
            mapping.append((int(res_id[1]), normalize_icode(res_id[2])))
    return "".join(seq_chars), mapping


def load_esm_data(
    input_data: Union[dict, object],
    mapping_dict: Optional[Dict[str, Dict[Tuple[int, str], int]]] = None,
) -> Dict[str, Dict[Tuple[int, str], torch.Tensor]]:
    """
    Return {chain_id: {(resseq, icode): embedding}}.
    Input may be {chain_id: sequence} or a Bio.PDB.Structure.
    """
    if not USE_ESM:
        return {}

    model_esm, batch_converter = get_esm_model()
    if model_esm is None or batch_converter is None:
        raise RuntimeError(
            "ESM model not loaded correctly (model/batch_converter is None). "
            "Check ESM_LOCAL_MODEL_PATH, or set ESM_DEVICE=cpu and retry."
        )

    device = next(model_esm.parameters()).device
    esm_features: Dict[str, Dict[Tuple[int, str], torch.Tensor]] = {}

    layers_to_extract = ESM_LAYERS if isinstance(ESM_LAYERS, list) and len(ESM_LAYERS) > 0 else [36]
    fusion_mode = ESM_FUSION_MODE.lower() if hasattr(ESM_FUSION_MODE, "lower") else "mean"

    if isinstance(input_data, dict):
        batch = [(cid, seq) for cid, seq in input_data.items() if len(seq) > 0]
        if not batch:
            return {}

        _, _, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model_esm(batch_tokens, repr_layers=layers_to_extract, return_contacts=False)

        for batch_idx, (cid, seq) in enumerate(batch):
            token_reps = [results["representations"][layer][batch_idx, 1: len(seq) + 1] for layer in layers_to_extract]
            if fusion_mode == "mean":
                emb = torch.mean(torch.stack(token_reps), dim=0)
            elif fusion_mode == "concat":
                emb = torch.cat(token_reps, dim=-1)
            else:
                emb = token_reps[-1]

            if mapping_dict and cid in mapping_dict:
                seq_to_res = mapping_dict[cid]
                chain_feats = {}
                for (resseq, icode), seq_idx in seq_to_res.items():
                    if seq_idx < emb.size(0):
                        chain_feats[(int(resseq), normalize_icode(icode))] = emb[seq_idx]
                esm_features[cid] = chain_feats
            else:
                esm_features[cid] = {(j + 1, " "): emb[j] for j in range(len(seq))}

        return esm_features

    batch = []
    chain_mappings = {}
    for model in input_data:
        for chain in model:
            chain_id = chain.get_id()
            seq, mapping = get_chain_sequence_and_mapping(chain)
            if len(seq) > 0:
                batch.append((chain_id, seq))
                chain_mappings[chain_id] = mapping

    if not batch:
        return {}

    _, _, batch_tokens = batch_converter(batch)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model_esm(batch_tokens, repr_layers=layers_to_extract, return_contacts=False)

    for batch_idx, (cid, seq) in enumerate(batch):
        token_reps = [results["representations"][layer][batch_idx, 1: len(seq) + 1] for layer in layers_to_extract]
        if fusion_mode == "mean":
            emb = torch.mean(torch.stack(token_reps), dim=0)
        elif fusion_mode == "concat":
            emb = torch.cat(token_reps, dim=-1)
        else:
            emb = token_reps[-1]

        mapping = chain_mappings[cid]
        esm_features[cid] = {
            (mapping[j][0], normalize_icode(mapping[j][1])): emb[j]
            for j in range(len(seq))
        }

    return esm_features


def load_dssp_data(
    pdb_id: str,
    dssp_dir: str = DSSP_WT_DIR,
    default_value: float = 0.0,
) -> Dict[str, Dict[Tuple[int, str], np.ndarray]]:
    """
    Read WT DSSP csv -> {chain: {(resseq, icode): 14-dim}}.
    Order: [ss_H, ss_E, ss_C, RASA, phi, psi,
           NH_O_1/O_NH_1/NH_O_2/O_NH_2 energy,
           ASA_complex, ASA_chain, Delta_ASA, ASA_ratio]
    """
    if not USE_DSSP:
        return {}

    dssp_path = os.path.join(dssp_dir, f"{pdb_id}.csv")
    if not os.path.exists(dssp_path):
        logging.warning(f"Missing DSSP file: {dssp_path}")
        return {}

    required_cols = [
        "chain_id", "resseq", "icode", "ss_H", "ss_E", "ss_C", "RASA",
        "phi", "psi", "NH_O_1_energy", "O_NH_1_energy",
        "NH_O_2_energy", "O_NH_2_energy", "ASA_complex",
        "ASA_chain", "Delta_ASA", "ASA_ratio",
    ]

    try:
        df = pd.read_csv(dssp_path)
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logging.warning(f"Missing columns in {dssp_path}: {missing_cols}")
            return {}

        dssp_dict: Dict[str, Dict[Tuple[int, str], np.ndarray]] = {}
        for idx, row in df.iterrows():
            try:
                chain_id = row["chain_id"]
                resseq = int(row["resseq"])
                icode = normalize_icode(row.get("icode", " "))

                def safe_float(val, default=default_value):
                    if pd.isna(val) or str(val).strip() in ["", "NA", "N/A"]:
                        return default
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return default

                feats = np.array([
                    safe_float(row.get("ss_H", default_value)),
                    safe_float(row.get("ss_E", default_value)),
                    safe_float(row.get("ss_C", default_value)),
                    safe_float(row.get("RASA", default_value)),
                    safe_float(row.get("phi", default_value)),
                    safe_float(row.get("psi", default_value)),
                    safe_float(row.get("NH_O_1_energy", default_value)),
                    safe_float(row.get("O_NH_1_energy", default_value)),
                    safe_float(row.get("NH_O_2_energy", default_value)),
                    safe_float(row.get("O_NH_2_energy", default_value)),
                    safe_float(row.get("ASA_complex", default_value)),
                    safe_float(row.get("ASA_chain", default_value)),
                    safe_float(row.get("Delta_ASA", default_value)),
                    safe_float(row.get("ASA_ratio", default_value)),
                ], dtype=np.float32)

                if np.isnan(feats).all():
                    continue

                if chain_id not in dssp_dict:
                    dssp_dict[chain_id] = {}
                dssp_dict[chain_id][(resseq, icode)] = feats

            except Exception as e:
                logging.warning(f"Row {idx} parse failed in {dssp_path}: {e}")
                continue

        return dssp_dict

    except Exception as e:
        logging.error(f"Failed to load DSSP for {pdb_id}: {e}")
        return {}


def load_structure(pdb_path: str):
    if not os.path.exists(pdb_path):
        return None
    parser = PDBParser(QUIET=True)
    try:
        return parser.get_structure(os.path.basename(pdb_path), pdb_path)
    except Exception:
        return None


def find_residue_by_resseq_icode(chain_obj, resseq: int, icode: str):
    icode = normalize_icode(icode)
    for res in chain_obj:
        if not is_aa(res):
            continue
        if int(res.id[1]) == int(resseq) and normalize_icode(res.id[2]) == icode:
            return res
    return None


def get_residue_coord(res) -> np.ndarray:
    if "CA" in res:
        return res["CA"].get_coord()
    return np.mean([atom.get_coord() for atom in res.get_atoms()], axis=0)


def build_local_fusion_graph(
    pdb_id: str,
    sample_id: str,
    mutation_str: str,
    center_dist: float = 25.0,
    edge_cutoff: float = 25.0,
):
    """
    Build a large graph for training-time dynamic cropping.
    Extra fields: node_dist_to_mut, edge_dist.
    """
    try:
        muts = parse_mutation_str(mutation_str)
        if not muts:
            return None, "Invalid Mutation"

        wt_pdb_path = os.path.join(WT_PDB_DIR, f"{pdb_id}.pdb")
        mut_pdb_path = os.path.join(MUT_PDB_DIR, f"{sample_id}.pdb")

        wt_structure = load_structure(wt_pdb_path)
        if wt_structure is None:
            return None, "WT PDB Load Failed"

        mut_structure = load_structure(mut_pdb_path)
        wt_model = wt_structure[0]

        dssp_wt = load_dssp_data(pdb_id, DSSP_WT_DIR) if USE_DSSP else {}
        esm_wt = load_esm_data(wt_structure) if USE_ESM else {}

        # If mutant PDB is missing, apply mutations on WT sequence and recompute ESM
        if USE_ESM:
            if mut_structure is None:
                chain_seqs = {}
                chain_seq_idx_map = {}
                for chain in wt_model:
                    cid = chain.id
                    seq, mapping = get_chain_sequence_and_mapping(chain)
                    if seq:
                        chain_seqs[cid] = seq
                        chain_seq_idx_map[cid] = {
                            (resseq, normalize_icode(icode)): idx
                            for idx, (resseq, icode) in enumerate(mapping)
                        }

                mut_chain_seqs = {}
                for cid, seq in chain_seqs.items():
                    mut_seq = list(seq)
                    for m in muts:
                        if m["chain"] == cid:
                            key = (m["resseq"], normalize_icode(m["icode"]))
                            seq_idx = chain_seq_idx_map[cid].get(key)
                            if seq_idx is not None:
                                mut_seq[seq_idx] = m["mut"]
                    mut_chain_seqs[cid] = "".join(mut_seq)

                esm_mut = load_esm_data(mut_chain_seqs, mapping_dict=chain_seq_idx_map)
            else:
                esm_mut = load_esm_data(mut_structure)
        else:
            esm_mut = {}

        target_atoms = []
        for m in muts:
            cid = m["chain"]
            resseq = m["resseq"]
            icode = normalize_icode(m["icode"])

            chain_obj = None
            for chain in wt_model:
                if chain.id == cid:
                    chain_obj = chain
                    break
            if chain_obj is None:
                continue

            res = find_residue_by_resseq_icode(chain_obj, resseq, icode)
            if res is None:
                continue

            if "CA" in res:
                target_atoms.append(res["CA"])
            else:
                target_atoms.extend(res.get_atoms())

        if not target_atoms:
            return None, "Mutation Site Not Found"

        atom_list = list(wt_model.get_atoms())
        ns = NeighborSearch(atom_list)
        nearby_residues = set()
        for atom in target_atoms:
            nearby_residues.update(ns.search(atom.get_coord(), center_dist, level="R"))

        nearby_residues = [res for res in nearby_residues if is_aa(res)]
        nearby_residues = sorted(
            nearby_residues,
            key=lambda r: (r.get_parent().id, int(r.id[1]), normalize_icode(r.id[2])),
        )

        if len(nearby_residues) == 0:
            return None, "No Nearby Residues"

        # Multi-site mutation: use geometric center of all center atoms
        mutation_center = np.mean([atom.get_coord() for atom in target_atoms], axis=0)

        node_feats = []
        coords = []
        node_dist_to_mut = []

        for res in nearby_residues:
            cid = res.get_parent().id
            resseq = int(res.id[1])
            icode = normalize_icode(res.id[2])

            wt_char = THREE_TO_ONE.get(res.get_resname(), "X")
            feat_wt = get_one_hot(wt_char)

            is_mut_site = False
            mut_char = wt_char
            for m in muts:
                if (
                    m["chain"] == cid
                    and m["resseq"] == resseq
                    and normalize_icode(m["icode"]) == icode
                ):
                    is_mut_site = True
                    mut_char = m["mut"]
                    break

            feat_mut = get_one_hot(mut_char)
            feat_flag = np.array([1.0 if is_mut_site else 0.0], dtype=np.float32)

            feats = [feat_wt, feat_mut, feat_flag]

            if USE_DSSP:
                dssp_vec = np.zeros(DSSP_FEATURE_DIM, dtype=np.float32)
                if cid in dssp_wt and (resseq, icode) in dssp_wt[cid]:
                    dssp_vec = dssp_wt[cid][(resseq, icode)]
                feats.append(dssp_vec)

            if USE_ESM:
                key = (resseq, icode)
                wt_vec = esm_wt.get(cid, {}).get(key)
                mut_vec = esm_mut.get(cid, {}).get(key)

                if wt_vec is None and mut_vec is None:
                    delta_vec = np.zeros(ESM_DIM, dtype=np.float32)
                else:
                    if wt_vec is None:
                        wt_vec = torch.zeros_like(mut_vec)
                    if mut_vec is None:
                        mut_vec = wt_vec.clone()
                    delta_vec = (mut_vec - wt_vec).detach().cpu().numpy().astype(np.float32)
                feats.append(delta_vec)

            full_vec = np.concatenate(feats, axis=0).astype(np.float32)
            node_feats.append(torch.from_numpy(full_vec))

            pos = get_residue_coord(res).astype(np.float32)
            coords.append(torch.from_numpy(pos))
            node_dist_to_mut.append(float(np.linalg.norm(pos - mutation_center)))

        x = torch.stack(node_feats).float()
        pos = torch.stack(coords).float()

        # Keep edge distances for training-time cropping by EFFECTIVE_EDGE_CUTOFF
        dist_mat = (pos.unsqueeze(0) - pos.unsqueeze(1)).norm(dim=-1)
        src_list, dst_list, edge_dists = [], [], []
        num_nodes = x.shape[0]

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and float(dist_mat[i, j]) < float(edge_cutoff):
                    src_list.append(i)
                    dst_list.append(j)
                    edge_dists.append(float(dist_mat[i, j]))

        if len(src_list) > 0:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_dist = torch.tensor(edge_dists, dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_dist = torch.zeros((0,), dtype=torch.float32)

        data = Data(x=x, pos=pos, edge_index=edge_index)
        data.edge_dist = edge_dist
        data.node_dist_to_mut = torch.tensor(node_dist_to_mut, dtype=torch.float32)
        data.y = None

        expected_dim = BASE_FEATURE_DIM
        if USE_DSSP:
            expected_dim += DSSP_FEATURE_DIM
        if USE_ESM:
            expected_dim += ESM_DIM
        if data.x.size(1) != expected_dim:
            return None, f"FeatureDimMismatch: got {data.x.size(1)}, expected {expected_dim}"

        return data, "Success"

    except Exception as e:
        print(f"[DEBUG] Error in build_local_fusion_graph: {type(e).__name__} - {str(e)} | Sample: {sample_id}")
        return None, str(e)
