import os
import argparse
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data

from dataset import build_dataset_with_esm, collate_with_graph
from ESM_Feature import infer_esm_feature_dim
from model import FusionPepNetDual



BATCH_SIZE      = 64
GNN_NODE_IN     = 48
GNN_EDGE_IN     = 3
USE_KAN         = True


PH              = 7.4
EDGE_MODE       = "hybrid"
WINDOW          = 3
KNN_K           = 8
SELF_LOOPS      = True


ESM_DIR_OVERRIDE = None
ESM_BATCH_SIZE   = 16
ESM_USE_FP16     = False


THRESHOLD        = 0.5


def load_state_dict_file(model_path: str, device: torch.device):
    try:
        return torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location=device)


def read_fasta_headers(fasta_path: str) -> List[str]:

    headers: List[str] = []
    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                headers.append(line.strip()[1:])
    return headers


@torch.no_grad()
def infer(model, loader, device, threshold: float = 0.5):

    model.eval()
    probs_all = []
    pred_lbls = []

    for feat, g, _ in loader:
        feat = feat.to(device)
        if hasattr(g, "to"):
            g = g.to(device)
        else:
            raise RuntimeError("Expected a PyG Batch; ensure as_pyg=True and torch_geometric is installed.")

        logits, _, _, _ = model(feat, g)
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(CPP)
        preds = (probs >= threshold).long()         # 1=CPP, 0=Non-CPP

        probs_all.append(probs.detach().cpu().numpy())
        pred_lbls.append(preds.detach().cpu().numpy())

    probs_all = np.concatenate(probs_all).astype(np.float64)
    pred_lbls = np.concatenate(pred_lbls).astype(np.int64)
    str_labels = np.where(pred_lbls == 1, "CPP", "Non-CPP")
    return probs_all, str_labels


def main():
    parser = argparse.ArgumentParser(description="Per-sample CPP prediction (CSV output).")
    parser.add_argument(
        "--test_fasta",
        type=str,
        default="Data/test.txt",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="best_model.pth",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="predictions.csv",
    )
    parser.add_argument("--esm_dir", type=str, default=ESM_DIR_OVERRIDE)
    parser.add_argument("--esm_batch_size", type=int, default=ESM_BATCH_SIZE)
    parser.add_argument("--esm_use_fp16", action="store_true", default=ESM_USE_FP16)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--disable_kan", action="store_true", help="Use a linear classifier instead of KAN.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    print(f"[INFO] cwd    = {os.getcwd()}")
    print(f"[INFO] config = {vars(args)}")
    print(f"[INFO] expected ESM feature dim = {infer_esm_feature_dim(args.esm_dir)}")

    # Dataset (ESM features computed on the fly)
    test_ds = build_dataset_with_esm(
        fasta_path=args.test_fasta,
        pH=PH, edge_mode=EDGE_MODE, window=WINDOW, knn_k=KNN_K,
        as_pyg=True, self_loops=SELF_LOOPS,
        esm_dir=args.esm_dir, esm_batch_size=args.esm_batch_size, esm_use_fp16=args.esm_use_fp16
    )
    print(f"[INFO] Test N={len(test_ds)}")
    print(f"[INFO] inferred input dim from dataset = {test_ds.feature_dim}")

    test_loader = Data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_with_graph
    )

    # Model
    mlp_in_dim = test_ds.feature_dim
    model = FusionPepNetDual(
        mlp_in_dim=mlp_in_dim,
        gnn_node_in=GNN_NODE_IN,
        gnn_edge_in=GNN_EDGE_IN,
        use_kan=(not args.disable_kan)
    ).to(device)

    # Load weights
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    state = load_state_dict_file(args.model_path, device)
    model.load_state_dict(state, strict=True)
    print(f"[INFO] Loaded model weights: {args.model_path}")

    # Inference: get P(CPP) and labels, then convert to predicted-class probability
    probs_cpp, labels = infer(model, test_loader, device, threshold=args.threshold)
    prob_pred = np.where(labels == "CPP", probs_cpp, 1.0 - probs_cpp)

    # IDs from FASTA headers (same order as dataset)
    ids = read_fasta_headers(args.test_fasta)
    if len(ids) != len(test_ds):
        print(f"[WARN] Header count ({len(ids)}) != dataset size ({len(test_ds)}). Falling back to index-based IDs.")
        ids = [f"sample_{i}" for i in range(len(test_ds))]

    # DataFrame with predicted-class probability
    df = pd.DataFrame({
        "id": ids,
        "prob": prob_pred,       # probability of the predicted class
        "pred_label": labels     # "CPP" or "Non-CPP"
    })

    # Print each row
    for i, row in df.iterrows():
        print(f"{row['id']}\tprob={row['prob']:.6f}\t{row['pred_label']}")

    # Save CSV
    out_dir = os.path.dirname(os.path.abspath(args.out_csv))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[INFO] Saved predictions to: {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
