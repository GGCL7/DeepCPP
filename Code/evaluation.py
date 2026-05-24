
import os
import argparse
import numpy as np
import torch
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, matthews_corrcoef, roc_auc_score

from dataset import build_dataset_with_esm, collate_with_graph
from ESM_Feature import infer_esm_feature_dim
from model import FusionPepNetDual


def load_state_dict_file(model_path: str, device: torch.device):
    try:
        return torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location=device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true_all, y_pred_all, y_score_all = [], [], []

    for feat, g, labels in loader:
        feat, labels = feat.to(device), labels.to(device)
        if hasattr(g, "to"):
            g = g.to(device)
        else:
            raise RuntimeError("Need PyG Batch")

        logits, _, _, _ = model(feat, g)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        y_true_all.append(labels.detach().cpu().numpy())
        y_pred_all.append(preds.detach().cpu().numpy())
        y_score_all.append(probs.detach().cpu().numpy())

    y_true  = np.concatenate(y_true_all).astype(np.int64)
    y_pred  = np.concatenate(y_pred_all).astype(np.int64)
    y_score = np.concatenate(y_score_all).astype(np.float64)

    acc = accuracy_score(y_true, y_pred)
    sen = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")

    return {
        "ACC": acc, "SEN": sen, "SPE": spe, "MCC": mcc, "AUC": auc,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DeepCPP model.")
    parser.add_argument("--test_fasta", type=str, default="Data/test.txt")
    parser.add_argument("--model_path", type=str, default="best_model.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gnn_node_in", type=int, default=48)
    parser.add_argument("--gnn_edge_in", type=int, default=3)
    parser.add_argument("--disable_kan", action="store_true", help="Use a linear classifier instead of KAN.")
    parser.add_argument("--esm_dir", type=str, default=None)
    parser.add_argument("--esm_batch_size", type=int, default=16)
    parser.add_argument("--esm_use_fp16", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    print(f"[INFO] cwd    = {os.getcwd()}")
    print(f"[INFO] expected ESM feature dim = {infer_esm_feature_dim(args.esm_dir)}")


    test_ds = build_dataset_with_esm(
        fasta_path=args.test_fasta,
        pH=7.4, edge_mode="hybrid", window=3, knn_k=8,
        as_pyg=True, self_loops=True,
        esm_dir=args.esm_dir, esm_batch_size=args.esm_batch_size, esm_use_fp16=args.esm_use_fp16
    )
    labels = np.array(test_ds.labels, dtype=np.int64)
    print(f"[INFO] Test N={len(test_ds)} | pos={(labels==1).sum()} neg={(labels==0).sum()}")
    print(f"[INFO] inferred input dim from dataset = {test_ds.feature_dim}")

    test_loader = Data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_with_graph
    )


    mlp_in_dim = test_ds.feature_dim
    model = FusionPepNetDual(
        mlp_in_dim=mlp_in_dim,
        gnn_node_in=args.gnn_node_in,
        gnn_edge_in=args.gnn_edge_in,
        use_kan=(not args.disable_kan)
    ).to(device)


    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Can't find model: {args.model_path}")
    state = load_state_dict_file(args.model_path, device)
    model.load_state_dict(state, strict=True)
    print(f"[INFO] loaded model weights: {args.model_path}")


    metrics = evaluate(model, test_loader, device)
    print(f"[TEST] ACC={metrics['ACC']:.4f} "
          f"SEN={metrics['SEN']:.4f} SPE={metrics['SPE']:.4f} "
          f"MCC={metrics['MCC']:.4f} AUC={metrics['AUC']:.4f} | "
          f"TP={metrics['TP']} TN={metrics['TN']} FP={metrics['FP']} FN={metrics['FN']}")

if __name__ == "__main__":
    main()
