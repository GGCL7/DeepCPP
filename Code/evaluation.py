
import os
import numpy as np
import torch
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, matthews_corrcoef, roc_auc_score

from dataset import build_dataset_with_esm, collate_with_graph
from model import FusionPepNetDual  


TEST_FASTA = "Data/test.txt"
MODEL_PATH = "best_model.pth"


BATCH_SIZE   = 64
GNN_NODE_IN  = 48
GNN_EDGE_IN  = 3
USE_KAN      = True


ESM_BATCH_SIZE   = 16
ESM_USE_FP16     = False
ESM_DIR_OVERRIDE = None

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    print(f"[INFO] cwd    = {os.getcwd()}")


    test_ds = build_dataset_with_esm(
        fasta_path=TEST_FASTA,
        pH=7.4, edge_mode="hybrid", window=3, knn_k=8,
        as_pyg=True, self_loops=True,
        esm_dir=ESM_DIR_OVERRIDE, esm_batch_size=ESM_BATCH_SIZE, esm_use_fp16=ESM_USE_FP16
    )
    labels = np.array(test_ds.labels, dtype=np.int64)
    print(f"[INFO] Test N={len(test_ds)} | pos={(labels==1).sum()} neg={(labels==0).sum()}")

    test_loader = Data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_with_graph
    )


    mlp_in_dim = test_ds.features.shape[1]
    model = FusionPepNetDual(
        mlp_in_dim=mlp_in_dim,
        gnn_node_in=GNN_NODE_IN,
        gnn_edge_in=GNN_EDGE_IN,
        use_kan=USE_KAN
    ).to(device)


    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Can't find model：{MODEL_PATH}")
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state, strict=True)
    print(f"[INFO]：{MODEL_PATH}")


    metrics = evaluate(model, test_loader, device)
    print(f"[TEST] ACC={metrics['ACC']:.4f} "
          f"SEN={metrics['SEN']:.4f} SPE={metrics['SPE']:.4f} "
          f"MCC={metrics['MCC']:.4f} AUC={metrics['AUC']:.4f} | "
          f"TP={metrics['TP']} TN={metrics['TN']} FP={metrics['FP']} FN={metrics['FN']}")

if __name__ == "__main__":
    main()
