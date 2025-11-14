import os
import math
import random
import argparse
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, matthews_corrcoef, roc_auc_score

from dataset import build_dataset_with_esm, collate_with_graph
from model import FusionPepNetDual, pair_hsic_loss



def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def safe_div(a, b):
    return a / b if b != 0 else 0.0



def train_one_epoch(model, train_loader, device, optimizer, criterion, hsic_lambda: float):
    model.train()
    loss_sum = 0.0; total = 0
    for feat, g, labels in train_loader:
        feat, labels = feat.to(device), labels.to(device)
        if hasattr(g, "to"):
            g = g.to(device)
        else:
            raise RuntimeError("Expected a PyG Batch. Ensure build_dataset_with_esm(as_pyg=True) and torch_geometric are available.")

        optimizer.zero_grad()
        logits, f_mlp, f_gnn, fused = model(feat, g)
        loss = criterion(logits, labels) + pair_hsic_loss(f_mlp, f_gnn, lambda_hsic=hsic_lambda)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        loss_sum += loss.item() * labels.size(0)
        total    += labels.size(0)
    return loss_sum / max(total, 1)


@torch.no_grad()
def evaluate(model, test_loader, device, criterion):
    model.eval()
    loss_sum = 0.0; total = 0

    probs_pos = []
    y_true_all, y_pred_all = [], []

    for feat, g, labels in test_loader:
        feat, labels = feat.to(device), labels.to(device)
        if hasattr(g, "to"):
            g = g.to(device)
        else:
            raise RuntimeError("Expected a PyG Batch. Ensure build_dataset_with_esm(as_pyg=True) and torch_geometric are available.")

        logits, *_ = model(feat, g)
        loss = criterion(logits, labels)
        loss_sum += loss.item() * labels.size(0); total += labels.size(0)

        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)

        probs_pos.append(prob.detach().cpu().numpy())
        y_true_all.append(labels.detach().cpu().numpy())
        y_pred_all.append(pred.detach().cpu().numpy())

    test_loss = loss_sum / max(total, 1)
    y_true  = np.concatenate(y_true_all).astype(np.int64)
    y_pred  = np.concatenate(y_pred_all).astype(np.int64)
    y_score = np.concatenate(probs_pos).astype(np.float64)

    acc = accuracy_score(y_true, y_pred)
    sen = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spe = safe_div(tn, tn + fp)
    mcc = matthews_corrcoef(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")

    metrics = dict(acc=acc, sen=sen, spe=spe, mcc=mcc, auc=auc,
                   tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))
    return test_loss, metrics



def main():
    parser = argparse.ArgumentParser(description="Train FusionPepNetDual with on-the-fly ESM features (CLI).")

    # Data
    parser.add_argument("--train_fasta", type=str,
                        default="Data/train.txt")
    parser.add_argument("--test_fasta", type=str,
                        default="Data/test.txt")

    # Graph construction
    parser.add_argument("--pH", type=float, default=7.4)
    parser.add_argument("--edge_mode", type=str, default="hybrid", choices=["window", "knn", "hybrid"])
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--knn_k", type=int, default=8)
    parser.add_argument("--no_self_loops", action="store_true", help="Disable self-loops (enabled by default).")

    # ESM feature extraction
    parser.add_argument("--esm_dir", type=str, default=None, help="Override ESM_DIR from esm_feature_mean_max.py.")
    parser.add_argument("--esm_batch_size", type=int, default=16)
    parser.add_argument("--esm_use_fp16", action="store_true")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--hsic_lambda", type=float, default=0.5)

    # Loss and class weights
    parser.add_argument("--use_weighted_ce", action="store_true", help="Use weighted cross-entropy.")
    parser.add_argument("--pos_alpha", type=float, default=1.5, help="Positive-class weight (only if --use_weighted_ce).")

    # Model & saving
    parser.add_argument("--gnn_node_in", type=int, default=48)
    parser.add_argument("--gnn_edge_in", type=int, default=3)
    parser.add_argument("--use_kan", action="store_true", help="Enable KAN branch if used during training.")
    parser.add_argument("--save_path", type=str, default="best_model_mcc_noadj2.pth")

    # Misc
    parser.add_argument("--seed", type=int, default=2024)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    print(f"[INFO] cwd    = {os.getcwd()}")
    print(f"[INFO] config = {vars(args)}")

    # Build datasets (ESM features on the fly)
    train_ds = build_dataset_with_esm(
        fasta_path=args.train_fasta,
        pH=args.pH, edge_mode=args.edge_mode, window=args.window, knn_k=args.knn_k,
        as_pyg=True, self_loops=(not args.no_self_loops),
        esm_dir=args.esm_dir, esm_batch_size=args.esm_batch_size, esm_use_fp16=args.esm_use_fp16
    )
    test_ds = build_dataset_with_esm(
        fasta_path=args.test_fasta,
        pH=args.pH, edge_mode=args.edge_mode, window=args.window, knn_k=args.knn_k,
        as_pyg=True, self_loops=(not args.no_self_loops),
        esm_dir=args.esm_dir, esm_batch_size=args.esm_batch_size, esm_use_fp16=args.esm_use_fp16
    )

    y_train = np.array(train_ds.labels, dtype=np.int64)
    y_test  = np.array(test_ds.labels,  dtype=np.int64)
    print(f"[INFO] Train N={len(train_ds)}, Test N={len(test_ds)}")
    print(f"[INFO] Train label counts: pos={(y_train==1).sum()} / neg={(y_train==0).sum()}")
    print(f"[INFO] Test  label counts: pos={(y_test==1).sum()} / neg={(y_test==0).sum()}")

    train_loader = Data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_with_graph)
    test_loader  = Data.DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=collate_with_graph)

    # Model
    mlp_in_dim = train_ds.features.shape[1]  # ESM mean+max -> 2H
    model = FusionPepNetDual(
        mlp_in_dim=mlp_in_dim,
        gnn_node_in=args.gnn_node_in,
        gnn_edge_in=args.gnn_edge_in,
        use_kan=args.use_kan
    ).to(device)

    # Loss
    if args.use_weighted_ce:
        class_weights = torch.tensor([1.0, args.pos_alpha], dtype=torch.float32, device=device)
        print(f"[INFO] Using weighted CE: neg=1.0, pos={args.pos_alpha:.2f}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        print(f"[INFO] Using standard CE (no class weights)")
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_mcc   = -1.0
    best_epoch = -1

    # Training loop (save by best MCC)
    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, device, optimizer, criterion, args.hsic_lambda)
        test_loss, m = evaluate(model, test_loader, device, criterion)

        print(f"Epoch {epoch:03d} | "
              f"Train Loss: {train_loss:.4f} || "
              f"Test Loss: {test_loss:.4f} | "
              f"ACC={m['acc']:.4f} SEN={m['sen']:.4f} SPE={m['spe']:.4f} "
              f"MCC={m['mcc']:.4f} AUC={m['auc']:.4f} | "
              f"TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")

        # Save by MCC; break ties by ACC>0 (kept from your logic)
        save_flag = (m["mcc"] > best_mcc) or (abs(m["mcc"] - best_mcc) < 1e-9 and m["acc"] > 0)
        if save_flag:
            best_mcc = m["mcc"]; best_epoch = epoch
            torch.save(model.state_dict(), args.save_path)
            print(f"New best model saved (epoch={best_epoch}, MCC={best_mcc:.4f}) -> {os.path.abspath(args.save_path)}")

    print(f"[DONE] Best epoch={best_epoch}, MCC={best_mcc:.4f} | model -> {os.path.abspath(args.save_path)}")


if __name__ == "__main__":
    main()




