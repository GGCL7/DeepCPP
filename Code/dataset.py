
from typing import List, Tuple, Optional, Any
import numpy as np
import torch
import torch.utils.data as Data
from ESM_Feature import esmfeature
from graph_features import build_peptide_graph

def read_protein_sequences_from_fasta(file_path: str) -> Tuple[List[str], List[int]]:
    sequences: List[str] = []
    labels: List[int] = []
    sequence = ''
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence)
                    sequence = ''
                labels.append(1 if 'pos' in line else 0)
            else:
                sequence += line
        if sequence:
            sequences.append(sequence)
    return sequences, labels


def load_graphs_from_fasta(
    fasta_path: str,
    pH: float = 7.4,
    edge_mode: str = "window",
    window: int = 3,
    knn_k: int = 8,
    as_pyg: Optional[bool] = True,
    self_loops: bool = True
) -> List[Any]:
    sequences, _ = read_protein_sequences_from_fasta(fasta_path)
    graphs = [
        build_peptide_graph(
            seq=s,
            pH=pH,
            edge_mode=edge_mode,
            window=window,
            knn_k=knn_k,
            as_pyg=as_pyg,
            self_loops=self_loops
        )
        for s in sequences
    ]
    return graphs


class MyGraphDataSet(Data.Dataset):
    def __init__(self, features: np.ndarray, graphs: List[Any], labels: List[int]):
        assert len(features) == len(graphs) == len(labels), \
            f"长度不一致：feat={len(features)}, graph={len(graphs)}, label={len(labels)}"
        self.features = features.astype(np.float32, copy=False)
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            self.graphs[idx],
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

def collate_with_graph(batch, pyg_batch: bool = True):
    fs, gs, ys = zip(*batch)
    f = torch.stack(fs, dim=0)                   # [B, D]
    y = torch.stack(ys, dim=0)                   # [B]
    if pyg_batch:
        try:
            from torch_geometric.data import Batch
            if len(gs) > 0 and hasattr(gs[0], "__class__") and gs[0].__class__.__name__ != "dict":
                g = Batch.from_data_list(list(gs))
            else:
                g = list(gs)
        except Exception:
            g = list(gs)
    else:
        g = list(gs)
    return f, g, y

def build_dataset_with_esm(
    fasta_path: str,
    pH: float = 7.4,
    edge_mode: str = "hybrid",
    window: int = 3,
    knn_k: int = 8,
    as_pyg: Optional[bool] = True,
    self_loops: bool = True,
    esm_dir: str | None = None,
    esm_batch_size: int = 16,
    esm_use_fp16: bool = False,
) -> MyGraphDataSet:
    sequences, labels = read_protein_sequences_from_fasta(fasta_path)
    feats_np = esmfeature(
        sequences,
        esm_dir=esm_dir,
        batch_size=esm_batch_size,
        use_fp16=esm_use_fp16,
        device=None,
        return_tensor=False
    )


    graphs = load_graphs_from_fasta(
        fasta_path=fasta_path,
        pH=pH, edge_mode=edge_mode, window=window, knn_k=knn_k,
        as_pyg=as_pyg, self_loops=self_loops
    )

    assert len(feats_np) == len(graphs) == len(labels), \
        f"feats={len(feats_np)}, graphs={len(graphs)}, labels={len(labels)}"

    return MyGraphDataSet(feats_np, graphs, labels)
