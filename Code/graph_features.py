
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch

try:
    from torch_geometric.data import Data, Batch
    _HAS_PYG = True
except Exception:
    _HAS_PYG = False


KD: Dict[str, float] = {
    'A':  1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C':  2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I':  4.5,
    'L':  3.8, 'K': -3.9, 'M':  1.9, 'F':  2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V':  4.2
}

PKA_SIDECHAIN: Dict[str, float] = {
    'D': 3.90, 'E': 4.07, 'H': 6.04, 'C': 8.37, 'Y': 10.46, 'K': 10.54, 'R': 12.48
}


BLOSUM62_TABLE: Dict[str, List[int]] = {
    'A': [4,  -1, -2, -2, 0,  -1, -1, 0, -2,  -1, -1, -1, -1, -2, -1, 1,  0,  -3, -2, 0],
    'R': [-1, 5,  0,  -2, -3, 1,  0,  -2, 0,  -3, -2, 2,  -1, -3, -2, -1, -1, -3, -2, -3],
    'N': [-2, 0,  6,  1,  -3, 0,  0,  0,  1,  -3, -3, 0,  -2, -3, -2, 1,  0,  -4, -2, -3],
    'D': [-2, -2, 1,  6,  -3, 0,  2,  -1, -1, -3, -4, -1, -3, -3, -1, 0,  -1, -4, -3, -3],
    'C': [0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    'Q': [-1, 1,  0,  0,  -3, 5,  2,  -2, 0,  -3, -2, 1,  0,  -3, -1, 0,  -1, -2, -1, -2],
    'E': [-1, 0,  0,  2,  -4, 2,  5,  -2, 0,  -3, -3, 1,  -2, -3, -1, 0,  -1, -3, -2, -2],
    'G': [0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2, -3, -3, -2, 0,  -2, -2, -3, -3],
    'H': [-2, 0,  1,  -1, -3, 0,  0,  -2, 8,  -3, -3, -1, -2, -1, -2, -1, -2, -2, 2,  -3],
    'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3, 1,  0,  -3, -2, -1, -3, -1, 3],
    'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2, 2,  0,  -3, -2, -1, -2, -1, 1],
    'K': [-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,  -1, -3, -1, 0,  -1, -3, -2, -2],
    'M': [-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1, 5,  0,  -2, -1, -1, -1, -1, 1],
    'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3, 0,  6,  -4, -2, -2, 1,  3,  -1],
    'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7,  -1, -1, -4, -3, -2],
    'S': [1,  -1, 1,  0,  -1, 0,  0,  0,  -1, -2, -2, 0,  -1, -2, -1, 4,  1,  -3, -2, -2],
    'T': [0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1,  5,  -2, -2, 0],
    'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1,  -4, -3, -2, 11, 2,  -3],
    'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2, -1, 3,  -3, -2, -2, 2,  7,  -1],
    'V': [0,  -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2, 1,  -1, -2, -2, 0,  -3, -1, 4],
}

ZSCALE_TABLE: Dict[str, List[float]] = {
    'A': [0.24,  -2.32,  0.60, -0.14,  1.30],
    'C': [0.84,  -1.67,  3.71,  0.18, -2.65],
    'D': [3.98,   0.93,  1.93, -2.46,  0.75],
    'E': [3.11,   0.26, -0.11, -0.34, -0.25],
    'F': [-4.22,  1.94,  1.06,  0.54, -0.62],
    'G': [2.05,  -4.06,  0.36, -0.82, -0.38],
    'H': [2.47,   1.95,  0.26,  3.90,  0.09],
    'I': [-3.89, -1.73, -1.71, -0.84,  0.26],
    'K': [2.29,   0.89, -2.49,  1.49,  0.31],
    'L': [-4.28, -1.30, -1.49, -0.72,  0.84],
    'M': [-2.85, -0.22,  0.47,  1.94, -0.98],
    'N': [3.05,   1.62,  1.04, -1.15,  1.61],
    'P': [-1.66,  0.27,  1.84,  0.70,  2.00],
    'Q': [1.75,   0.50, -1.44, -1.34,  0.66],
    'R': [3.52,   2.50, -3.50,  1.99, -0.17],
    'S': [2.39,  -1.07,  1.15, -1.39,  0.67],
    'T': [0.75,  -2.18, -1.12, -1.46, -0.40],
    'V': [-2.59, -2.64, -1.54, -0.85, -0.02],
    'W': [-4.36,  3.94,  0.59,  3.44, -1.59],
    'Y': [-2.54,  2.44,  0.43,  0.04, -1.47],
}

AA20 = "ARNDCQEGHILKMFPSTWYV"
AA2IDX = {a: i for i, a in enumerate(AA20)}


def _frac_charge_sidechain(aa: str, pH: float = 7.4) -> float:
    if aa not in PKA_SIDECHAIN:
        return 0.0
    pKa = PKA_SIDECHAIN[aa]
    if aa in ('D', 'E', 'C', 'Y'):
        alpha = 1.0 / (1.0 + 10.0**(pKa - pH))
        return -alpha
    else:  # H, K, R
        alpha = 1.0 / (1.0 + 10.0**(pH - pKa))
        return +alpha

def _aa_blosum(aa: str) -> np.ndarray:
    return np.array(BLOSUM62_TABLE.get(aa, [0]*20), dtype=np.float32)

def _aa_zscale(aa: str) -> np.ndarray:
    return np.array(ZSCALE_TABLE.get(aa, [0.0]*5), dtype=np.float32)

def seq_to_node_features(seq: str, pH: float = 7.4) -> np.ndarray:

    seq = seq.strip().upper()
    L = len(seq)

    kd  = np.array([KD.get(a, 0.0) for a in seq], dtype=np.float32)               # [L]
    ch  = np.array([_frac_charge_sidechain(a, pH=pH) for a in seq], dtype=np.float32)
    dkd = np.zeros(L, dtype=np.float32)
    if L >= 2:
        dkd[1:] = kd[1:] - kd[:-1]

    onehot = np.zeros((L, 20), dtype=np.float32)
    blosum = np.zeros((L, 20), dtype=np.float32)
    zsc    = np.zeros((L, 5),  dtype=np.float32)
    for i, a in enumerate(seq):
        j = AA2IDX.get(a)
        if j is not None:
            onehot[i, j] = 1.0
        blosum[i, :] = _aa_blosum(a)
        zsc[i, :]    = _aa_zscale(a)

    X = np.concatenate([
        kd[:, None], ch[:, None], dkd[:, None],
        onehot,
        blosum,
        zsc
    ], axis=1).astype(np.float32)
    return X  # [L,48]


def _extract_continuous_subspace(Xnode: np.ndarray) -> np.ndarray:
    if Xnode.shape[1] < 8:
        raise ValueError("Xnode dimension anomaly prevents extraction of continuous subspaces (expected to include at least KD/charge/dKD/Z-scale).")
    cont = np.concatenate([Xnode[:, 0:3], Xnode[:, -5:]], axis=1)  # [L, 8]
    return cont


def _edge_attr_from_nodes(i: int, j: int, kd: np.ndarray, ch: np.ndarray) -> List[float]:
    return [abs(i - j), kd[i] * kd[j], ch[i] * ch[j]]

def _edges_full(L: int, self_loops: bool) -> List[Tuple[int, int]]:
    pairs = []
    for i in range(L):
        for j in range(L):
            if not self_loops and i == j:
                continue
            pairs.append((i, j))
    return pairs

def _edges_window(L: int, w: int, self_loops: bool) -> List[Tuple[int, int]]:
    pairs = []
    w = max(1, int(w))
    for i in range(L):
        j0 = max(0, i - w)
        j1 = min(L, i + w + 1)
        for j in range(j0, j1):
            if not self_loops and i == j:
                continue
            pairs.append((i, j))
    return pairs

def _edges_knn_cont(Xcont: np.ndarray, k: int) -> List[Tuple[int, int]]:

    L = Xcont.shape[0]
    if L <= 1:
        return []
    k = min(int(k), L - 1)
    if k <= 0:
        return []


    norm2 = np.sum(Xcont * Xcont, axis=1, keepdims=True)
    d2 = norm2 + norm2.T - 2.0 * (Xcont @ Xcont.T)
    np.fill_diagonal(d2, np.inf)

    nn_idx = np.argpartition(d2, kth=k, axis=1)[:, :k]
    pairs = []
    for i in range(L):
        for j in nn_idx[i]:
            pairs.append((i, int(j)))
    return pairs


def seq_to_edges(
    seq: str,
    Xnode: np.ndarray,
    edge_mode: str = "window",
    window: int = 3,
    knn_k: int = 8,
    self_loops: bool = True
) -> Tuple[np.ndarray, np.ndarray]:

    L = len(seq)
    kd = Xnode[:, 0]
    ch = Xnode[:, 1]

    if L == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 3), dtype=np.float32)

    pairs: List[Tuple[int, int]] = []

    if edge_mode == "full":
        pairs = _edges_full(L, self_loops)

    elif edge_mode == "window":
        pairs = _edges_window(L, window, self_loops)

    elif edge_mode == "knn":
        Xcont = _extract_continuous_subspace(Xnode)
        pairs = _edges_knn_cont(Xcont, knn_k)

    elif edge_mode == "hybrid":
        pw = _edges_window(L, window, self_loops)
        Xcont = _extract_continuous_subspace(Xnode)
        pk = _edges_knn_cont(Xcont, knn_k)
        pair_set = set(pw)
        pair_set.update(pk)
        pairs = list(pair_set)

    else:
        raise ValueError(f"Unknown edge_mode: {edge_mode}")

    if self_loops:
        base_set = set(pairs)
        for i in range(L):
            base_set.add((i, i))
        pairs = list(base_set)

    src, dst, feats = [], [], []
    for (i, j) in pairs:
        src.append(i)
        dst.append(j)
        feats.append(_edge_attr_from_nodes(i, j, kd, ch))

    edge_index = np.vstack([np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)])  # [2,E]
    edge_attr  = np.array(feats, dtype=np.float32)                                          # [E,3]
    return edge_index, edge_attr


def build_peptide_graph(
    seq: str,
    pH: float = 7.4,
    edge_mode: str = "window",
    window: int = 3,
    knn_k: int = 8,
    as_pyg: Optional[bool] = None,
    self_loops: bool = True
):

    X = seq_to_node_features(seq, pH=pH)  # [L,48]
    edge_index, edge_attr = seq_to_edges(
        seq, X, edge_mode=edge_mode, window=window, knn_k=knn_k, self_loops=self_loops
    )
    x_t  = torch.from_numpy(X).float()
    ei_t = torch.from_numpy(edge_index).long()
    ea_t = torch.from_numpy(edge_attr).float()

    if as_pyg is None:
        as_pyg = _HAS_PYG
    if as_pyg and _HAS_PYG:
        g = Data(x=x_t, edge_index=ei_t, edge_attr=ea_t)
        g.seq = seq
        return g
    else:
        return {"x": x_t, "edge_index": ei_t, "edge_attr": ea_t, "seq": seq}


def read_fasta_with_labels(file_path: str):
    names, seqs, labels = [], [], []
    name, acc = None, []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith('>'):
                if name is not None:
                    seqs.append(''.join(acc)); acc = []
                name = s[1:]; names.append(name)
                labels.append(1 if 'pos' in s else 0)
            else:
                acc.append(s)
        if name is not None:
            seqs.append(''.join(acc))
    return names, seqs, labels

def graphs_from_fasta(
    fasta_path: str,
    pH: float = 7.4,
    edge_mode: str = "window",
    window: int = 3,
    knn_k: int = 8,
    as_pyg: Optional[bool] = None
):
    names, seqs, labels = read_fasta_with_labels(fasta_path)
    graphs = [
        build_peptide_graph(
            s, pH=pH, edge_mode=edge_mode, window=window, knn_k=knn_k, as_pyg=as_pyg
        ) for s in seqs
    ]
    return graphs, np.array(labels, dtype=np.int64), names


def pyg_batch(graph_list: List[Data]):
    if not _HAS_PYG:
        raise RuntimeError("torch_geometric uninstall")
    return Batch.from_data_list(graph_list)
