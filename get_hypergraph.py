from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
import os
import os.path as osp
import scipy.sparse as sp
import json
import pickle
import torch_geometric as pyg
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from datetime import datetime
from scipy.sparse import csr_matrix, issparse

from utils import *

def get_hypergraph(processed_file_path, path='data/raw_data', dataset='iAF1260b',
                   train_size=0.5, val_size=0.25, s_values=[1]):

    def log_message(msg):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

    log_message(f"Starting preprocessing for dataset: {dataset}")

    # Load node features
    feature_path = f'./{path}/{dataset}/{dataset}.pt'
    log_message(f"Loading node features from: {feature_path}")
    node_features = torch.load(feature_path, map_location=torch.device('cpu')) # Use map_location for safety
    log_message(f"Node features loaded. Shape: {node_features.shape}")

    # Load raw incidence matrix from .mat
    mat_path = f'./{path}/{dataset}/{dataset}.mat'
    log_message(f"Loading incidence matrix from: {mat_path}")
    data = loadmat(mat_path)
    H = data[dataset]['S']
    if isinstance(H, np.ndarray) and H.shape == (1, 1):
        H = H[0, 0]
    log_message(f"Incidence matrix loaded. Shape: {H.shape}")

    incidence_matrix = np.where(H > 0, 1, np.where(H < 0, 1, 0)).astype(np.float32)
    incidence_matrix = torch.tensor(incidence_matrix, dtype=torch.float)

    # Negative sampling
    log_message("Starting negative sampling...")
    combined_H, num_neg_edges = negative_sampling(H) # combined_H shape: (#node, #pos_edges + #neg_edges)
    combined_H = torch.tensor(combined_H, dtype=torch.float)
    log_message(f"Negative sampling finished. Combined_H shape: {combined_H.shape}")

    num_pos_edges = H.shape[1]
    total_edges = combined_H.shape[1]

    full_incidence_matrix = torch.tensor(np.where(combined_H > 0, 1, np.where(combined_H < 0, 1, 0)).astype(np.float32), dtype=torch.float)

    # --- THIS IS THE SUSPECTED AREA ---
    log_message("Starting diff_pooling to calculate edge features...")
    edge_features = diff_pooling(node_features, combined_H)
    log_message(f"Finished diff_pooling. Edge features calculated. Shape: {edge_features.shape}")
    # --- END OF SUSPECTED AREA ---

    dist_mats = {}
    norm_mats = {}

    for s in s_values:
        log_message(f"Starting distance calculations for s={s}...")
        
        log_message(f"--> Calling shortest_edge_distances for s={s}...")
        edge_dist = shortest_edge_distances(incidence_matrix, full_incidence_matrix, s=s)
        log_message(f"--> Finished shortest_edge_distances for s={s}. Matrix shape: {edge_dist.shape}")

        # Build a normalization matrix
        log_message(f"--> Building normalization matrix for s={s}...")
        normalization_matrix = edge_dist.clone()
        for i, entry in enumerate(edge_dist):
            distances_counts = torch.unique(entry, return_counts=True)
            if len(distances_counts[0]) > 0:
                normalization_matrix[i].apply_(
                    lambda x: distances_counts[1][(distances_counts[0] == x).nonzero().item()])
        log_message(f"--> Finished normalization matrix for s={s}.")
        
        # Optionally invert the distances (avoid zero-dist by adding 1)
        edge_dist_inv = edge_dist.clone()
        edge_dist_inv += 1
        edge_dist_inv = 1.0 / edge_dist_inv

        dist_mats[f's{s}'] = edge_dist_inv
        norm_mats[f's{s}'] = normalization_matrix
        log_message(f"Finished all calculations for s={s}.")

    # Create labels & do train/val/test splits
    log_message("Creating labels and train/validation/test splits...")
    labels = np.concatenate([np.ones(num_pos_edges), np.zeros(num_neg_edges)])
    assert len(labels) == total_edges, "Mismatch in label length"

    indices = np.arange(total_edges)
    train_val_size = train_size + val_size
    test_size = 1 - train_val_size
    val_ratio = val_size / train_val_size

    train_val_idx, test_idx, train_val_y, test_y = train_test_split(
        indices, labels, test_size=test_size, stratify=labels, random_state=42
    )
    train_idx, val_idx, train_y, val_y = train_test_split(
        train_val_idx, train_val_y, test_size=val_ratio, stratify=train_val_y, random_state=42
    )

    # Build boolean masks
    train_mask = torch.zeros(total_edges, dtype=torch.bool)
    test_mask = torch.zeros(total_edges, dtype=torch.bool)

    train_mask[train_idx] = True
    test_mask[test_idx] = True
    log_message("Splits created successfully.")

    # Pack everything into a Data object
    log_message("Packing data into PyG Data object...")
    hypergraph = Data(
        x=torch.tensor(edge_features, dtype=torch.float),  
        h=combined_H,                      
        y=torch.tensor(labels, dtype=torch.float),        
        dist_mats=dist_mats,                        
        norm_mats=norm_mats,           
        train_mask=train_mask,
        test_mask=test_mask
    )
    log_message("Data object created.")

    if not os.path.exists(f'{processed_file_path}'):
      log_message(f"Saving preprocessed data to '{processed_file_path}'...")
      torch.save(hypergraph, f'{processed_file_path}')
      log_message(f"Saved preprocessed {dataset} dataset.")

    log_message(f"Finished preprocessing for dataset: {dataset}")
    return hypergraph

def load_LE_dataset(processed_file_path, path='data/raw_data', data_name="Mushroom", train_size=0.5, val_size=0.25, s_values=[1]):
    # adapted from YOU ARE ALLSET: A MULTISET LEARNING FRAMEWORK FOR HYPERGRAPH NEURAL NETWORKS, https://github.com/jianhao2016/AllSet
    # load edges, features, and labels.
    print('Loading {} dataset...'.format(data_name))

    file_name = f'{data_name}.content'
    p2idx_features_labels = osp.join(path, data_name, file_name)
    idx_features_labels = np.genfromtxt(p2idx_features_labels,dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = torch.LongTensor(idx_features_labels[:, -1].astype(float))
    if data_name == 'zoo':
        labels = labels - 1 

    print('load features')

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    file_name = f'{data_name}.edges'
    p2edges_unordered = osp.join(path, data_name, file_name)
    edges_unordered = np.genfromtxt(p2edges_unordered,
                                    dtype=np.int32)
    print(f'edges_unordered: {edges_unordered}')
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=np.int32).reshape(edges_unordered.shape)
    output_file = "edges.txt"
    np.savetxt(output_file, edges, fmt='%d', delimiter=' ')
    print('load edges')
    print(f"edges:{edges}")

    # From adjacency matrix to edge_list
    edge_index = edges.T

    assert edge_index[0].max() == edge_index[1].min() - 1

    # check if values in edge_index is consecutive. i.e. no missing value for node_id/he_id.
    assert len(np.unique(edge_index)) == edge_index.max() + 1

    num_nodes = edge_index[0].max() + 1
    num_hyperedges = edge_index[1].max() - num_nodes + 1

    edge_index = np.hstack((edge_index, edge_index[::-1, :]))
    edge_index = torch.from_numpy(edge_index)

    H = torch.zeros((num_nodes, num_hyperedges), dtype=torch.float32)
    print(f'H.shape:{H.shape}')

    for i in range(edge_index.shape[1]//2):
        v = edge_index[0, :][i]
        e = edge_index[1, :][i] - num_nodes
        H[v, e] = 1
    print(f"num_class:{len(np.unique(labels[:num_nodes].numpy()))}")

    features = torch.FloatTensor(np.array(features[:num_nodes].todense()))

    dist_mats = {}
    norm_mats = {}
    for s in s_values:
        print(f'Calculating node distances for s={s}...')
        node_distances = shortest_node_distances(H, s=s)

        normalization_matrix = node_distances.clone()
        for i, entry in enumerate(node_distances):
            distances_counts = torch.unique(entry, return_counts=True)
            if len(distances_counts[0]) > 0:
                normalization_matrix[i] = normalization_matrix[i].cpu().apply_(
                    lambda x: distances_counts[1][(distances_counts[0].float() == x).nonzero().item()]
                )

        node_distances_inv = 1 / node_distances
        
        dist_mats[f's{s}'] = torch.tensor(node_distances_inv, dtype=torch.float).squeeze(0)
        norm_mats[f's{s}'] = torch.tensor(normalization_matrix, dtype=torch.float).squeeze(0)
        print(f"Finished calculating distances for s={s}!")

    labels = labels[:num_nodes]

    indices = np.arange(num_nodes)
    train_val_size = train_size + val_size
    test_size = 1 - train_val_size
    val_ratio = val_size / train_val_size

    train_val_idx, test_idx, train_val_y, test_y = train_test_split(
        indices, labels, test_size=test_size, stratify=labels, random_state=42
    )

    train_idx, val_idx, train_y, val_y = train_test_split(
        train_val_idx, train_val_y, test_size=val_ratio, stratify=train_val_y, random_state=42
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    # build torch data class
    data = Data(x=features, 
                y=labels,
                edge_index=edge_index,
                h=H,
                dist_mats=dist_mats,
                norm_mats=norm_mats,
                n_x=num_nodes,
                num_hyperedges=num_hyperedges,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask)
    
    if not os.path.exists(f'{processed_file_path}'):
        torch.save(data, f'{processed_file_path}')
        print(f'Saved preprocessed {data_name} dataset')
    return data


def build_incidence_matrix(hyperedges_dict, *, sort_edges=True, dtype=np.int8):
    """
    将 {hyperedge_id: [node_id, ...]} 转为稀疏incidence matrix (CSR)。
    矩阵形状： (#nodes, #hyperedges)，A[i, j] = 1 表示节点 i 属于超边 j。
    返回: A(node×edge, csr), node_index(dict), edge_index(dict)
    """
    # 1) 收集全部节点ID
    all_nodes = set()
    for nodes in hyperedges_dict.values():
        all_nodes.update(nodes)
    all_nodes = sorted(all_nodes)
    node_index = {nid: i for i, nid in enumerate(all_nodes)}

    # 2) 决定超边列的顺序
    edge_ids = list(hyperedges_dict.keys())
    if sort_edges:
        try:
            edge_ids = sorted(edge_ids, key=lambda x: int(x))
        except Exception:
            edge_ids = sorted(edge_ids)
    edge_index = {eid: j for j, eid in enumerate(edge_ids)}

    # 3) 组装COO
    rows, cols, data = [], [], []
    for eid in edge_ids:
        j = edge_index[eid]
        for nid in set(hyperedges_dict[eid]):  # 0/1 incidence
            i = node_index[nid]
            rows.append(i); cols.append(j); data.append(1)

    n_rows, n_cols = len(all_nodes), len(edge_ids)
    A = csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols), dtype=dtype)
    return A, node_index, edge_index



def build_feature_matrix(feature: np.ndarray, node_index: dict) -> np.ndarray:
    num_nodes = len(node_index)
    X = np.zeros((num_nodes, feature.shape[1]), dtype=feature.dtype)
    for nid, row_idx in node_index.items():
        X[row_idx, :] = feature[nid, :]
    return X

def build_label_vector(y: np.ndarray, node_index: dict) -> np.ndarray:
    num_nodes = len(node_index)
    y_new = np.zeros(num_nodes, dtype=y.dtype)
    for nid, row_idx in node_index.items():
        y_new[row_idx] = y[nid]
    return y_new

def load_heter_dataset(
    processed_file_path,
    path: str = 'data/raw_data',
    data_name: str = 'amazon',
    compute_distances: bool = True,
    s_values=(1,2,3),
):
    def log(msg: str):
        print(f"[load_heter_dataset] {msg}")

    log(f"Loading {data_name} dataset from pickles in {osp.join(path, data_name)} ...")

    with open(osp.join(path, data_name, 'features.pickle'), 'rb') as f:
        features = pickle.load(f)
    if issparse(features):
        features = features.todense()
    features = np.asarray(features)  # (N_raw, F)
    num_nodes_raw, feature_dim = features.shape

    with open(osp.join(path, data_name, 'labels.pickle'), 'rb') as f:
        labels = pickle.load(f)
    labels = np.asarray(labels).reshape(-1)
    assert num_nodes_raw == len(labels), \
        f"features number({num_nodes_raw}) is not aligned with label length({len(labels)})."
    
    with open(osp.join(path, data_name, 'hypergraph.pickle'), 'rb') as f:
        hypergraph = pickle.load(f)  # { he_id: [nodes], ... }

    H_csr, node_index, edge_index = build_incidence_matrix(
        hypergraph, sort_edges=True, dtype=np.int8
    )
    num_nodes, num_hyperedges = H_csr.shape

    max_node_id = max(node_index.keys()) if len(node_index) else -1
    assert max_node_id < num_nodes_raw, \
        f"Node ID({max_node_id}) exceeds features/labels range ({num_nodes_raw})"

    X = build_feature_matrix(features, node_index)
    y_aligned = build_label_vector(labels, node_index)

    x = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y_aligned, dtype=torch.long)
    H = torch.as_tensor(H_csr.toarray(), dtype=torch.float32) 

    dist_mats = {}
    norm_mats = {}

    if compute_distances and s_values:
        def _row_value_counts_replace(D: torch.Tensor) -> torch.Tensor:
            N = D.size(0)
            out = torch.empty_like(D)
            Dc = D.detach().cpu()
            outc = out.detach().cpu()
            for i in range(N):
                row = Dc[i]
                vals, cnts = torch.unique(row, return_counts=True)
                idx = (row.unsqueeze(1) == vals.unsqueeze(0)).to(torch.int64).argmax(dim=1)
                outc[i] = cnts[idx].to(row.dtype)
            return outc.to(D.dtype)

        for s in s_values:
            try:
                log(f"Computing shortest_node_distances with s={s} ...")
                D = shortest_node_distances(H, s=int(s)).to(torch.float32)

                norm_mat = _row_value_counts_replace(D)

                D_inv = 1.0 / D
                D_inv = torch.nan_to_num(D_inv, nan=0.0, posinf=0.0, neginf=0.0)

                dist_mats[f's{s}'] = D_inv
                norm_mats[f's{s}'] = norm_mat
                log(f"Done s={s}: dist_mats['s{s}'].shape={tuple(D_inv.shape)}")
            except Exception as e:
                log(f"[WARN] Encount error when computing distance mat for s={s}: {e}")

    data_kwargs = dict(
        x=x,
        y=y,
        h=H,
        n_x=num_nodes,
        num_hyperedges=num_hyperedges,
    )
    if dist_mats and norm_mats:
        data_kwargs.update(dict(
            dist_mats=dist_mats,
            norm_mats=norm_mats
        ))
    data = Data(**data_kwargs)

    out_path = f'{processed_file_path}'
    save_dir = os.path.dirname(out_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if not osp.exists(out_path):
        torch.save(data, out_path)
        log(f"Saved to {out_path}")
    else:
        log(f"[INFO] File already exists: {out_path}")

    log(f"Done. num_nodes={num_nodes}, num_hyperedges={num_hyperedges}, "
        f"x={tuple(x.shape)}, y={tuple(y.shape)}, H={tuple(H.shape)}")
    if dist_mats:
        keys = ','.join(sorted(dist_mats.keys()))
        log(f"dist_mats / norm_mats keys: {keys}")
    return data