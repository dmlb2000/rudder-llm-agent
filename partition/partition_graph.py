import argparse
import time
import os
import numpy as np
import scipy.sparse as sp

import dgl
import dgl.distributed
import torch as th
from dgl.data import RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset

def always_yes(*args, **kwargs):
    return 'y'

original_input = __builtins__.input
__builtins__.input = always_yes


def load_reddit(self_loop=True):
    """Load reddit dataset."""
    data = RedditDataset(self_loop=self_loop)
    g = data[0]
    g.ndata["features"] = g.ndata.pop("feat")
    g.ndata["labels"] = g.ndata.pop("label")
    return g, data.num_classes


def load_ogb(name, root):
    """Load ogbn dataset."""
    data = DglNodePropPredDataset(name=name, root=root)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    graph.ndata["features"] = graph.ndata.pop("feat")
    graph.ndata["labels"] = labels
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    train_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata["train_mask"] = train_mask
    graph.ndata["val_mask"] = val_mask
    graph.ndata["test_mask"] = test_mask
    return graph, num_labels


def load_photo_from_npz(root, use_context=False):
    """
    Load photo dataset from NPZ files (CSR format).
    
    Args:
        root: Directory containing photo_v.npz (primary embeddings) and photo_c.npz (context embeddings)
        use_context: If True, use context embeddings (C); otherwise use primary embeddings (V)
    
    Returns:
        graph, num_classes
    """
    npz_file = os.path.join(root, "photo_c.npz" if use_context else "photo_v.npz")
    
    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"Expected {npz_file}")
    
    print(f"Loading photo dataset from {npz_file}")
    data = np.load(npz_file)
    
    # Extract CSR components
    rowptr = data["rowptr"].astype(np.int32)
    tails = data["tails"].astype(np.int32)
    weights = data["weights"]
    features = th.from_numpy(data["features"]).float()
    labels = th.from_numpy(data["labels"]).long()
    num_nodes_arr = data["num_nodes"]
    num_nodes = int(num_nodes_arr.item() if hasattr(num_nodes_arr, 'item') else num_nodes_arr[0])
    
    print(f"  Nodes: {num_nodes}, Features shape: {features.shape}, Labels shape: {labels.shape}")
    print(f"  CSR edges: {len(tails)}")
    
    # Build CSR sparse matrix, then convert to COO for DGL
    csr = sp.csr_matrix(
        (weights, tails, rowptr),
        shape=(num_nodes, num_nodes)
    )
    coo = csr.tocoo()
    
    # Create DGL graph
    g = dgl.graph((th.from_numpy(coo.row), th.from_numpy(coo.col)))
    g.ndata["features"] = features
    g.ndata["labels"] = labels
    
    # Create train/val/test masks (60/20/20 split)
    num_train = int(0.6 * num_nodes)
    num_val = int(0.2 * num_nodes)
    
    train_mask = th.zeros(num_nodes, dtype=th.bool)
    val_mask = th.zeros(num_nodes, dtype=th.bool)
    test_mask = th.zeros(num_nodes, dtype=th.bool)
    
    train_mask[:num_train] = True
    val_mask[num_train:num_train + num_val] = True
    test_mask[num_train + num_val:] = True
    
    g.ndata["train_mask"] = train_mask
    g.ndata["val_mask"] = val_mask
    g.ndata["test_mask"] = test_mask
    
    num_classes = len(th.unique(labels))
    return g, num_classes


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Partition graph")
    argparser.add_argument(
        "--dataset",
        type=str,
        default="reddit",
        help="datasets: reddit, ogbn-products, ogbn-papers100M, ogbn-arxiv, photo, photo-context",
    )
    argparser.add_argument(
        "--num_parts", type=int, default=4, help="number of partitions"
    )
    argparser.add_argument(
        "--part_method", type=str, default="metis", help="the partition method"
    )
    argparser.add_argument(
        "--balance_train",
        action="store_true",
        help="balance the training size in each partition.",
    )
    argparser.add_argument(
        "--undirected",
        action="store_true",
        help="turn the graph into an undirected graph.",
    )
    argparser.add_argument(
        "--balance_edges",
        action="store_true",
        help="balance the number of edges in each partition.",
    )
    argparser.add_argument(
        "--num_trainers_per_machine",
        type=int,
        default=1,
        help="the number of trainers per machine. The trainer ids are stored\
                                in the node feature 'trainer_id'",
    )
    argparser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output path of partitioned graph.",
    )
    argparser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="The directory containing the dataset.",
    )
    args = argparser.parse_args()
    print("Arguments:", args)
    start = time.time()
    if args.dataset == "reddit":
        g, _ = load_reddit()
    elif args.dataset in ["ogbn-products", "ogbn-papers100M", "ogbn-arxiv"]:
        g, _ = load_ogb(args.dataset, args.dataset_dir)
    elif args.dataset == "photo":
        # Use primary embeddings (V)
        g, _ = load_photo_from_npz(args.dataset_dir, use_context=False)
    elif args.dataset == "photo-context":
        # Use context embeddings (C)
        g, _ = load_photo_from_npz(args.dataset_dir, use_context=True)
    else:
        raise RuntimeError(f"Unknown dataset: {args.dataset}")
    print(
        "Load {} takes {:.3f} seconds".format(args.dataset, time.time() - start)
    )
    print("|V|={}, |E|={}".format(g.num_nodes(), g.num_edges()))
    print(
        "train: {}, valid: {}, test: {}".format(
            th.sum(g.ndata["train_mask"]),
            th.sum(g.ndata["val_mask"]),
            th.sum(g.ndata["test_mask"]),
        )
    )
    if args.balance_train:
        balance_ntypes = g.ndata["train_mask"]
    else:
        balance_ntypes = None

    if args.undirected:
        sym_g = dgl.to_bidirected(g, readonly=True)
        for key in g.ndata:
            sym_g.ndata[key] = g.ndata[key]
        g = sym_g

    g = g.long()
    dgl.distributed.partition_graph(
        g,
        args.dataset,
        args.num_parts,
        args.output,
        part_method=args.part_method,
        balance_ntypes=balance_ntypes,
        balance_edges=args.balance_edges,
        num_trainers_per_machine=args.num_trainers_per_machine,
    )