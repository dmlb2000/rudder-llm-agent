#!/usr/bin/env python3
import argparse
import os
import re

import dgl
import dgl.distributed
import numpy as np
import torch as th


def always_yes(*args, **kwargs):
    return "y"


original_input = __builtins__.input
__builtins__.input = always_yes


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build and partition a DistDGL graph from NOMAD graph artifacts so "
            "Rudder trains on the same node universe NOMAD used."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Graph name to store in the DistDGL partition metadata.",
    )
    parser.add_argument(
        "--nomad_graph_bin",
        required=True,
        help="Path to NOMAD CSR graph .bin used to train embeddings.",
    )
    parser.add_argument(
        "--idmap",
        required=True,
        help="Path to NOMAD idmap file with lines: <original_id> <nomad_id>.",
    )
    parser.add_argument(
        "--ogb_root",
        help="OGB dataset root used to load labels and splits in OGB mode.",
    )
    parser.add_argument(
        "--label_file",
        help="Custom labels with one '<original_id> <class_id>' pair per line.",
    )
    parser.add_argument(
        "--label_id_space",
        choices=["original", "nomad"],
        default="original",
        help=(
            "ID space used by --label_file (default: original). Use 'nomad' "
            "for .label files already remapped by NOMAD converter.py."
        ),
    )
    parser.add_argument(
        "--train_idx",
        help="Custom training split with one original node ID per line.",
    )
    parser.add_argument(
        "--val_idx",
        help="Custom validation split with one original node ID per line.",
    )
    parser.add_argument(
        "--test_idx",
        help="Custom test split with one original node ID per line.",
    )
    parser.add_argument(
        "--split_ratios",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VAL", "TEST"),
        help=(
            "Generate custom splits from labeled nodes using three ratios. "
            "Accepts fractions such as '0.8 0.1 0.1' or percentages such as "
            "'80 10 10'. Cannot be combined with explicit split files."
        ),
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Random seed used with --split_ratios (default: 42).",
    )
    parser.add_argument(
        "--nomad_features",
        help="Optional NOMAD_V.*.bin to use as g.ndata['features'].",
    )
    parser.add_argument(
        "--feature_mode",
        choices=["nomad", "zeros"],
        default="nomad",
        help="How to initialize g.ndata['features'].",
    )
    parser.add_argument(
        "--num_parts",
        type=int,
        required=True,
        help="Number of DistDGL partitions to create.",
    )
    parser.add_argument(
        "--part_method",
        type=str,
        default="metis",
        help="Partition method passed to dgl.distributed.partition_graph.",
    )
    parser.add_argument(
        "--balance_train",
        action="store_true",
        help="Balance the training size in each partition.",
    )
    parser.add_argument(
        "--balance_edges",
        action="store_true",
        help="Balance the number of edges in each partition.",
    )
    parser.add_argument(
        "--num_trainers_per_machine",
        type=int,
        default=1,
        help="Number of trainers per machine.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for the DistDGL partitioned graph.",
    )
    parser.add_argument(
        "--undirected",
        action="store_true",
        help="Convert the built graph to a bidirected graph before partitioning.",
    )
    args = parser.parse_args()
    split_files = [args.train_idx, args.val_idx, args.test_idx]
    if any(split_files) and not all(split_files):
        parser.error(
            "explicit custom splits require --train_idx, --val_idx, and --test_idx"
        )
    if args.split_ratios and any(split_files):
        parser.error("--split_ratios cannot be combined with explicit split files")
    custom_mode = bool(args.label_file)
    if custom_mode and not (all(split_files) or args.split_ratios):
        parser.error(
            "custom mode requires either all three split files or --split_ratios"
        )
    if (all(split_files) or args.split_ratios) and not custom_mode:
        parser.error("custom split options require --label_file")
    if not args.ogb_root and not custom_mode:
        parser.error(
            "provide --ogb_root for an OGB dataset, or --label_file with custom splits"
        )
    return args


def load_idmap(path: str):
    old_to_nomad = {}
    nomad_to_old = {}
    with open(path, "r", encoding="utf-8") as infile:
        for line_no, line in enumerate(infile, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(f"Malformed idmap line {line_no}: {stripped!r}")
            old_id = int(parts[0])
            nomad_id = int(parts[1])
            old_to_nomad[old_id] = nomad_id
            nomad_to_old[nomad_id] = old_id
    if not old_to_nomad:
        raise ValueError(f"No mappings found in {path}")
    return old_to_nomad, nomad_to_old


def parse_nomad_shape(path: str):
    fname = os.path.basename(path)
    match = re.match(r"NOMAD_[VC]\.(\d+)-(\d+)(?:-|\.|$)", fname)
    if not match:
        raise ValueError(
            f"Could not parse NOMAD embedding shape from filename '{fname}'."
        )
    return int(match.group(1)), int(match.group(2))


def load_nomad_embedding_memmap(path: str):
    num_nodes, dim = parse_nomad_shape(path)
    with open(path, "rb") as infile:
        header = np.fromfile(infile, dtype=np.int64, count=1)
    if header.size != 1:
        raise ValueError(f"Missing header in embedding file {path}")
    total = int(header[0])
    expected = num_nodes * dim
    if total != expected:
        raise ValueError(
            f"Embedding header mismatch: header={total}, expected={expected}"
        )
    matrix = np.memmap(path, mode="r", dtype=np.float32, offset=8, shape=(num_nodes, dim))
    return matrix, num_nodes, dim


def load_nomad_csr(path: str):
    with open(path, "rb") as infile:
        num_nodes_arr = np.fromfile(infile, dtype=np.int64, count=1)
        num_edges_arr = np.fromfile(infile, dtype=np.int64, count=1)
        if num_nodes_arr.size != 1 or num_edges_arr.size != 1:
            raise ValueError(f"Missing NOMAD CSR header in {path}")
        num_nodes = int(num_nodes_arr[0])
        num_edges = int(num_edges_arr[0])
        rowptr = np.fromfile(infile, dtype=np.int64, count=num_nodes + 1)
        edge_dtype = np.dtype([("tail", np.int64), ("weight", np.float64)])
        edges = np.fromfile(infile, dtype=edge_dtype, count=num_edges)

    if rowptr.size != num_nodes + 1:
        raise ValueError(
            f"Expected rowptr length {num_nodes + 1}, got {rowptr.size} in {path}"
        )
    if edges.size != num_edges:
        raise ValueError(
            f"Expected {num_edges} edge records, got {edges.size} in {path}"
        )

    src = np.repeat(np.arange(num_nodes, dtype=np.int64), np.diff(rowptr))
    dst = edges["tail"].astype(np.int64, copy=False)
    return num_nodes, src, dst


def build_graph_from_nomad_bin(path: str):
    num_nodes, src, dst = load_nomad_csr(path)
    graph = dgl.graph((th.from_numpy(src), th.from_numpy(dst)), num_nodes=num_nodes)
    graph.ndata["nomad_id"] = th.arange(num_nodes, dtype=th.int64)
    return graph


def attach_original_ids(graph, nomad_to_old: dict):
    num_nodes = graph.num_nodes()
    orig_ids = np.empty(num_nodes, dtype=np.int64)
    for nomad_id in range(num_nodes):
        if nomad_id not in nomad_to_old:
            raise ValueError(f"Missing inverse idmap entry for NOMAD node {nomad_id}")
        orig_ids[nomad_id] = nomad_to_old[nomad_id]
    graph.ndata["orig_id"] = th.from_numpy(orig_ids)
    return orig_ids


def read_id_value_file(path: str):
    values = {}
    with open(path, "r", encoding="utf-8") as infile:
        for line_no, line in enumerate(infile, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{line_no}: expected '<original_id> <class_id>'"
                )
            node_id, value = map(int, parts)
            if node_id in values:
                raise ValueError(f"{path}:{line_no}: duplicate node ID {node_id}")
            values[node_id] = value
    if not values:
        raise ValueError(f"No labels found in {path}")
    return values


def read_id_file(path: str):
    node_ids = []
    seen = set()
    with open(path, "r", encoding="utf-8") as infile:
        for line_no, line in enumerate(infile, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) != 1:
                raise ValueError(f"{path}:{line_no}: expected one original node ID")
            node_id = int(parts[0])
            if node_id in seen:
                raise ValueError(f"{path}:{line_no}: duplicate node ID {node_id}")
            seen.add(node_id)
            node_ids.append(node_id)
    if not node_ids:
        raise ValueError(f"No node IDs found in {path}")
    return node_ids


def remap_split(name: str, original_ids, old_to_nomad: dict, labels: dict):
    missing_labels = [node_id for node_id in original_ids if node_id not in labels]
    if missing_labels:
        raise ValueError(
            f"{name} split has {len(missing_labels)} nodes without labels; "
            f"sample={missing_labels[:10]}"
        )
    kept = [old_to_nomad[node_id] for node_id in original_ids if node_id in old_to_nomad]
    dropped = len(original_ids) - len(kept)
    if dropped:
        print(f"Custom {name} split: dropped {dropped} nodes absent from NOMAD idmap")
    if not kept:
        raise ValueError(f"Custom {name} split is empty after applying the NOMAD idmap")
    return kept


def generate_splits(label_by_orig: dict, old_to_nomad: dict, ratios, seed: int):
    ratios = np.asarray(ratios, dtype=np.float64)
    if np.any(ratios <= 0):
        raise ValueError("All --split_ratios values must be greater than zero")
    total = float(ratios.sum())
    if np.isclose(total, 100.0):
        ratios /= 100.0
    elif not np.isclose(total, 1.0):
        raise ValueError(
            f"--split_ratios must sum to 1 or 100; got {total:g}"
        )

    eligible = np.asarray(
        sorted(node_id for node_id in label_by_orig if node_id in old_to_nomad),
        dtype=np.int64,
    )
    if eligible.size < 3:
        raise ValueError(
            "At least three labeled nodes surviving the NOMAD idmap are required "
            "to generate train/val/test splits"
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(eligible)
    train_end = int(np.floor(eligible.size * ratios[0]))
    val_end = train_end + int(np.floor(eligible.size * ratios[1]))
    splits = {
        "train": eligible[:train_end].tolist(),
        "val": eligible[train_end:val_end].tolist(),
        "test": eligible[val_end:].tolist(),
    }
    if any(not ids for ids in splits.values()):
        counts = {name: len(ids) for name, ids in splits.items()}
        raise ValueError(f"Generated split is empty; counts={counts}")
    print(
        "Generated custom splits: "
        f"seed={seed} train={len(splits['train'])} "
        f"val={len(splits['val'])} test={len(splits['test'])}"
    )
    return splits


def attach_custom_labels_and_splits(graph, args, old_to_nomad: dict, nomad_to_old: dict):
    orig_ids = attach_original_ids(graph, nomad_to_old)
    label_by_orig = read_id_value_file(args.label_file)
    if args.label_id_space == "nomad":
        unknown = [node_id for node_id in label_by_orig if node_id not in nomad_to_old]
        if unknown:
            raise ValueError(
                f"Label file has {len(unknown)} IDs outside the NOMAD idmap; "
                f"sample={unknown[:10]}"
            )
        label_by_orig = {
            nomad_to_old[nomad_id]: label
            for nomad_id, label in label_by_orig.items()
        }
        print("Converted custom label IDs from NOMAD IDs to original IDs")

    class_ids = sorted(set(label_by_orig.values()))
    if class_ids != list(range(len(class_ids))):
        raise ValueError(
            "Custom class IDs must be contiguous and zero-based; "
            f"found={class_ids[:20]}"
        )

    # Unsupervised nodes use class 0 but are excluded by all split masks.
    labels = th.zeros(graph.num_nodes(), dtype=th.int64)
    for nomad_id, orig_id in enumerate(orig_ids.tolist()):
        if orig_id in label_by_orig:
            labels[nomad_id] = label_by_orig[orig_id]

    if args.split_ratios:
        split_original = generate_splits(
            label_by_orig, old_to_nomad, args.split_ratios, args.split_seed
        )
    else:
        split_original = {
            "train": read_id_file(args.train_idx),
            "val": read_id_file(args.val_idx),
            "test": read_id_file(args.test_idx),
        }
    split_sets = {name: set(ids) for name, ids in split_original.items()}
    overlaps = {
        "train/val": split_sets["train"] & split_sets["val"],
        "train/test": split_sets["train"] & split_sets["test"],
        "val/test": split_sets["val"] & split_sets["test"],
    }
    for names, overlap in overlaps.items():
        if overlap:
            raise ValueError(
                f"Custom splits {names} overlap on {len(overlap)} nodes; "
                f"sample={sorted(overlap)[:10]}"
            )

    graph.ndata["labels"] = labels
    counts = {}
    for name, original_ids in split_original.items():
        nomad_ids = remap_split(name, original_ids, old_to_nomad, label_by_orig)
        mask = th.zeros(graph.num_nodes(), dtype=th.bool)
        mask[th.tensor(nomad_ids, dtype=th.int64)] = True
        graph.ndata[f"{name}_mask"] = mask
        counts[name] = int(mask.sum())

    print(
        "Attached custom labels/splits: "
        f"classes={len(class_ids)} train={counts['train']} "
        f"val={counts['val']} test={counts['test']}"
    )


def attach_ogb_labels_and_splits(graph, dataset_name: str, ogb_root: str, old_to_nomad: dict, nomad_to_old: dict):
    from ogb.nodeproppred import DglNodePropPredDataset

    data = DglNodePropPredDataset(name=dataset_name, root=ogb_root)
    splits = data.get_idx_split()
    _, labels = data[0]
    labels = labels[:, 0]

    num_nodes = graph.num_nodes()
    orig_ids = attach_original_ids(graph, nomad_to_old)
    graph.ndata["labels"] = labels[th.from_numpy(orig_ids)]

    train_mask = th.zeros((num_nodes,), dtype=th.bool)
    val_mask = th.zeros((num_nodes,), dtype=th.bool)
    test_mask = th.zeros((num_nodes,), dtype=th.bool)

    train_nomad = [old_to_nomad[int(node_id)] for node_id in splits["train"].tolist() if int(node_id) in old_to_nomad]
    val_nomad = [old_to_nomad[int(node_id)] for node_id in splits["valid"].tolist() if int(node_id) in old_to_nomad]
    test_nomad = [old_to_nomad[int(node_id)] for node_id in splits["test"].tolist() if int(node_id) in old_to_nomad]

    train_mask[th.tensor(train_nomad, dtype=th.int64)] = True
    val_mask[th.tensor(val_nomad, dtype=th.int64)] = True
    test_mask[th.tensor(test_nomad, dtype=th.int64)] = True

    graph.ndata["train_mask"] = train_mask
    graph.ndata["val_mask"] = val_mask
    graph.ndata["test_mask"] = test_mask

    print(
        "Attached labels/splits: "
        f"train={int(train_mask.sum())} val={int(val_mask.sum())} test={int(test_mask.sum())}"
    )


def attach_features(graph, args):
    if args.feature_mode == "zeros":
        if args.nomad_features:
            _, num_nodes, dim = load_nomad_embedding_memmap(args.nomad_features)
        else:
            raise ValueError("--feature_mode=zeros still requires --nomad_features to infer dim")
        if num_nodes != graph.num_nodes():
            raise ValueError(
                f"Feature file node count {num_nodes} != graph nodes {graph.num_nodes()}"
            )
        graph.ndata["features"] = th.zeros((num_nodes, dim), dtype=th.float32)
        return

    if not args.nomad_features:
        raise ValueError("--feature_mode=nomad requires --nomad_features")

    matrix, num_nodes, dim = load_nomad_embedding_memmap(args.nomad_features)
    if num_nodes != graph.num_nodes():
        raise ValueError(
            f"Feature file node count {num_nodes} != graph nodes {graph.num_nodes()}"
        )
    graph.ndata["features"] = th.from_numpy(np.asarray(matrix, dtype=np.float32))
    print(f"Attached NOMAD features: shape=({num_nodes}, {dim})")


def main():
    args = parse_args()
    print(f"Arguments: {args}")

    old_to_nomad, nomad_to_old = load_idmap(args.idmap)
    graph = build_graph_from_nomad_bin(args.nomad_graph_bin)
    print(f"Loaded NOMAD graph: |V|={graph.num_nodes()}, |E|={graph.num_edges()}")

    if len(old_to_nomad) != graph.num_nodes():
        raise ValueError(
            f"idmap entries ({len(old_to_nomad)}) != NOMAD graph nodes ({graph.num_nodes()})"
        )

    if args.label_file:
        attach_custom_labels_and_splits(graph, args, old_to_nomad, nomad_to_old)
    else:
        attach_ogb_labels_and_splits(
            graph,
            dataset_name=args.dataset,
            ogb_root=args.ogb_root,
            old_to_nomad=old_to_nomad,
            nomad_to_old=nomad_to_old,
        )
    attach_features(graph, args)

    if args.undirected:
        sym_graph = dgl.to_bidirected(graph, copy_ndata=True)
        graph = dgl.to_simple(sym_graph, copy_ndata=True)
        print(f"Converted to bidirected simple graph: |E|={graph.num_edges()}")

    balance_ntypes = graph.ndata["train_mask"] if args.balance_train else None

    print(
        "Partitioning graph with "
        f"num_parts={args.num_parts} method={args.part_method} output={args.output}"
    )
    dgl.distributed.partition_graph(
        graph,
        args.dataset,
        args.num_parts,
        args.output,
        part_method=args.part_method,
        balance_ntypes=balance_ntypes,
        balance_edges=args.balance_edges,
        num_trainers_per_machine=args.num_trainers_per_machine,
    )


if __name__ == "__main__":
    main()
