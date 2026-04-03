"""
inspect_pickle.py — General purpose pickle file inspector
Usage: python inspect_pickle.py <file.pkl>
"""

import pickle
import sys
import argparse
import numpy as np
from collections.abc import Mapping, Sequence


def truncate(s, max_len=80):
    s = str(s)
    return s if len(s) <= max_len else s[:max_len] + "..."


def type_summary(obj):
    """Return a compact type+shape/len string for an object."""
    t = type(obj).__name__
    if isinstance(obj, type):  # obj is a class itself, not an instance
        return f"type = {obj.__name__}"
    if hasattr(obj, "shape"):  # numpy arrays, tensors
        return f"{t}{list(obj.shape)} dtype={obj.dtype}"
    if hasattr(obj, "__len__") and not isinstance(obj, str):
        try:
            return f"{t}[{len(obj)}]"
        except TypeError:
            pass
    return f"{t} = {truncate(repr(obj))}"


def inspect(obj, indent=0, max_depth=5, max_items=5, _visited=None):
    """Recursively print the structure of any Python object."""
    if _visited is None:
        _visited = set()

    pad = "  " * indent
    obj_id = id(obj)

    if indent > max_depth:
        print(f"{pad}... (max depth reached)")
        return

    if obj_id in _visited and not isinstance(obj, (str, int, float, bool, type(None))):
        print(f"{pad}... (circular reference)")
        return
    _visited.add(obj_id)

    # --- dict-like ---
    if isinstance(obj, Mapping):
        print(f"{pad}{type(obj).__name__}  ({len(obj)} keys)")
        keys = list(obj.keys())
        show_keys = keys[:max_items]
        for k in show_keys:
            v = obj[k]
            print(f"{pad}  [{truncate(repr(k), 40)}] : {type_summary(v)}")
            if isinstance(v, (Mapping, list, tuple, set)) or hasattr(v, "__dict__"):
                inspect(v, indent + 2, max_depth, max_items, _visited)
        if len(keys) > max_items:
            print(f"{pad}  ... and {len(keys) - max_items} more keys")

    # --- list / tuple / set ---
    elif isinstance(obj, (list, tuple, set)) and not isinstance(obj, str):
        items = list(obj)
        print(f"{pad}{type(obj).__name__}  ({len(items)} items)")
        show_items = items[:max_items]
        for i, v in enumerate(show_items):
            print(f"{pad}  [{i}] : {type_summary(v)}")
            if isinstance(v, (Mapping, list, tuple, set)) or hasattr(v, "__dict__"):
                inspect(v, indent + 2, max_depth, max_items, _visited)
        if len(items) > max_items:
            print(f"{pad}  ... and {len(items) - max_items} more items")

    # --- numpy array ---
    elif isinstance(obj, np.ndarray):
        print(f"{pad}ndarray  shape={list(obj.shape)}  dtype={obj.dtype}")
        if obj.ndim <= 2 and obj.size <= 20:
            print(f"{pad}  values: {obj}")
        else:
            flat = obj.flatten()
            print(f"{pad}  min={flat.min():.4g}  max={flat.max():.4g}  mean={flat.mean():.4g}")

    # --- objects with __dict__ (class instances) ---
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        print(f"{pad}{type(obj).__name__}  (object)")
        inspect(obj.__dict__, indent + 1, max_depth, max_items, _visited)

    # --- NetworkX graphs ---
    elif type(obj).__name__ in ("Graph", "DiGraph", "MultiGraph", "MultiDiGraph"):
        import networkx as nx
        print(f"{pad}{type(obj).__name__}  "
              f"nodes={obj.number_of_nodes()}  edges={obj.number_of_edges()}")
        sample_nodes = list(obj.nodes)[:3]
        print(f"{pad}  sample nodes: {sample_nodes}")
        sample_edges = list(obj.edges(data=True))[:3]
        print(f"{pad}  sample edges: {sample_edges}")

    # --- primitives ---
    else:
        print(f"{pad}{type_summary(obj)}")


def inspect_graph_nodes(G, scene_id, max_nodes=5):
    """Print all keys and data types for nodes and edges of a NetworkX graph."""
    print(f"\n{'='*60}")
    print(f"  Node/Edge attribute inspection — scene: {scene_id}")
    print(f"{'='*60}")

    nodes = list(G.nodes(data=True))
    edges = list(G.edges(data=True))

    # --- Node attributes ---
    print(f"\nNodes: {len(nodes)} total")
    if nodes:
        all_keys = set()
        for _, attrs in nodes:
            all_keys.update(attrs.keys())

        if all_keys:
            print(f"  Attribute keys: {sorted(all_keys)}")
        else:
            print("  No attributes — position tuple is the node ID itself")

        print(f"\n  Sample nodes (first {min(max_nodes, len(nodes))}):")
        for node, attrs in nodes[:max_nodes]:
            print(f"    node : {node}  type={type(node).__name__}")
            for k, v in attrs.items():
                print(f"           .{k} = {truncate(repr(v))}  type={type(v).__name__}")

    # --- Edge attributes ---
    print(f"\nEdges: {len(edges)} total")
    if edges:
        all_keys = set()
        for _, _, attrs in edges:
            all_keys.update(attrs.keys())

        if all_keys:
            print(f"  Attribute keys: {sorted(all_keys)}")
        else:
            print("  No attributes on edges")

        print(f"\n  Sample edges (first {min(max_nodes, len(edges))}):")
        for u, v, attrs in edges[:max_nodes]:
            print(f"    {truncate(repr(u), 30)} -> {truncate(repr(v), 30)}")
            for k, val in attrs.items():
                print(f"           .{k} = {truncate(repr(val))}  type={type(val).__name__}")


def main():
    parser = argparse.ArgumentParser(description="Inspect the structure of a .pkl file")
    parser.add_argument("file", help="Path to the pickle file")
    parser.add_argument("--depth", type=int, default=5, help="Max recursion depth (default: 5)")
    parser.add_argument("--items", type=int, default=5, help="Max items to show per container (default: 5)")
    parser.add_argument("--scene", type=str, default=None, help="Specific scene ID to inspect nodes for")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  File : {args.file}")
    print(f"{'='*60}\n")

    with open(args.file, "rb") as f:
        data = pickle.load(f)

    print(f"Top-level type : {type_summary(data)}\n")
    inspect(data, max_depth=args.depth, max_items=args.items)

    # Auto-detect dict of graphs and inspect nodes for one scene
    if isinstance(data, Mapping):
        graphs = {k: v for k, v in data.items()
                  if type(v).__name__ in ("Graph", "DiGraph", "MultiGraph", "MultiDiGraph")}
        if graphs:
            if args.scene and args.scene in graphs:
                scene_id = args.scene
            else:
                scene_id = list(graphs.keys())[0]
                if args.scene:
                    print(f"\nWarning: scene '{args.scene}' not found, using '{scene_id}'")

            inspect_graph_nodes(graphs[scene_id], scene_id, max_nodes=args.items)

    print()


if __name__ == "__main__":
    main()