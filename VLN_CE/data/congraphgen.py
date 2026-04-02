import habitat_sim
import numpy as np
import networkx as nx
import pickle
from itertools import product
import os
import uuid
import hashlib

def point_to_uuid(pt):
    """Generate a deterministic UUID from an XYZ point."""
    key = f"{pt[0]:.6f},{pt[1]:.6f},{pt[2]:.6f}"
    return hashlib.md5(key.encode()).hexdigest()

def make_sim_config(scene_path: str):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.load_semantic_mesh = False   # suppress semantic annotation warning
    sim_cfg.create_renderer = False      # headless — no GPU/geometry needed

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = []

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def sample_navigable_points(sim, grid_spacing=0.5, y_tolerance=0.5):
    """
    Sample positions by laying a grid over the scene's AABB
    and keeping only navigable points.
    """
    navmesh_bounds = sim.pathfinder.get_bounds()
    lower, upper = navmesh_bounds[0], navmesh_bounds[1]

    xs = np.arange(lower[0], upper[0], grid_spacing)
    zs = np.arange(lower[2], upper[2], grid_spacing)

    navigable_points = []
    for x, z in product(xs, zs):
        # snap to navmesh — returns closest valid y
        pt = sim.pathfinder.snap_point(np.array([x, lower[1], z]))
        if sim.pathfinder.is_navigable(pt, max_y_delta=y_tolerance):
            navigable_points.append(tuple(np.round(pt, 4)))

    # Deduplicate (snapping can produce duplicates)
    return list(set(navigable_points))


def build_graph(sim, navigable_points, max_edge_dist=1.5):
    G = nx.Graph()

    # Add nodes with UUID as ID and position as attribute
    point_to_id = {}
    for pt in navigable_points:
        node_id = point_to_uuid(pt)
        point_to_id[pt] = node_id
        G.add_node(node_id, position=np.array(pt))

    # Use KDTree for efficient neighbor search
    from scipy.spatial import KDTree
    pts = np.array(navigable_points)
    tree = KDTree(pts)
    candidate_pairs = tree.query_pairs(r=max_edge_dist)

    for i, j in candidate_pairs:
        p1, p2 = navigable_points[i], navigable_points[j]

        path = habitat_sim.ShortestPath()
        path.requested_start = np.array(p1)
        path.requested_end = np.array(p2)
        found = sim.pathfinder.find_path(path)

        if found and path.geodesic_distance < max_edge_dist * 1.5:
            n1, n2 = point_to_id[p1], point_to_id[p2]
            G.add_edge(n1, n2, weight=path.geodesic_distance)

    return G


def generate_connectivity_graphs(
    scene_paths: dict,
    output_path: str,
    grid_spacing: float = 1.0,
    max_edge_dist: float = 1.5,
):
    all_graphs = {}

    for scene_id, scene_path in scene_paths.items():
        print(f"Processing {scene_id}...")

        cfg = make_sim_config(scene_path)
        sim = habitat_sim.Simulator(cfg)

        # Only recompute navmesh if one wasn't loaded from disk
        if not sim.pathfinder.is_loaded:
            print(f"  No navmesh found for {scene_id}, skipping.")
            sim.close()
            continue

        points = sample_navigable_points(sim, grid_spacing=grid_spacing)
        print(f"  Found {len(points)} navigable points")

        graph = build_graph(sim, points, max_edge_dist=max_edge_dist)
        print(f"  Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

        all_graphs[scene_id] = graph
        sim.close()

    with open(output_path, "wb") as f:
        pickle.dump(all_graphs, f)

    print(f"Saved to {output_path}")
    return all_graphs

# --- Usage ---
gibson_dir = "./gibson"

scenes = {
    os.path.splitext(f)[0]: os.path.join(gibson_dir, f)
    for f in os.listdir(gibson_dir)
    if f.endswith(".glb")
}

generate_connectivity_graphs(scenes, "connectivity_graphs.pkl")

