"""Habitat Env that can load navmeshes from a separate root from scene meshes.

Episodes still point at .glb paths under DATASET.SCENES_DIR. When the task YAML
sets **DATASET.NAVMESH_DIR** (non-empty), after each scene load we reload the
navmesh from ``<NAVMESH_DIR>/<scene_stem>.navmesh`` when that file exists (scene
stem is the basename of the episode ``scene_id`` without extension). If
**NAVMESH_DIR** is empty, behavior matches stock ``habitat.Env``.
"""
from __future__ import annotations

import os
import warnings
from typing import Optional

from habitat.config import Config
from habitat.core.env import Env


def _resolve_split_navmesh_file(navmesh_dir: str, scene_path: str) -> Optional[str]:
    if not (navmesh_dir and str(navmesh_dir).strip()):
        return None
    base = os.path.basename(scene_path)
    stem, _ext = os.path.splitext(base)
    candidate = os.path.join(navmesh_dir, f"{stem}.navmesh")
    if os.path.isfile(candidate):
        return candidate
    return None


def _load_navmesh(pathfinder, filepath: str) -> None:
    for name in ("load_nav_mesh", "load_navmesh"):
        fn = getattr(pathfinder, name, None)
        if callable(fn):
            fn(filepath)
            return
    raise RuntimeError(
        "Cannot load navmesh: pathfinder has neither load_nav_mesh nor load_navmesh "
        "(habitat-sim / habitat-lab version mismatch)."
    )


class SplitNavmeshEnv(Env):
    """Same as habitat.Env; if TASK_CONFIG.DATASET.NAVMESH_DIR is set, reloads navmesh from there."""

    def reconfigure(self, config: Config) -> None:
        super().reconfigure(config)
        navmesh_dir = getattr(self._config.DATASET, "NAVMESH_DIR", "")
        if not (navmesh_dir and str(navmesh_dir).strip()):
            return
        scene_path = self.current_episode.scene_id
        path = _resolve_split_navmesh_file(str(navmesh_dir), scene_path)
        if path is None:
            stem = os.path.splitext(os.path.basename(scene_path))[0]
            warnings.warn(
                f"NAVMESH_DIR={navmesh_dir!r} set but no file "
                f"{stem!r}.navmesh found there for scene {scene_path!r}; "
                "using simulator default navmesh.",
                UserWarning,
                stacklevel=1,
            )
            return
        _load_navmesh(self._sim.pathfinder, path)
