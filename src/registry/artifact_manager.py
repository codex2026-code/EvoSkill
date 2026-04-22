"""Filesystem-backed program manager for artifact-based iteration tracking.

This manager mirrors ProgramManager's public API used by the loop, but avoids
Git branch/tag operations. Program states are versioned as snapshots under:

    <artifacts_root>/<question_id>/programs/<program_name>/

Each program stores:
- program.yaml (ProgramConfig)
- state/        (managed workspace snapshot)
- commits.jsonl (commit-like log records)

Frontier membership and the program index are tracked in JSON files under
``artifacts_root``.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from .models import ProgramConfig


class ArtifactProgramManager:
    """Program manager that persists iteration states under artifacts/ directories."""

    PROGRAM_FILE = ".claude/program.yaml"

    def __init__(self, artifacts_root: str | Path, cwd: str | Path | None = None):
        self.cwd = Path(cwd) if cwd else Path.cwd()
        self.artifacts_root = Path(artifacts_root)
        self.artifacts_root.mkdir(parents=True, exist_ok=True)

        self._index_file = self.artifacts_root / "program_index.json"
        self._frontier_file = self.artifacts_root / "frontier.json"
        self._meta_file = self.artifacts_root / "meta.json"
        self._current_file = self.artifacts_root / "current_program.txt"

        self._active_question_id = "global"
        self._managed_paths = [
            Path(".claude/program.yaml"),
            Path(".claude/skills"),
            Path("src/agent_profiles/base_agent/prompt.txt"),
        ]

        self._ensure_state_files()

    # ----- public helpers -----
    def set_iteration_context(self, question_id: str) -> None:
        """Set logical question id used for organizing new program snapshots."""
        cleaned = (question_id or "").strip() or "global"
        self._active_question_id = cleaned

    # ----- ProgramManager-compatible API -----
    def create_program(self, name: str, config: ProgramConfig, parent: str | None = None) -> str:
        if parent:
            self.switch_to(parent)

        program_dir = self._program_dir(name, self._active_question_id)
        if program_dir.exists():
            shutil.rmtree(program_dir)
        program_dir.mkdir(parents=True, exist_ok=True)

        # Seed state from parent snapshot if available, else from current workspace.
        parent_dir = self._get_program_dir(parent) if parent else None
        state_dir = program_dir / "state"
        if parent_dir and (parent_dir / "state").exists():
            shutil.copytree(parent_dir / "state", state_dir)
        else:
            state_dir.mkdir(parents=True, exist_ok=True)
            self._capture_workspace_to(state_dir)

        # Make config active in workspace and persist in this program directory.
        self._write_config_to_workspace(config)
        self._write_config_to_program(program_dir, config)
        self._capture_workspace_to(state_dir)

        self._set_current(name)
        self._index_set(name, program_dir)
        self._append_commit(name, f"Create program: {name}")
        return name

    def switch_to(self, name: str) -> None:
        program_dir = self._get_program_dir(name)
        if program_dir is None:
            raise ValueError(f"Program not found: {name}")
        self._restore_workspace_from(program_dir / "state")
        self._set_current(name)

    def get_current(self) -> ProgramConfig:
        config_path = self.cwd / self.PROGRAM_FILE
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return ProgramConfig.model_validate(data)

    def get_current_name(self) -> str:
        if self._current_file.exists():
            return self._current_file.read_text(encoding="utf-8").strip() or ""
        return ""

    def list_programs(self) -> list[str]:
        return sorted(self._load_index().keys())

    def get_lineage(self, name: str) -> list[str]:
        lineage = [name]
        current = name
        while True:
            cfg = self._read_config_from_program(current)
            if cfg.parent is None:
                break
            parent = cfg.parent.replace("program/", "")
            lineage.append(parent)
            current = parent
        return lineage

    def get_children(self, name: str) -> list[str]:
        parent_ref = f"program/{name}"
        children: list[str] = []
        for program in self.list_programs():
            if program == name:
                continue
            cfg = self._read_config_from_program(program)
            if cfg.parent == parent_ref:
                children.append(program)
        return children

    def discard(self, name: str) -> None:
        frontier = self.get_frontier()
        if name in frontier:
            self.unmark_frontier(name)

        program_dir = self._get_program_dir(name)
        if program_dir and program_dir.exists():
            shutil.rmtree(program_dir)

        index = self._load_index()
        index.pop(name, None)
        self._save_index(index)

    def mark_frontier(self, name: str) -> None:
        frontier = self.get_frontier()
        if name not in frontier:
            frontier.append(name)
            self._save_frontier(frontier)

    def unmark_frontier(self, name: str) -> None:
        frontier = [n for n in self.get_frontier() if n != name]
        self._save_frontier(frontier)

    def get_frontier(self) -> list[str]:
        if not self._frontier_file.exists():
            return []
        data = json.loads(self._frontier_file.read_text(encoding="utf-8"))
        return list(data.get("frontier", []))

    def get_frontier_with_scores(self) -> list[tuple[str, float]]:
        scored: list[tuple[str, float]] = []
        for name in self.get_frontier():
            try:
                cfg = self._read_config_from_program(name)
                score = cfg.get_score()
                if score is not None:
                    scored.append((name, score))
            except Exception:
                continue
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def select_from_frontier(self, strategy: str, iteration: int = 0) -> str | None:
        scored = self.get_frontier_with_scores()
        if not scored:
            return None
        if strategy == "random":
            return random.choice(scored)[0]
        if strategy == "round_robin":
            return scored[iteration % len(scored)][0]
        return scored[0][0]

    def get_best_from_frontier(self) -> str | None:
        scored = self.get_frontier_with_scores()
        return scored[0][0] if scored else None

    def update_frontier(self, name: str, score: float, max_size: int = 5) -> bool:
        cfg = self._read_config_from_program(name)
        updated = cfg.with_score(score)
        self._write_config_to_program(self._require_program_dir(name), updated)

        if self.get_current_name() == name:
            self._write_config_to_workspace(updated)

        self._append_commit(name, f"Update score: {score:.4f}")

        scored = self.get_frontier_with_scores()
        if len(scored) < max_size:
            self.mark_frontier(name)
            return True

        worst_name, worst_score = scored[-1]
        if score > worst_score:
            self.unmark_frontier(worst_name)
            self.mark_frontier(name)
            return True
        return False

    def commit(self, message: str | None = None) -> bool:
        current = self.get_current_name()
        if not current:
            return False

        program_dir = self._require_program_dir(current)
        state_dir = program_dir / "state"
        before = self._hash_tree(state_dir)
        self._capture_workspace_to(state_dir)
        after = self._hash_tree(state_dir)

        if before == after:
            return False

        default_msg = f"Update program: {current}"
        self._append_commit(current, message or default_msg)
        return True

    # ----- internals -----
    def _ensure_state_files(self) -> None:
        if not self._index_file.exists():
            self._save_index({})
        if not self._frontier_file.exists():
            self._save_frontier([])
        if not self._meta_file.exists():
            self._meta_file.write_text(
                json.dumps({"created_at": datetime.utcnow().isoformat()}, indent=2),
                encoding="utf-8",
            )

    def _program_dir(self, name: str, question_id: str) -> Path:
        return self.artifacts_root / question_id / "programs" / name

    def _index_set(self, name: str, program_dir: Path) -> None:
        index = self._load_index()
        index[name] = str(program_dir.relative_to(self.artifacts_root))
        self._save_index(index)

    def _load_index(self) -> dict[str, str]:
        if not self._index_file.exists():
            return {}
        return json.loads(self._index_file.read_text(encoding="utf-8"))

    def _save_index(self, data: dict[str, str]) -> None:
        self._index_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _save_frontier(self, frontier: list[str]) -> None:
        self._frontier_file.write_text(
            json.dumps({"frontier": frontier}, indent=2), encoding="utf-8"
        )

    def _set_current(self, name: str) -> None:
        self._current_file.write_text(name, encoding="utf-8")

    def _get_program_dir(self, name: str | None) -> Path | None:
        if not name:
            return None
        rel = self._load_index().get(name)
        if not rel:
            return None
        return self.artifacts_root / rel

    def _require_program_dir(self, name: str) -> Path:
        program_dir = self._get_program_dir(name)
        if program_dir is None:
            raise ValueError(f"Program not found: {name}")
        return program_dir

    def _read_config_from_program(self, name: str) -> ProgramConfig:
        program_dir = self._require_program_dir(name)
        cfg_path = program_dir / "program.yaml"
        with open(cfg_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return ProgramConfig.model_validate(data)

    def _write_config_to_program(self, program_dir: Path, config: ProgramConfig) -> None:
        with open(program_dir / "program.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)

    def _write_config_to_workspace(self, config: ProgramConfig) -> None:
        config_path = self.cwd / self.PROGRAM_FILE
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)

    def _append_commit(self, name: str, message: str) -> None:
        program_dir = self._require_program_dir(name)
        log_path = program_dir / "commits.jsonl"
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "message": message,
            "question_id": self._active_question_id,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _capture_workspace_to(self, state_dir: Path) -> None:
        if state_dir.exists():
            shutil.rmtree(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)

        for rel in self._managed_paths:
            src = self.cwd / rel
            dst = state_dir / rel
            if not src.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    def _restore_workspace_from(self, state_dir: Path) -> None:
        if not state_dir.exists():
            return

        for rel in self._managed_paths:
            target = self.cwd / rel
            snap = state_dir / rel

            if target.exists() or target.is_symlink():
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink()

            if not snap.exists():
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            if snap.is_dir():
                shutil.copytree(snap, target)
            else:
                shutil.copy2(snap, target)

    @staticmethod
    def _hash_tree(root: Path) -> str:
        if not root.exists():
            return ""
        h = hashlib.sha256()
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            rel = path.relative_to(root)
            h.update(str(rel).encode("utf-8"))
            h.update(path.read_bytes())
        return h.hexdigest()
