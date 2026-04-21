from __future__ import annotations

import os
import shutil
from pathlib import Path


SKILLS_ENV_VAR = "EVOSKILL_SKILLS_DIR"


def runtime_skills_dir(project_root: str | Path) -> Path:
    return Path(project_root) / ".claude" / "skills"


def configured_skills_dir(project_root: str | Path, configured: str | Path | None) -> Path:
    if configured is None:
        raw = os.getenv(SKILLS_ENV_VAR, "").strip()
        if raw:
            configured = raw
        else:
            return runtime_skills_dir(project_root)

    p = Path(configured)
    if not p.is_absolute():
        p = Path(project_root) / p
    return p


def _replace_tree(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    shutil.copytree(src, dst)


def activate_skills_profile(project_root: str | Path, configured: str | Path | None) -> Path:
    """Activate task/profile-specific skills by syncing profile -> runtime dir.

    Rules:
    - Runtime path is always <project_root>/.claude/skills (used by tool runtimes)
    - Profile path is configurable and becomes source of truth.
    - If profile path doesn't exist, it's initialized from runtime contents.
    """
    project_root = Path(project_root)
    runtime = runtime_skills_dir(project_root)
    profile = configured_skills_dir(project_root, configured)

    runtime.parent.mkdir(parents=True, exist_ok=True)
    if not runtime.exists():
        runtime.mkdir(parents=True, exist_ok=True)

    if profile.resolve() == runtime.resolve():
        os.environ[SKILLS_ENV_VAR] = str(profile)
        return profile

    profile.parent.mkdir(parents=True, exist_ok=True)
    if not profile.exists():
        _replace_tree(runtime, profile)

    _replace_tree(profile, runtime)
    os.environ[SKILLS_ENV_VAR] = str(profile)
    return profile


def persist_runtime_to_profile(project_root: str | Path, configured: str | Path | None) -> Path:
    """Persist current runtime skills back to configured profile directory."""
    project_root = Path(project_root)
    runtime = runtime_skills_dir(project_root)
    profile = configured_skills_dir(project_root, configured)

    if profile.resolve() == runtime.resolve():
        os.environ[SKILLS_ENV_VAR] = str(profile)
        return profile

    if not runtime.exists():
        raise FileNotFoundError(f"Runtime skills directory not found: {runtime}")

    profile.parent.mkdir(parents=True, exist_ok=True)
    _replace_tree(runtime, profile)
    os.environ[SKILLS_ENV_VAR] = str(profile)
    return profile
