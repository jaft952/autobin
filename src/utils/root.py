"""
Utilities for locating the repository root reliably across environments.
"""

import os
from pathlib import Path
from typing import Optional, Sequence, Union


PathLike = Union[str, Path]


def _normalize_start_path(start_path: Optional[PathLike]) -> Path:
	"""Return an absolute directory path to start searching from."""
	if start_path is None:
		return Path.cwd().resolve()

	p = Path(start_path).expanduser().resolve()
	return p if p.is_dir() else p.parent


def _has_markers(candidate: Path, markers: Sequence[str]) -> bool:
	"""Check whether a path contains all required root markers."""
	for marker in markers:
		marker_path = candidate / marker
		if marker.endswith("/"):
			if not marker_path.is_dir():
				return False
		else:
			if not marker_path.exists():
				return False
	return True


def find_repo_root(
	start_path: Optional[PathLike] = None,
	env_var: str = "AUTOBIN_ROOT",
	markers: Sequence[str] = ("README.md", "src/"),
) -> Path:
	"""
	Find the project repository root by walking up from a starting path.

	Resolution order:
	1) Environment override via ``env_var`` (default: AUTOBIN_ROOT)
	2) Upward search from ``start_path`` (or current working directory)

	Args:
		start_path: File or directory path to start from.
		env_var: Environment variable for explicit root override.
		markers: Required files/directories that identify repo root.

	Returns:
		Path to the detected repository root.

	Raises:
		FileNotFoundError: If no matching root directory is found.
	"""
	env_root = os.getenv(env_var)
	if env_root:
		env_path = Path(env_root).expanduser().resolve()
		if _has_markers(env_path, markers):
			return env_path
		raise FileNotFoundError(
			f"{env_var} is set to '{env_path}', but required markers {tuple(markers)} were not found there."
		)

	start = _normalize_start_path(start_path)
	for candidate in (start, *start.parents):
		if _has_markers(candidate, markers):
			return candidate

	raise FileNotFoundError(
		f"Could not locate repository root from '{start}'. Expected markers: {tuple(markers)}. "
		f"You can also set {env_var} to the repository root path."
	)


def repo_path(*parts: str, start_path: Optional[PathLike] = None) -> Path:
	"""Build an absolute path under the detected repository root."""
	return find_repo_root(start_path=start_path).joinpath(*parts)

