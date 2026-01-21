"""YAML heuristic file discovery and loading.

This module handles discovering and loading YAML heuristic files from
both the builtin directory and custom user directories.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

from post_training_toolkit.heuristics.schema import YAMLHeuristic


# Path to builtin heuristics directory
BUILTIN_DIR = Path(__file__).parent / "builtin"


class HeuristicLoader:
    """Loads YAML heuristics from builtin and custom directories.

    The loader searches for YAML files in:
    1. The builtin directory (post_training_toolkit/heuristics/builtin/)
    2. Any custom directories specified by the user

    Directory structure:
        builtin/
        ├── common/           # Heuristics for all trainers
        │   └── reward_variance.yaml
        ├── dpo/              # DPO-specific heuristics
        │   ├── margin_collapse.yaml
        │   └── loss_random.yaml
        ├── ppo/              # PPO-specific heuristics
        │   └── entropy_collapse.yaml
        └── ...

    Usage:
        >>> loader = HeuristicLoader()
        >>> heuristics = loader.load_for_trainer("dpo")
        >>> for h in heuristics:
        ...     print(h.name, h.severity)
    """

    def __init__(
        self,
        custom_dirs: Optional[List[Path]] = None,
        include_builtin: bool = True,
    ):
        """Initialize the heuristic loader.

        Args:
            custom_dirs: Optional list of custom directories to search
            include_builtin: Whether to include builtin heuristics (default: True)
        """
        if yaml is None:
            raise ImportError(
                "PyYAML is required for YAML heuristics. "
                "Install it with: pip install pyyaml"
            )

        self.custom_dirs = custom_dirs or []
        self.include_builtin = include_builtin
        self._cache: Dict[str, List[YAMLHeuristic]] = {}

    def _get_search_dirs(self) -> List[Path]:
        """Get all directories to search for heuristics."""
        dirs = []
        if self.include_builtin and BUILTIN_DIR.exists():
            dirs.append(BUILTIN_DIR)
        for custom_dir in self.custom_dirs:
            path = Path(custom_dir)
            if path.exists():
                dirs.append(path)
        return dirs

    def _load_yaml_file(self, path: Path) -> Optional[YAMLHeuristic]:
        """Load a single YAML heuristic file.

        Args:
            path: Path to the YAML file

        Returns:
            YAMLHeuristic if valid, None if invalid or disabled
        """
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)

            if data is None:
                return None

            heuristic = YAMLHeuristic.from_dict(data)

            if not heuristic.enabled:
                return None

            return heuristic

        except (yaml.YAMLError, ValueError) as e:
            # Log warning but don't fail
            import warnings
            warnings.warn(f"Failed to load heuristic from {path}: {e}")
            return None

    def _discover_yaml_files(self, base_dir: Path, trainer_type: Optional[str] = None) -> List[Path]:
        """Discover YAML files in a directory structure.

        Args:
            base_dir: Base directory to search
            trainer_type: Optional trainer type to filter by

        Returns:
            List of paths to YAML files
        """
        yaml_files = []

        # Always include common heuristics
        common_dir = base_dir / "common"
        if common_dir.exists():
            yaml_files.extend(common_dir.glob("*.yaml"))
            yaml_files.extend(common_dir.glob("*.yml"))

        # Include trainer-specific heuristics if specified
        if trainer_type:
            trainer_dir = base_dir / trainer_type.lower()
            if trainer_dir.exists():
                yaml_files.extend(trainer_dir.glob("*.yaml"))
                yaml_files.extend(trainer_dir.glob("*.yml"))

        # Also check for YAML files directly in base_dir (flat structure support)
        yaml_files.extend(base_dir.glob("*.yaml"))
        yaml_files.extend(base_dir.glob("*.yml"))

        return list(set(yaml_files))  # Deduplicate

    def load_all(self) -> List[YAMLHeuristic]:
        """Load all heuristics from all directories.

        Returns:
            List of all valid YAMLHeuristic objects
        """
        cache_key = "_all_"
        if cache_key in self._cache:
            return self._cache[cache_key]

        heuristics = []
        seen_names: Set[str] = set()

        for search_dir in self._get_search_dirs():
            # Get all trainer subdirectories
            if search_dir.is_dir():
                for subdir in search_dir.iterdir():
                    if subdir.is_dir():
                        for yaml_file in subdir.glob("*.yaml"):
                            h = self._load_yaml_file(yaml_file)
                            if h and h.name not in seen_names:
                                heuristics.append(h)
                                seen_names.add(h.name)
                        for yaml_file in subdir.glob("*.yml"):
                            h = self._load_yaml_file(yaml_file)
                            if h and h.name not in seen_names:
                                heuristics.append(h)
                                seen_names.add(h.name)

                # Also check root-level YAML files
                for yaml_file in search_dir.glob("*.yaml"):
                    h = self._load_yaml_file(yaml_file)
                    if h and h.name not in seen_names:
                        heuristics.append(h)
                        seen_names.add(h.name)
                for yaml_file in search_dir.glob("*.yml"):
                    h = self._load_yaml_file(yaml_file)
                    if h and h.name not in seen_names:
                        heuristics.append(h)
                        seen_names.add(h.name)

        self._cache[cache_key] = heuristics
        return heuristics

    def load_for_trainer(self, trainer_type: str) -> List[YAMLHeuristic]:
        """Load heuristics applicable to a specific trainer type.

        Args:
            trainer_type: The trainer type (e.g., "dpo", "ppo", "sft")

        Returns:
            List of YAMLHeuristic objects that apply to this trainer
        """
        cache_key = trainer_type.lower()
        if cache_key in self._cache:
            return self._cache[cache_key]

        heuristics = []
        seen_names: Set[str] = set()

        for search_dir in self._get_search_dirs():
            yaml_files = self._discover_yaml_files(search_dir, trainer_type)

            for yaml_file in yaml_files:
                h = self._load_yaml_file(yaml_file)
                if h and h.name not in seen_names and h.applies_to_trainer(trainer_type):
                    heuristics.append(h)
                    seen_names.add(h.name)

        self._cache[cache_key] = heuristics
        return heuristics

    def get_heuristic(self, name: str) -> Optional[YAMLHeuristic]:
        """Get a specific heuristic by name.

        Args:
            name: The heuristic name

        Returns:
            YAMLHeuristic if found, None otherwise
        """
        all_heuristics = self.load_all()
        for h in all_heuristics:
            if h.name == name:
                return h
        return None

    def clear_cache(self):
        """Clear the internal cache."""
        self._cache.clear()


# Module-level loader instance for convenience
_default_loader: Optional[HeuristicLoader] = None


def get_loader(
    custom_dirs: Optional[List[Path]] = None,
    include_builtin: bool = True,
) -> HeuristicLoader:
    """Get or create a HeuristicLoader instance.

    Args:
        custom_dirs: Optional list of custom directories
        include_builtin: Whether to include builtin heuristics

    Returns:
        HeuristicLoader instance
    """
    global _default_loader

    # Create new loader if custom_dirs specified or no default exists
    if custom_dirs is not None or _default_loader is None:
        _default_loader = HeuristicLoader(
            custom_dirs=custom_dirs,
            include_builtin=include_builtin,
        )

    return _default_loader


def load_heuristics_for_trainer(
    trainer_type: str,
    custom_dirs: Optional[List[Path]] = None,
) -> List[YAMLHeuristic]:
    """Convenience function to load heuristics for a trainer type.

    Args:
        trainer_type: The trainer type (e.g., "dpo", "ppo")
        custom_dirs: Optional list of custom directories

    Returns:
        List of applicable YAMLHeuristic objects
    """
    loader = get_loader(custom_dirs=custom_dirs)
    return loader.load_for_trainer(trainer_type)
