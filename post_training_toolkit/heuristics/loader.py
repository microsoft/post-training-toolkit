
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

from post_training_toolkit.heuristics.schema import YAMLHeuristic

BUILTIN_DIR = Path(__file__).parent / "builtin"

class HeuristicLoader:
    def __init__(
        self,
        custom_dirs: Optional[List[Path]] = None,
        include_builtin: bool = True,
    ):
        if yaml is None:
            raise ImportError(
                "PyYAML is required for YAML heuristics. "
                "Install it with: pip install pyyaml"
            )

        self.custom_dirs = custom_dirs or []
        self.include_builtin = include_builtin
        self._cache: Dict[str, List[YAMLHeuristic]] = {}

    def _get_search_dirs(self) -> List[Path]:
        dirs = []
        if self.include_builtin and BUILTIN_DIR.exists():
            dirs.append(BUILTIN_DIR)
        for custom_dir in self.custom_dirs:
            path = Path(custom_dir)
            if path.exists():
                dirs.append(path)
        return dirs

    def _load_yaml_file(self, path: Path) -> Optional[YAMLHeuristic]:
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
            import warnings
            warnings.warn(f"Failed to load heuristic from {path}: {e}")
            return None

    def _discover_yaml_files(self, base_dir: Path, trainer_type: Optional[str] = None) -> List[Path]:
        yaml_files = []

        common_dir = base_dir / "common"
        if common_dir.exists():
            yaml_files.extend(common_dir.glob("*.yaml"))
            yaml_files.extend(common_dir.glob("*.yml"))

        if trainer_type:
            trainer_dir = base_dir / trainer_type.lower()
            if trainer_dir.exists():
                yaml_files.extend(trainer_dir.glob("*.yaml"))
                yaml_files.extend(trainer_dir.glob("*.yml"))

        yaml_files.extend(base_dir.glob("*.yaml"))
        yaml_files.extend(base_dir.glob("*.yml"))

        return list(set(yaml_files))

    def load_all(self) -> List[YAMLHeuristic]:
        cache_key = "_all_"
        if cache_key in self._cache:
            return self._cache[cache_key]

        heuristics = []
        seen_names: Set[str] = set()

        for search_dir in self._get_search_dirs():
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
        all_heuristics = self.load_all()
        for h in all_heuristics:
            if h.name == name:
                return h
        return None

    def clear_cache(self):
        self._cache.clear()

_default_loader: Optional[HeuristicLoader] = None

def get_loader(
    custom_dirs: Optional[List[Path]] = None,
    include_builtin: bool = True,
) -> HeuristicLoader:
    global _default_loader

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
    loader = get_loader(custom_dirs=custom_dirs)
    return loader.load_for_trainer(trainer_type)
