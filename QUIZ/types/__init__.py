import importlib
import os
import pkgutil
from typing import Any, Dict, List

TYPE_REGISTRY: Dict[str, Dict[str, Any]] = {}


def load_all_quiz_types() -> Dict[str, Dict[str, Any]]:
    if TYPE_REGISTRY:
        return TYPE_REGISTRY

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    for _, mod_name, is_pkg in pkgutil.iter_modules([pkg_dir]):
        if is_pkg or mod_name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{mod_name}")
        spec = getattr(module, "TYPE_SPEC", None)
        if isinstance(spec, dict) and spec.get("name"):
            TYPE_REGISTRY[spec["name"]] = spec
    return TYPE_REGISTRY


def get_quiz_type_names() -> List[str]:
    return list(load_all_quiz_types().keys())


def get_quiz_type_spec(type_name: str) -> Dict[str, Any]:
    reg = load_all_quiz_types()
    return reg.get(type_name) or next(iter(reg.values()))