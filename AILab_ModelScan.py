# ComfyUI-QwenVL local model discovery
#
# Every node in this pack picks models that already exist on disk; nothing is
# downloaded automatically. This module centralises base-dir resolution and the
# filesystem scans shared by the Transformers and GGUF backends.
#
# This integration script follows GPL-3.0 License.
#
# Source: https://github.com/1038lab/ComfyUI-QwenVL

import json
from dataclasses import dataclass
from pathlib import Path

import folder_paths

DEFAULT_BASE_DIRS = ["text_encoders", "LLM"]

# Older workflows stored GGUF picks as "[local] sub/dir/model.gguf".
LEGACY_LOCAL_PREFIX = "[local] "

NO_MODELS_PLACEHOLDER = "(no models found — see console)"


def read_base_dirs(config_path: Path, default: list[str] | None = None) -> list[str]:
    """Read "base_dirs" (or the legacy single "base_dir") out of a catalog json."""
    default = list(default if default is not None else DEFAULT_BASE_DIRS)
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
    except FileNotFoundError:
        return default
    except Exception as exc:
        print(f"[QwenVL] {config_path.name} load failed: {exc}")
        return default

    configured = data.get("base_dirs") or data.get("base_dir")
    if isinstance(configured, str):
        configured = [configured]
    if not isinstance(configured, list):
        return default
    values = [str(value).strip() for value in configured if str(value).strip()]
    return values or default


def resolve_base_dirs(base_dir_values) -> list[Path]:
    """Every root the nodes scan, in priority order and without duplicates.

    A value is either an absolute path, a folder_paths key (so extra_model_paths.yaml
    entries are picked up — including ComfyUI's own "text_encoders"), or a plain
    subfolder of ComfyUI/models.
    """
    if isinstance(base_dir_values, str):
        base_dir_values = [base_dir_values]
    roots: list[Path] = []
    for value in base_dir_values or DEFAULT_BASE_DIRS:
        for path in _resolve_one_base_dir(value):
            if path not in roots:
                roots.append(path)
    return roots


def _resolve_one_base_dir(base_dir_value: str) -> list[Path]:
    base_dir = Path(base_dir_value)
    if base_dir.is_absolute():
        return [base_dir]
    # Check extra_model_paths.yaml via folder_paths
    folder_key = base_dir.parts[0] if base_dir.parts else base_dir_value
    sub_path = Path(*base_dir.parts[1:]) if len(base_dir.parts) > 1 else Path()
    if folder_key in folder_paths.folder_names_and_paths:
        paths = folder_paths.get_folder_paths(folder_key)
        if paths:
            return [Path(path) / sub_path for path in paths]
    return [Path(folder_paths.models_dir) / base_dir]


def normalize_model_name(model_name: str) -> str:
    """Drop the legacy "[local] " prefix and normalise separators to posix."""
    name = (model_name or "").strip()
    if name.startswith(LEGACY_LOCAL_PREFIX):
        name = name[len(LEGACY_LOCAL_PREFIX):].strip()
    return name.replace("\\", "/")


def lookup(entries: dict, model_name: str):
    """Find *model_name* in a scan result, tolerating legacy names and bare filenames."""
    name = normalize_model_name(model_name)
    if not name:
        return None
    if name in entries:
        return entries[name]
    tail = name.rsplit("/", 1)[-1]
    for key, value in entries.items():
        if key.rsplit("/", 1)[-1] == tail:
            return value
    return None


def dropdown(names: list[str]) -> list[str]:
    """ComfyUI combos must not be empty; show a hint instead."""
    return names or [NO_MODELS_PLACEHOLDER]


def missing_model_error(model_name: str, base_dirs: list[Path], kind: str) -> FileNotFoundError:
    listed = "\n".join(f"  - {path}" for path in base_dirs) or "  - (no base dir configured)"
    return FileNotFoundError(
        f"[QwenVL] {kind} not found on disk: {model_name}\n"
        f"Automatic downloading is disabled. Place the model under one of these folders and reload ComfyUI:\n"
        f"{listed}\n"
        f"自動ダウンロードは無効です。上記いずれかのフォルダにモデルを配置して ComfyUI を再読み込みしてください。"
    )


def _is_hidden(path: Path, base_dir: Path) -> bool:
    try:
        rel = path.relative_to(base_dir)
    except ValueError:
        return False
    return any(part.startswith(".") for part in rel.parts)


def _rel_key(path: Path, base_dir: Path) -> str:
    try:
        rel = path.relative_to(base_dir)
    except ValueError:
        return path.name
    return rel.as_posix() if rel.parts else path.name


def is_mmproj(filename: str) -> bool:
    return "mmproj" in (filename or "").lower()


def _merge(found: dict, key: str, value, path: Path, what: str):
    """First root wins, mirroring how ComfyUI merges same-named files across roots."""
    if key in found:
        print(f"[QwenVL] Ignoring duplicate {what} name '{key}' at {path}")
        return
    found[key] = value


def scan_gguf_files(base_dirs: list[Path]) -> dict[str, Path]:
    """Model .gguf files under *base_dirs*, keyed by their path relative to their root."""
    return _scan_gguf(base_dirs, want_mmproj=False)


def scan_mmproj_files(base_dirs: list[Path]) -> dict[str, Path]:
    """Vision projector .gguf files under *base_dirs*, keyed the same way."""
    return _scan_gguf(base_dirs, want_mmproj=True)


def _scan_gguf(base_dirs: list[Path], want_mmproj: bool) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for base_dir in base_dirs:
        if not base_dir.is_dir():
            continue
        for path in base_dir.rglob("*.gguf"):
            if not path.is_file() or _is_hidden(path, base_dir):
                continue
            if is_mmproj(path.name) != want_mmproj:
                continue
            _merge(found, _rel_key(path, base_dir), path, path, "gguf")
    return dict(sorted(found.items()))


_VISION_CONFIG_KEYS = ("vision_config", "visual_config", "vision_tower_config")
_VISION_NAME_HINTS = ("_vl", "-vl", "vision", "imagetext", "image_text", "llava", "multimodal")


@dataclass(frozen=True)
class HFLocalModel:
    name: str
    path: Path
    is_vision: bool
    is_prequantized: bool
    weight_bytes: int

    @property
    def weight_gib(self) -> float:
        return self.weight_bytes / 1024**3


def scan_hf_models(base_dirs: list[Path]) -> dict[str, HFLocalModel]:
    """Transformers checkpoints under *base_dirs*, keyed by their relative path.

    A directory counts as a checkpoint when it holds a config.json next to at
    least one weight shard.
    """
    found: dict[str, HFLocalModel] = {}
    for base_dir in base_dirs:
        if not base_dir.is_dir():
            continue
        for config_path in base_dir.rglob("config.json"):
            if not config_path.is_file() or _is_hidden(config_path, base_dir):
                continue
            model_dir = config_path.parent
            weights = sorted(model_dir.glob("*.safetensors")) or sorted(model_dir.glob("*.bin"))
            if not weights:
                continue
            name = _rel_key(model_dir, base_dir)
            _merge(found, name, _describe_hf_model(name, model_dir, config_path, weights),
                   model_dir, "checkpoint")
    return dict(sorted(found.items()))


def _describe_hf_model(name: str, model_dir: Path, config_path: Path, weights: list[Path]) -> HFLocalModel:
    config: dict = {}
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh) or {}
    except Exception as exc:
        print(f"[QwenVL] Could not read {config_path}: {exc}")

    text_config: dict = config.get("text_config") or {}
    haystack = " ".join(
        [name, str(config.get("model_type") or "")]
        + [str(arch) for arch in (config.get("architectures") or [])]
    ).lower()
    is_vision = any(key in config for key in _VISION_CONFIG_KEYS) or any(
        hint in haystack for hint in _VISION_NAME_HINTS
    )

    is_prequantized = bool(config.get("quantization_config") or text_config.get("quantization_config"))
    if not is_prequantized:
        is_prequantized = any(tag in name.lower() for tag in ("-fp8", "_fp8"))

    weight_bytes = 0
    for weight in weights:
        try:
            weight_bytes += weight.stat().st_size
        except OSError:
            pass

    return HFLocalModel(
        name=name,
        path=model_dir,
        is_vision=is_vision,
        is_prequantized=is_prequantized,
        weight_bytes=weight_bytes,
    )
