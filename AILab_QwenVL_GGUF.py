# ComfyUI-QwenVL (GGUF)
# GGUF nodes powered by llama.cpp for Qwen-VL models, including Qwen3-VL and Qwen2.5-VL.
# Provides vision-capable GGUF inference and prompt execution.
#
# Models are loaded via llama-cpp-python from .gguf files already present under
# the configured base dirs (models/text_encoders and models/LLM by default);
# nothing is downloaded automatically.
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/1038lab/ComfyUI-QwenVL

import base64
import gc
import io
import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from AILab_OutputCleaner import OutputCleanConfig, clean_model_output
import AILab_ModelScan as model_scan

NODE_DIR = Path(__file__).parent
CONFIG_PATH = NODE_DIR / "hf_models.json"
SYSTEM_PROMPTS_PATH = NODE_DIR / "AILab_System_Prompts.json"
GGUF_CONFIG_PATH = NODE_DIR / "gguf_models.json"


def _load_prompt_config():
    preset_prompts = ["🖼️ Detailed Description"]
    system_prompts: dict[str, str] = {}

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        preset_prompts = data.get("_preset_prompts") or preset_prompts
        system_prompts = data.get("_system_prompts") or system_prompts
    except Exception as exc:
        print(f"[QwenVL] Config load failed: {exc}")

    try:
        with open(SYSTEM_PROMPTS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        qwenvl_prompts = data.get("qwenvl") or {}
        preset_override = data.get("_preset_prompts") or []
        if isinstance(qwenvl_prompts, dict) and qwenvl_prompts:
            system_prompts = qwenvl_prompts
        if isinstance(preset_override, list) and preset_override:
            preset_prompts = preset_override
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"[QwenVL] System prompts load failed: {exc}")

    return preset_prompts, system_prompts


PRESET_PROMPTS, SYSTEM_PROMPTS = _load_prompt_config()


MMPROJ_AUTO = "auto"

TOOLTIPS = {
    "model_name": "Pick a .gguf already present under models/text_encoders or models/LLM. Nothing is downloaded automatically — copy the file in yourself, then reload ComfyUI.",
    "mmproj_name": "Vision projector to pair with the model. auto picks the first *mmproj*.gguf sitting next to it.",
}


@dataclass(frozen=True)
class GGUFVLResolved:
    display_name: str
    model_path: Path
    mmproj_path: Path | None
    context_length: int = 8192
    image_max_tokens: int = 8192
    image_min_tokens: int = 1024
    n_batch: int = 8192
    gpu_layers: int = -1
    top_k: int = 0
    pool_size: int = 4194304


GGUF_BASE_DIRS: list[str] = model_scan.read_base_dirs(GGUF_CONFIG_PATH)
LOCAL_GGUF_MODELS: dict[str, Path] = {}
LOCAL_MMPROJ_FILES: dict[str, Path] = {}


def _is_gemma_model_name(name: str) -> bool:
    """Detect Gemma models by filename substring (covers relative paths)."""
    return "gemma" in (name or "").lower()


def refresh_local_gguf():
    """Re-scan the base dirs. Called from INPUT_TYPES so new files show up on reload."""
    global LOCAL_GGUF_MODELS, LOCAL_MMPROJ_FILES
    base_dirs = model_scan.resolve_base_dirs(GGUF_BASE_DIRS)
    LOCAL_GGUF_MODELS = model_scan.scan_gguf_files(base_dirs)
    LOCAL_MMPROJ_FILES = model_scan.scan_mmproj_files(base_dirs)


def list_model_names() -> list[str]:
    refresh_local_gguf()
    return list(LOCAL_GGUF_MODELS.keys())


def list_mmproj_names() -> list[str]:
    return [MMPROJ_AUTO] + list(LOCAL_MMPROJ_FILES.keys())


refresh_local_gguf()


def _filter_kwargs_for_callable(fn, kwargs: dict) -> dict:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return dict(kwargs)

    params = list(sig.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return dict(kwargs)

    allowed: set[str] = set()
    for p in params:
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            allowed.add(p.name)
    return {k: v for k, v in kwargs.items() if k in allowed}


import math


def _estimate_image_tokens(width: int, height: int) -> int:
    """Estimate Qwen2VL image tokens: ceil(H/28) * ceil(W/28)."""
    return math.ceil(height / 28) * math.ceil(width / 28)


def _resize_image_to_token_budget(pil_img: Image.Image, max_tokens: int) -> Image.Image:
    """Shrink image so its estimated token count fits within *max_tokens*."""
    w, h = pil_img.size
    cur_tokens = _estimate_image_tokens(w, h)
    if cur_tokens <= max_tokens:
        return pil_img
    scale = math.sqrt(max_tokens / cur_tokens)
    new_w = max(int(w * scale) // 28 * 28, 28)
    new_h = max(int(h * scale) // 28 * 28, 28)
    print(f"[QwenVL] Auto-resizing image from {w}x{h} ({cur_tokens} tokens) "
          f"to {new_w}x{new_h} ({_estimate_image_tokens(new_w, new_h)} tokens) to fit ctx budget")
    return pil_img.resize((new_w, new_h), Image.LANCZOS)


def _tensor_to_pil(tensor) -> Image.Image | None:
    """Convert a ComfyUI IMAGE tensor to a PIL Image."""
    if tensor is None:
        return None
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = (tensor * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(array, mode="RGB")


def _pil_to_base64_png(pil_img: Image.Image) -> str:
    """Encode a PIL Image as base64 PNG string."""
    buf = io.BytesIO()
    try:
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    finally:
        buf.close()


def _tensor_to_base64_png(tensor) -> str | None:
    pil_img = _tensor_to_pil(tensor)
    if pil_img is None:
        return None
    return _pil_to_base64_png(pil_img)


def _sample_video_frames(video, frame_count: int):
    if video is None:
        return []
    if video.ndim != 4:
        return [video]
    total = int(video.shape[0])
    frame_count = max(int(frame_count), 1)
    if total <= frame_count:
        return [video[i] for i in range(total)]
    idx = np.linspace(0, total - 1, frame_count, dtype=int)
    return [video[i] for i in idx]


def _pick_device(device_choice: str) -> str:
    if device_choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device_choice.startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    if device_choice == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _autodetect_mmproj(model_path: Path) -> Path | None:
    """First *mmproj*.gguf next to the model (matches "mmproj-*.gguf" and "*.mmproj-*.gguf")."""
    for candidate in sorted(model_path.parent.glob("*.gguf")):
        if candidate.is_file() and model_scan.is_mmproj(candidate.name):
            return candidate
    return None


def _resolve_model_entry(model_name: str, mmproj_name: str = MMPROJ_AUTO) -> GGUFVLResolved:
    base_dirs = model_scan.resolve_base_dirs(GGUF_BASE_DIRS)

    model_path = model_scan.lookup(LOCAL_GGUF_MODELS, model_name)
    if model_path is None or not model_path.is_file():
        refresh_local_gguf()
        model_path = model_scan.lookup(LOCAL_GGUF_MODELS, model_name)
    if model_path is None or not model_path.is_file():
        raise model_scan.missing_model_error(model_name, base_dirs, "GGUF model")

    if mmproj_name and mmproj_name != MMPROJ_AUTO:
        mmproj_path = model_scan.lookup(LOCAL_MMPROJ_FILES, mmproj_name)
        if mmproj_path is None or not mmproj_path.is_file():
            raise model_scan.missing_model_error(mmproj_name, base_dirs, "mmproj file")
    else:
        mmproj_path = _autodetect_mmproj(model_path)

    return GGUFVLResolved(
        display_name=model_name,
        model_path=model_path,
        mmproj_path=mmproj_path,
    )


class QwenVLGGUFBase:
    def __init__(self):
        self.llm = None
        self.chat_handler = None
        self.current_signature = None
        self._is_gemma = False

    def clear(self):
        self.llm = None
        self.chat_handler = None
        self.current_signature = None
        self._is_gemma = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_backend(self):
        try:
            from llama_cpp import Llama  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "[QwenVL] llama_cpp is not available. Install the GGUF vision dependency first. See docs/GGUF_MANUAL_INSTALL.md"
            ) from exc

    def _load_model(
        self,
        model_name: str,
        device: str,
        ctx: int | None,
        n_batch: int | None,
        gpu_layers: int | None,
        image_max_tokens: int | None,
        image_min_tokens: int | None,
        top_k: int | None,
        pool_size: int | None,
        enable_thinking: bool = False,
        mmproj_name: str = MMPROJ_AUTO,
    ):
        self._load_backend()

        resolved = _resolve_model_entry(model_name, mmproj_name)
        model_path = resolved.model_path
        mmproj_path = resolved.mmproj_path

        device_kind = _pick_device(device)

        n_ctx = int(ctx) if ctx is not None else resolved.context_length
        n_batch_val = int(n_batch) if n_batch is not None else resolved.n_batch
        top_k_val = int(top_k) if top_k is not None else resolved.top_k
        pool_size_val = int(pool_size) if pool_size is not None else resolved.pool_size

        if device_kind == "cuda":
            n_gpu_layers = int(gpu_layers) if gpu_layers is not None else resolved.gpu_layers
        else:
            n_gpu_layers = 0

        img_max = int(image_max_tokens) if image_max_tokens is not None else resolved.image_max_tokens
        img_min = int(image_min_tokens) if image_min_tokens is not None else resolved.image_min_tokens

        has_mmproj = mmproj_path is not None and mmproj_path.exists()
        is_gemma = _is_gemma_model_name(model_path.name) or _is_gemma_model_name(model_name)

        signature = (
            str(model_path),
            str(mmproj_path) if has_mmproj else "",
            n_ctx,
            n_batch_val,
            n_gpu_layers,
            img_max,
            img_min,
            top_k_val,
            pool_size_val,
        )
        if self.llm is not None and self.current_signature == signature:
            return

        self.clear()

        from llama_cpp import Llama

        self.chat_handler = None
        if has_mmproj:
            handler_cls = None
            if is_gemma:
                try:
                    from llama_cpp.llama_chat_format import Gemma4ChatHandler

                    handler_cls = Gemma4ChatHandler
                except ImportError:
                    raise RuntimeError(
                        "[QwenVL] Gemma 4 requires llama-cpp-python v0.3.35+ with Gemma4ChatHandler "
                        "(JamePeng fork). Update your llama_cpp install. See docs/GGUF_MANUAL_INSTALL.md"
                    )
            else:
                try:
                    from llama_cpp.llama_chat_format import Qwen3VLChatHandler

                    handler_cls = Qwen3VLChatHandler
                except ImportError:
                    try:
                        from llama_cpp.llama_chat_format import Qwen25VLChatHandler

                        handler_cls = Qwen25VLChatHandler
                    except ImportError:
                        raise RuntimeError(
                            "[QwenVL] Missing Qwen VL chat handler in llama_cpp. Install the correct fork/wheel. See docs/GGUF_MANUAL_INSTALL.md"
                        )

            # Build handler kwargs per family. Gemma4ChatHandler validates kwargs in
            # its parent __init__ at runtime (not via signature), so _filter_kwargs_for_callable
            # cannot protect us — pass only keys we know each handler accepts.
            mmproj_kwargs = {
                "clip_model_path": str(mmproj_path),
                "image_max_tokens": img_max,
                "verbose": False,
            }
            if not is_gemma:
                mmproj_kwargs["force_reasoning"] = False
            mmproj_kwargs = _filter_kwargs_for_callable(getattr(handler_cls, "__init__", handler_cls), mmproj_kwargs)
            if "image_max_tokens" not in mmproj_kwargs:
                print(
                    "[QwenVL] Warning: installed llama_cpp chat handler does not support image_max_tokens; "
                    "image token budget will be controlled by ctx only."
                )
            self.chat_handler = handler_cls(**mmproj_kwargs)

        llm_kwargs = {
            "model_path": str(model_path),
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "n_batch": n_batch_val,
            "swa_full": True,
            "verbose": False,
            "pool_size": pool_size_val,
            "top_k": top_k_val,
        }
        if has_mmproj and self.chat_handler is not None:
            llm_kwargs["chat_handler"] = self.chat_handler
            llm_kwargs["image_min_tokens"] = img_min
            llm_kwargs["image_max_tokens"] = img_max

        print(f"[QwenVL] Loading GGUF: {model_path.name} (device={device_kind}, gpu_layers={n_gpu_layers}, ctx={n_ctx})")
        llm_kwargs_filtered = _filter_kwargs_for_callable(getattr(Llama, "__init__", Llama), llm_kwargs)
        if has_mmproj and self.chat_handler is not None and "chat_handler" not in llm_kwargs_filtered:
            print(
                "[QwenVL] Warning: installed llama_cpp Llama() does not accept chat_handler; images will be ignored. "
                "Update llama-cpp-python to a multimodal-capable build."
            )
        if device_kind == "cuda" and n_gpu_layers == 0:
            print("[QwenVL] Warning: device=cuda selected but n_gpu_layers=0; model will run on CPU.")
        try:
            self.llm = Llama(**llm_kwargs_filtered)
        except Exception:
            self.chat_handler = None
            raise
        self._is_gemma = is_gemma
        self.current_signature = signature

    def _invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        images_b64: list[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        seed: int,
        stop_words: list[str] | None = None,
    ) -> str:
        if images_b64:
            content = [{"type": "text", "text": user_prompt}]
            for img in images_b64:
                if not img:
                    continue
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        if self._is_gemma:
            default_stop = ["<|turn>", "<|channel>", "<end_of_turn>", "<start_of_turn>"]
        else:
            default_stop = ["<|im_end|>", "<|im_start|>"]

        start = time.perf_counter()
        result = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            repeat_penalty=float(repetition_penalty),
            seed=int(seed),
            stop=default_stop + (stop_words or []),
        )
        elapsed = max(time.perf_counter() - start, 1e-6)

        usage = result.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if isinstance(completion_tokens, int) and completion_tokens > 0:
            tok_s = completion_tokens / elapsed
            if isinstance(prompt_tokens, int) and prompt_tokens >= 0:
                print(
                    f"[QwenVL] Tokens: prompt={prompt_tokens}, completion={completion_tokens}, "
                    f"time={elapsed:.2f}s, speed={tok_s:.2f} tok/s"
                )
            else:
                print(f"[QwenVL] Tokens: completion={completion_tokens}, time={elapsed:.2f}s, speed={tok_s:.2f} tok/s")

        content = (result.get("choices") or [{}])[0].get("message", {}).get("content", "")
        cleaned = clean_model_output(str(content or ""), OutputCleanConfig(mode="text"))
        return cleaned.strip()

    def run(
        self,
        model_name: str,
        preset_prompt: str,
        custom_prompt: str,
        image,
        video,
        frame_count: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        seed: int,
        keep_model_loaded: bool,
        device: str,
        ctx: int | None,
        n_batch: int | None,
        gpu_layers: int | None,
        image_max_tokens: int | None,
        image_min_tokens: int | None,
        top_k: int | None,
        pool_size: int | None,
        enable_thinking: bool = False,
        stop_words: list[str] | None = None,
        image2=None,
        image3=None,
        mmproj_name: str = MMPROJ_AUTO,
    ):
        torch.manual_seed(int(seed))

        prompt = SYSTEM_PROMPTS.get(preset_prompt, preset_prompt)
        if custom_prompt and custom_prompt.strip():
            prompt = custom_prompt.strip()

        is_gemma = _is_gemma_model_name(model_name)
        if not is_gemma:
            # Qwen uses inline /think /no_think tokens; Gemma 4 uses the handler's enable_thinking flag.
            think_prefix = "/think" if enable_thinking else "/no_think"
            prompt = f"{think_prefix}\n{prompt}"

        # Collect all PIL images first (static images + video frames)
        pil_images: list[Image.Image] = []
        for img_tensor in (image, image2, image3):
            pil = _tensor_to_pil(img_tensor)
            if pil is not None:
                pil_images.append(pil)
        if video is not None:
            for frame in _sample_video_frames(video, int(frame_count)):
                pil = _tensor_to_pil(frame)
                if pil is not None:
                    pil_images.append(pil)

        # Auto-resize images to fit within ctx budget (prevent MROPE seq_add crash on Qwen2VL).
        # Gemma 4 does not use MROPE, so we skip this Qwen-specific guard.
        if pil_images and ctx is not None and not is_gemma:
            text_token_overhead = 256  # system prompt + user prompt + formatting
            token_budget_for_images = max(ctx - max_tokens - text_token_overhead, 0)
            if token_budget_for_images == 0:
                print(f"[QwenVL] Warning: ctx={ctx} is too small for max_tokens={max_tokens}; images may cause a crash")
            else:
                per_image_budget = token_budget_for_images // len(pil_images)
                effective_cap = min(per_image_budget, image_max_tokens or per_image_budget)
                pil_images = [_resize_image_to_token_budget(img, effective_cap) for img in pil_images]

        images_b64: list[str] = [_pil_to_base64_png(img) for img in pil_images]
        del pil_images

        try:
            self._load_model(
                model_name=model_name,
                device=device,
                ctx=ctx,
                n_batch=n_batch,
                gpu_layers=gpu_layers,
                image_max_tokens=image_max_tokens,
                image_min_tokens=image_min_tokens,
                top_k=top_k,
                pool_size=pool_size,
                enable_thinking=enable_thinking,
                mmproj_name=mmproj_name,
            )
            if images_b64 and self.chat_handler is None:
                print("[QwenVL] Warning: images provided but this model entry has no mmproj_file; images will be ignored")
            text = self._invoke(
                system_prompt="You are a helpful vision-language assistant.",
                user_prompt=prompt,
                images_b64=images_b64 if self.chat_handler is not None else [],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
                stop_words=stop_words,
            )
            return (text,)
        finally:
            if not keep_model_loaded:
                self.clear()


class AILab_QwenVL_GGUF(QwenVLGGUFBase):
    @classmethod
    def INPUT_TYPES(cls):
        model_keys = model_scan.dropdown(list_model_names())
        default_model = model_keys[0]

        prompts = PRESET_PROMPTS or ["🖼️ Detailed Description"]
        preferred_prompt = "🖼️ Detailed Description"
        default_prompt = preferred_prompt if preferred_prompt in prompts else prompts[0]

        return {
            "required": {
                "model_name": (model_keys, {"default": default_model, "tooltip": TOOLTIPS["model_name"]}),
                "preset_prompt": (prompts, {"default": default_prompt}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 32768}),
                "enable_thinking": ("BOOLEAN", {"default": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "process"
    CATEGORY = "QwenVL-F"

    def process(
        self,
        model_name,
        preset_prompt,
        custom_prompt,
        max_tokens,
        enable_thinking,
        keep_model_loaded,
        seed,
        image=None,
        video=None,
    ):
        return self.run(
            model_name=model_name,
            preset_prompt=preset_prompt,
            custom_prompt=custom_prompt,
            image=image,
            video=video,
            frame_count=16,
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            seed=seed,
            keep_model_loaded=keep_model_loaded,
            device="auto",
            ctx=None,
            n_batch=None,
            gpu_layers=None,
            image_max_tokens=None,
            image_min_tokens=None,
            top_k=None,
            pool_size=None,
            enable_thinking=enable_thinking,
        )


class AILab_QwenVL_GGUF_Advanced(QwenVLGGUFBase):
    @classmethod
    def INPUT_TYPES(cls):
        model_keys = model_scan.dropdown(list_model_names())
        default_model = model_keys[0]
        mmproj_keys = list_mmproj_names()

        prompts = PRESET_PROMPTS or ["🖼️ Detailed Description"]
        preferred_prompt = "🖼️ Detailed Description"
        default_prompt = preferred_prompt if preferred_prompt in prompts else prompts[0]

        num_gpus = torch.cuda.device_count()
        gpu_list = [f"cuda:{i}" for i in range(num_gpus)]
        device_options = ["auto", "cpu", "mps"] + gpu_list

        return {
            "required": {
                "model_name": (model_keys, {"default": default_model, "tooltip": TOOLTIPS["model_name"]}),
                "mmproj_name": (mmproj_keys, {"default": MMPROJ_AUTO, "tooltip": TOOLTIPS["mmproj_name"]}),
                "device": (device_options, {"default": "auto"}),
                "preset_prompt": (prompts, {"default": default_prompt}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 32768}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 2.0}),
                "frame_count": ("INT", {"default": 16, "min": 1, "max": 64}),
                "ctx": ("INT", {"default": 32768, "min": 1024, "max": 262144, "step": 512}),
                "n_batch": ("INT", {"default": 8192, "min": 64, "max": 32768, "step": 64}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 200}),
                "image_max_tokens": ("INT", {"default": 8192, "min": 256, "max": 1024000, "step": 256}),
                "image_min_tokens": ("INT", {"default": 1024, "min": 64, "max": 1024000, "step": 64}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 32768}),
                "pool_size": ("INT", {"default": 4194304, "min": 1048576, "max": 10485760, "step": 524288}),
                "enable_thinking": ("BOOLEAN", {"default": False}),
                "stop_words": ("STRING", {"default": ""}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "video": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "process"
    CATEGORY = "QwenVL-F"

    def process(
        self,
        model_name,
        mmproj_name,
        device,
        preset_prompt,
        custom_prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        frame_count,
        ctx,
        n_batch,
        gpu_layers,
        image_max_tokens,
        image_min_tokens,
        top_k,
        pool_size,
        enable_thinking,
        stop_words,
        keep_model_loaded,
        seed,
        image=None,
        image2=None,
        image3=None,
        video=None,
    ):
        parsed = [w.strip() for w in stop_words.split(",") if w.strip()] if stop_words else None
        return self.run(
            model_name=model_name,
            preset_prompt=preset_prompt,
            custom_prompt=custom_prompt,
            image=image,
            video=video,
            frame_count=frame_count,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            keep_model_loaded=keep_model_loaded,
            device=device,
            ctx=ctx,
            n_batch=n_batch,
            gpu_layers=gpu_layers,
            image_max_tokens=image_max_tokens,
            image_min_tokens=image_min_tokens,
            top_k=top_k,
            pool_size=pool_size,
            enable_thinking=enable_thinking,
            stop_words=parsed,
            image2=image2,
            image3=image3,
            mmproj_name=mmproj_name,
        )


NODE_CLASS_MAPPINGS = {
    "QwenVL-F_GGUF": AILab_QwenVL_GGUF,
    "QwenVL-F_GGUF_Advanced": AILab_QwenVL_GGUF_Advanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVL-F_GGUF": "QwenVL-F (GGUF)",
    "QwenVL-F_GGUF_Advanced": "QwenVL-F Advanced (GGUF)",
}
