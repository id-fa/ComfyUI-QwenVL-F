# ComfyUI-QwenVL GGUF prompt enhancer
#
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

import gc
import json
import os
import re
from pathlib import Path

import torch
from llama_cpp import Llama

from AILab_OutputCleaner import OutputCleanConfig, clean_model_output
import AILab_ModelScan as model_scan

NODE_DIR = Path(__file__).parent
GGUF_CONFIG_PATH = NODE_DIR / "gguf_models.json"
PROMPT_CONFIG_PATH = NODE_DIR / "AILab_System_Prompts.json"


def load_prompt_config():
    if not PROMPT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"[QwenVL] Missing AILab_System_Prompts.json at {PROMPT_CONFIG_PATH}")
    try:
        with open(PROMPT_CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        qwen_text = data.get("qwen_text") or {}
        styles = qwen_text.get("styles")
        translation_prompt = qwen_text.get("translation_prompt")
        if not styles or not translation_prompt:
            raise ValueError("AILab_System_Prompts.json must include qwen_text.styles and qwen_text.translation_prompt")
        return {"styles": styles, "translation_prompt": translation_prompt}
    except Exception as exc:
        raise RuntimeError(f"[QwenVL] Failed to load AILab_System_Prompts.json: {exc}") from exc


PROMPT_CONFIG = load_prompt_config()
STYLES = PROMPT_CONFIG.get("styles", {})


GGUF_BASE_DIRS: list[str] = model_scan.read_base_dirs(GGUF_CONFIG_PATH)
LOCAL_GGUF_MODELS: dict[str, Path] = {}


def _is_gemma_model_name(name: str) -> bool:
    """Detect Gemma models by filename substring (covers relative paths)."""
    return "gemma" in (name or "").lower()


def refresh_local_gguf() -> dict[str, Path]:
    """Re-scan the base dirs. Called from INPUT_TYPES so new files show up on reload."""
    global LOCAL_GGUF_MODELS
    LOCAL_GGUF_MODELS = model_scan.scan_gguf_files(model_scan.resolve_base_dirs(GGUF_BASE_DIRS))
    return LOCAL_GGUF_MODELS


def list_model_names() -> list[str]:
    return list(refresh_local_gguf().keys())


class AILab_QwenVL_GGUF_PromptEnhancer:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ENHANCED_OUTPUT",)
    FUNCTION = "process"
    CATEGORY = "QwenVL-F"

    def __init__(self):
        self.llm = None
        self.current_signature = None
        self.styles = STYLES

    @classmethod
    def INPUT_TYPES(cls):
        styles = list(STYLES.keys())
        preferred_style = "📝 Enhance"
        default_style = preferred_style if preferred_style in styles else (styles[0] if styles else "📝 Enhance")
        model_keys = model_scan.dropdown(list_model_names())
        default_model = model_keys[0]
        return {
            "required": {
                "model_name": (model_keys, {"default": default_model, "tooltip": "Pick a .gguf already present under models/text_encoders or models/LLM. Nothing is downloaded automatically — copy the file in yourself, then reload ComfyUI."}),
                "prompt_text": ("STRING", {"default": "", "multiline": True, "tooltip": "Prompt text to enhance. Leave blank to just emit the preset instruction."}),
                "preset_system_prompt": (styles, {"default": default_style}),
                "custom_system_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 256, "min": 32, "max": 1024}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 2.0}),
                "ctx": ("INT", {"default": 8192, "min": 1024, "max": 262144, "step": 512, "tooltip": "Context length passed to llama.cpp."}),
                "english_output": ("BOOLEAN", {"default": False, "tooltip": "Force final output in English using translation prompt."}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto", "tooltip": "Select device; auto prefers GPU when available."}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "tooltip": "Keep the model in memory after execution. Disable to free VRAM."}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
            }
        }

    def clear(self):
        self.llm = None
        self.current_signature = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _resolve_model_path(self, model_name) -> Path:
        resolved = model_scan.lookup(LOCAL_GGUF_MODELS, model_name)
        if resolved is None or not resolved.is_file():
            resolved = model_scan.lookup(refresh_local_gguf(), model_name)
        if resolved is None or not resolved.is_file():
            raise model_scan.missing_model_error(
                model_name, model_scan.resolve_base_dirs(GGUF_BASE_DIRS), "GGUF model"
            )
        return resolved

    def _load_model(self, model_name, device, context_length):
        resolved = self._resolve_model_path(model_name)
        context_length = int(context_length)
        signature = (resolved, context_length, device)
        if self.llm is not None and self.current_signature == signature:
            return
        self.clear()
        print(f"[QwenVL] Loading GGUF model from {resolved}")
        if device == "auto":
            device_choice = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        else:
            device_choice = device
        auto_gpu_layers = -1 if device_choice == "cuda" else 0
        threads = None
        if device_choice == "cpu":
            threads = max(os.cpu_count() or 1, 1)
        kwargs = {
            "model_path": str(resolved),
            "n_ctx": context_length,
            "n_gpu_layers": auto_gpu_layers,
            "n_threads": None if threads == 0 else threads,
            "n_batch": 1024,
            "verbose": False,
        }
        # Qwen text-only GGUFs need an explicit chat_format. Gemma ships its own
        # chat template inside the GGUF, so let llama_cpp pick it up automatically.
        if not _is_gemma_model_name(model_name):
            kwargs["chat_format"] = "qwen"
        self.llm = Llama(**kwargs)
        self.current_signature = signature

    def _invoke_llama(
        self,
        system_prompt,
        user_prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        seed,
    ):
        def _looks_like_planning(text: str) -> bool:
            if not text:
                return False
            return bool(
                re.search(
                    r"(?im)^\s*(okay[,.:]?|first[,.:]?|next[,.:]?|then[,.:]?|wait[,.:]?)\b",
                    text,
                )
                or re.search(r"(?i)\b(i\s+(should|need|must|will|am\s+going\s+to|have\s+to))\b", text)
            )

        def _call(system: str, user: str, temp: float, seed_val: int) -> str:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temp,
                top_p=top_p,
                repeat_penalty=repetition_penalty,
                seed=seed_val,
            )
            if not response or "choices" not in response or not response["choices"]:
                raise RuntimeError("[QwenVL] llama_cpp returned empty response")
            return (response["choices"][0].get("message", {}).get("content", "") or "").strip()

        raw = _call(system_prompt, user_prompt, float(temperature), int(seed))
        cleaned = clean_model_output(raw, OutputCleanConfig(mode="prompt"))

        # If the model only emitted thinking/planning (common with some Qwen variants),
        # do a single constrained retry asking for final prompt text only.
        if not cleaned or _looks_like_planning(cleaned) or "<think" in raw.lower():
            retry_system = (
                "You are a professional photography prompt writer.\n"
                "Output ONLY ONE final photography prompt paragraph.\n"
                "No analysis, no planning steps, no first-person, and no <think>.\n"
                "No bullet points, no headings, no JSON, no markdown, no quotes."
            )
            retry_user = (
                "Rewrite the following into the final prompt paragraph:\n\n"
                f"{raw}\n"
            )
            raw_retry = _call(retry_system, retry_user, 0.4, int(seed) + 999)
            cleaned_retry = clean_model_output(raw_retry, OutputCleanConfig(mode="prompt"))
            if cleaned_retry and not _looks_like_planning(cleaned_retry):
                return cleaned_retry

        return cleaned or ""

    def process(
        self,
        model_name,
        prompt_text,
        preset_system_prompt,
        custom_system_prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        ctx,
        english_output,
        device,
        keep_model_loaded,
        seed,
    ):
        style_entry = self.styles.get(preset_system_prompt, {})
        system_prompt = (custom_system_prompt.strip() or style_entry.get("system_prompt") or "").strip()
        if not system_prompt:
            raise ValueError("system_prompt is empty; check AILab_System_Prompts.json or preset selection.")
        system_prompt = (
            f"{system_prompt}\n\n"
            "Return only the final prompt text. No preface, no explanations, no analysis, no JSON, no markdown fences, and no <think>.\n"
            "Do not write planning steps (no 'First', 'Next', 'Then') and do not use first-person ('I', 'we')."
        )
        user_prompt = prompt_text.strip() or "Describe a scene vividly."
        merged_prompt = user_prompt
        self._load_model(model_name, device, ctx)
        enhanced = self._invoke_llama(
            system_prompt=system_prompt,
            user_prompt=merged_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
        if english_output:
            translated = self._invoke_llama(
                system_prompt=(
                    PROMPT_CONFIG.get("translation_prompt")
                    or "Return a single English paragraph (150-300 words). No prefixes, bullets, JSON, or <think>. "
                    "Cover subject, environment, lighting, camera settings, composition, color/texture, and style. Output only the prompt."
                ),
                user_prompt=enhanced,
                max_tokens=max_tokens,
                temperature=0.3,
                top_p=0.95,
                repetition_penalty=1.05,
                seed=seed + 1,
            )
            final = clean_model_output(translated, OutputCleanConfig(mode="prompt")) or translated.strip()
        else:
            final = clean_model_output(enhanced, OutputCleanConfig(mode="prompt")) or enhanced.strip()
        if not keep_model_loaded:
            self.clear()
        return (final,)

    @staticmethod
    def _is_english(text):
        letters = len(re.findall(r"[A-Za-z]", text))
        tokens = len(re.findall(r"\S", text))
        return tokens > 0 and letters / tokens > 0.7

    @staticmethod
    def _strip_think(text):
        return clean_model_output(text, OutputCleanConfig(mode="prompt"))


NODE_CLASS_MAPPINGS = {
    "QwenVL-F_GGUF_PromptEnhancer": AILab_QwenVL_GGUF_PromptEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVL-F_GGUF_PromptEnhancer": "QwenVL-F Prompt Enhancer (GGUF)",
}
