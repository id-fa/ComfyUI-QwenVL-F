# ComfyUI-QwenVL (GGUF) — llama-cpp-python compatibility layer
#
# The JamePeng fork moves fast and the GGUF nodes have to keep working across the
# wheels users already have installed:
#   - multimodal chat handlers moved from `llama_chat_format` to `llama_multimodal`
#   - `clip_model_path` was renamed to `mmproj_path`
#   - new model families arrived (Qwen3.5 / Qwen3.6 via Qwen35ChatHandler,
#     Qwen3.8 via the template-driven GenericMTMDChatHandler)
#   - speculative decoding (MTP) landed in v0.3.48 as `Llama(speculative=...)`
#
# Everything version-dependent is probed here, so an older wheel degrades with a
# console warning instead of a TypeError at node execution time.
#
# This integration script follows GPL-3.0 License.

import importlib
import inspect
import re

# --- model families ---------------------------------------------------------

FAMILY_GEMMA = "gemma4"
FAMILY_QWEN35 = "qwen3.5"      # Qwen3.5 and Qwen3.6 share Qwen35ChatHandler
FAMILY_QWEN38 = "qwen3.8"
FAMILY_QWEN_VL = "qwen-vl"     # Qwen2.5-VL / Qwen3-VL, and anything unrecognised

# Version markers are written with a dot ("Qwen3.6-27B-Q4_K_M.gguf") or, in a few
# repackaged filenames, an underscore. Hyphens are deliberately NOT accepted:
# "Qwen3-8B" is a plain Qwen3 parameter count, not Qwen3.8.
_QWEN35_RE = re.compile(r"qwen[ _-]?3[._][56](?!\d)", re.IGNORECASE)
_QWEN38_RE = re.compile(r"qwen[ _-]?3[._]8(?!\d)", re.IGNORECASE)


def detect_model_family(*names: str) -> str:
    """Guess the chat-template family from the GGUF filename / relative path."""
    text = " ".join(n or "" for n in names).lower()
    if "gemma" in text:
        return FAMILY_GEMMA
    if _QWEN38_RE.search(text):
        return FAMILY_QWEN38
    if _QWEN35_RE.search(text):
        return FAMILY_QWEN35
    return FAMILY_QWEN_VL


def is_gemma_family(*names: str) -> bool:
    return detect_model_family(*names) == FAMILY_GEMMA


# --- signature probing ------------------------------------------------------


def filter_kwargs_for_callable(fn, kwargs: dict) -> dict:
    """Drop keys the callable cannot take. Callables with **kwargs keep everything."""
    try:
        sig = inspect.signature(fn)
    except Exception:
        return dict(kwargs)

    params = list(sig.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return dict(kwargs)

    allowed = {
        p.name
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {k: v for k, v in kwargs.items() if k in allowed}


def handler_accepts(handler_cls, name: str) -> bool:
    """True when *name* is a real parameter somewhere in the handler's __init__ chain.

    Every MTMD handler declares **kwargs and validates unknown keys at runtime
    (the base class raises TypeError), so `filter_kwargs_for_callable` would wave
    every key through. Walk the MRO and look at actual parameter names instead.
    """
    for cls in getattr(handler_cls, "__mro__", [handler_cls]):
        init = cls.__dict__.get("__init__")
        if init is None:
            continue
        try:
            param = inspect.signature(init).parameters.get(name)
        except (TypeError, ValueError):
            continue
        if param is not None and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


# --- chat handlers ----------------------------------------------------------

# Handler preference per family, best first; later entries are fallbacks for
# older wheels. The mapping follows the fork's README model/handler table.
_HANDLER_CANDIDATES = {
    FAMILY_GEMMA: ("Gemma4ChatHandler",),
    FAMILY_QWEN35: ("Qwen35ChatHandler", "Qwen3VLChatHandler", "Qwen25VLChatHandler"),
    FAMILY_QWEN38: ("GenericMTMDChatHandler", "Qwen35ChatHandler", "Qwen3VLChatHandler"),
    FAMILY_QWEN_VL: ("Qwen3VLChatHandler", "Qwen25VLChatHandler"),
}

_MISSING_HANDLER_HINT = {
    FAMILY_GEMMA: "Gemma 4 requires llama-cpp-python v0.3.35+ with Gemma4ChatHandler (JamePeng fork).",
    FAMILY_QWEN35: "Qwen3.5 / Qwen3.6 need a JamePeng fork wheel that ships Qwen35ChatHandler.",
    FAMILY_QWEN38: "Qwen3.8 needs a JamePeng fork wheel that ships GenericMTMDChatHandler.",
}


class ChatHandlerBuild:
    """Result of instantiating a chat handler: the object plus what it supports."""

    def __init__(self, handler, name: str, thinking_handled: bool, image_tokens_handled: bool):
        self.handler = handler
        self.name = name
        self.thinking_handled = thinking_handled
        self.image_tokens_handled = image_tokens_handled


def import_chat_handler(name: str):
    """Import a chat handler class, preferring the newer llama_multimodal module."""
    for module_name in ("llama_cpp.llama_multimodal", "llama_cpp.llama_chat_format"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        handler_cls = getattr(module, name, None)
        if handler_cls is not None:
            return handler_cls
    return None


def build_chat_handler(
    family: str,
    mmproj_path,
    image_max_tokens: int | None = None,
    enable_thinking: bool = False,
    verbose: bool = False,
) -> ChatHandlerBuild:
    """Instantiate the chat handler for *family* with only the kwargs it accepts."""
    candidates = _HANDLER_CANDIDATES.get(family, _HANDLER_CANDIDATES[FAMILY_QWEN_VL])

    handler_cls = None
    for candidate in candidates:
        handler_cls = import_chat_handler(candidate)
        if handler_cls is not None:
            break

    if handler_cls is None:
        hint = _MISSING_HANDLER_HINT.get(
            family, "Install a vision-capable llama-cpp-python build (JamePeng fork)."
        )
        raise RuntimeError(
            f"[QwenVL] No usable chat handler for {family} in llama_cpp "
            f"(tried {', '.join(candidates)}). {hint} "
            "See docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md"
        )

    kwargs: dict[str, object] = {"verbose": verbose}
    mmproj_key = "mmproj_path" if handler_accepts(handler_cls, "mmproj_path") else "clip_model_path"
    kwargs[mmproj_key] = str(mmproj_path)

    image_tokens_handled = False
    if image_max_tokens is not None and handler_accepts(handler_cls, "image_max_tokens"):
        kwargs["image_max_tokens"] = int(image_max_tokens)
        image_tokens_handled = True

    thinking_handled = False
    if handler_cls.__name__ == "GenericMTMDChatHandler":
        # Template-driven handler: chat_format=None makes it read the template
        # embedded in the GGUF, and reasoning switches are Jinja variables.
        kwargs["chat_format"] = None
        kwargs["extra_template_arguments"] = {"enable_thinking": bool(enable_thinking)}
        thinking_handled = True
    elif handler_accepts(handler_cls, "enable_thinking"):
        kwargs["enable_thinking"] = bool(enable_thinking)
        thinking_handled = True
    elif handler_accepts(handler_cls, "force_reasoning"):
        # Qwen3-VL: thinking is steered by /think, /no_think in the user prompt.
        kwargs["force_reasoning"] = False

    return ChatHandlerBuild(
        handler_cls(**kwargs),
        handler_cls.__name__,
        thinking_handled,
        image_tokens_handled,
    )


# --- reasoning control without a handler flag -------------------------------


def text_reasoning_kwargs(family: str, enable_thinking: bool) -> dict:
    """create_chat_completion kwargs that keep a thinking model quiet.

    Used when no chat handler owns the switch (text-only runs, or an older wheel
    whose handler has no `enable_thinking`). `reasoning_budget=0` closes the
    reasoning block as soon as it opens; Qwen3.5 / 3.6 / 3.8 templates open
    `<think>` in the prompt itself, so counting must start at the first generated
    token there.
    """
    if enable_thinking:
        return {}

    kwargs: dict[str, object] = {"reasoning_budget": 0}
    if family in (FAMILY_QWEN35, FAMILY_QWEN38):
        kwargs["reasoning_start_in_prompt"] = True
    return kwargs


# --- speculative decoding (MTP) ---------------------------------------------


def build_spec_config(draft_tokens: int):
    """SpecConfig for the model's built-in NextN/MTP heads, or None when off.

    `Llama` enables target MTP tensor loading by itself when no draft model path
    is given. MTP is text-only in llama-cpp-python, so callers must not combine
    it with a multimodal chat handler.
    """
    try:
        draft_tokens = int(draft_tokens or 0)
    except (TypeError, ValueError):
        return None
    if draft_tokens <= 0:
        return None

    try:
        from llama_cpp.llama_speculative import SpecConfig, SpeculativeType
    except Exception:
        print(
            "[QwenVL] MTP requested but this llama-cpp-python build has no llama_speculative "
            "(needs v0.3.48+); continuing without speculative decoding."
        )
        return None

    config = SpecConfig(spec_type=SpeculativeType.DRAFT_MTP, draft_n_max=draft_tokens)
    try:
        config.validate()
    except Exception as exc:
        print(f"[QwenVL] MTP config rejected ({exc}); continuing without speculative decoding.")
        return None
    return config


def close_llama(llm) -> None:
    """Release native resources. close() also frees draft contexts held by MTP."""
    if llm is None:
        return
    close = getattr(llm, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as exc:
        print(f"[QwenVL] Llama.close() failed: {exc}")
