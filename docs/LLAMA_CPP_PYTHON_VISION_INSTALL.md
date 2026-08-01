# Install `llama-cpp-python` (Vision / Qwen-VL GGUF)

This plugin’s **QwenVL (GGUF)** vision nodes require a `llama-cpp-python` build that includes multimodal chat handlers such as:

- `Qwen3VLChatHandler`
- `Qwen25VLChatHandler`
- `Gemma4ChatHandler` (Gemma 4 GGUF, JamePeng fork v0.3.35 or later — see [discussion #109](https://github.com/JamePeng/llama-cpp-python/discussions/109))

The upstream `llama-cpp-python` from PyPI often does **not** include these vision handlers. Use a fork/build that provides them (e.g. JamePeng’s fork) and install a **Release wheel**.

Release wheels (download `.whl` here):

- [https://github.com/JamePeng/llama-cpp-python/releases](https://github.com/JamePeng/llama-cpp-python/releases/)

## Quick start: let the helper build the pip command

`tools/install_helper.py` detects your CUDA version, checks the PyTorch index and the
JamePeng release assets, and prints the exact `pip install` command for the Python
environment you point it at. It installs nothing unless you pass `--run`, and it tells
you when everything is already up to date instead of printing a command.

```bat
python tools\install_helper.py --python "C:\AI\ComfyUI\python_embeded\python.exe"
```

Useful flags: `--cuda cu130` (skip detection), `--run` (execute the printed commands),
`--force` (re-emit a command even when current), `--no-torch` / `--no-llama`.

With `--python` you do **not** need the target env activated — the helper builds
commands around that absolute path, and `--run` invokes it directly. Drop `--python`
only when the env is already active, otherwise you will be reporting on whichever
Python happens to be first on your PATH. On conda, activating is still the safer
route on Windows: without it the probe's `import torch` can fail on DLL lookup, which
costs the helper its `torch.version.cuda` reading (it then falls back to the version's
local tag, e.g. `+cu130`).

## 0) Close ComfyUI first

Stop ComfyUI before installing/replacing packages, especially on Windows portable.

## 1) Identify the exact Python ComfyUI uses

### Windows portable (common)

Your Python is usually:

`ComfyUI\\python_embeded\\python.exe`

Check:

```bat
C:\AI\ComfyUI\python_embeded\python.exe -V
C:\AI\ComfyUI\python_embeded\python.exe -c "import sys; print(sys.executable)"
```

### venv / conda

Activate your env, then:

```bash
python -V
python -c "import sys; print(sys.executable)"
```

## 2) Backup your environment (recommended)

```bat
C:\AI\ComfyUI\python_embeded\python.exe -m pip freeze > C:\AI\ComfyUI\requirements-backup.txt
```

## 3) Install the Release wheel (recommended)

Download a **Release `.whl`** from:
[https://github.com/JamePeng/llama-cpp-python/releases/](https://github.com/JamePeng/llama-cpp-python/releases/)

The wheel **must match ALL of the following**:

* **Python version** used by ComfyUI
  (`cp310` / `cp311` / `cp312` / `cp313`)
* **Platform**
  `win_amd64` (Windows 64-bit)
* **Build type**

  * **CPU wheel** → safest option (no CUDA toolkit required)
  * **CUDA wheel (`cuXXX`)** → requires a compatible CUDA runtime / toolkit

> [!WARNING]
> **Windows note (important)**
> If you install a CUDA wheel, the CUDA build tag (e.g. `cu121`, `cu122`) must be compatible with your installed CUDA runtime/toolkit.

A mismatch can cause errors like **“cannot load ggml.dll” even though the file exists**.

If you are unsure, **use a CPU wheel**.

Install with force-reinstall (safer than manual uninstall):

```
C:\AI\ComfyUI\python_embeded\python.exe -m pip install --upgrade --force-reinstall C:\path\to\llama_cpp_python-*.whl
```

Notes:

* Warnings about leftover folders like `~umpy` are usually safe to ignore while ComfyUI is closed.
* Make sure ComfyUI is **fully stopped** before installing.

## 4) Verify vision handlers exist

```bat
C:\AI\ComfyUI\python_embeded\python.exe -c "from llama_cpp.llama_chat_format import Qwen3VLChatHandler, Qwen25VLChatHandler; print('handlers OK')"
```

If this fails, you installed a wheel that does not include vision support (or installed into the wrong Python environment).

## 5) Fix common dependency conflicts (Windows)

Some wheels may upgrade dependencies (notably `numpy` / `pillow`) and cause conflicts with other packages (like OpenCV).

### OpenCV conflict (recommended fix)

If you see errors like:

- `opencv-python ... requires numpy<2.3.0,>=2; ... but you have numpy 2.3.x`

Pin numpy back:

```bat
C:\AI\ComfyUI\python_embeded\python.exe -m pip install --upgrade "numpy<2.3"
```

### Pillow conflicts (optional)

If you don’t use packages that depend on an older Pillow, you can ignore Pillow warnings. Otherwise:

```bat
C:\AI\ComfyUI\python_embeded\python.exe -m pip install --upgrade "pillow<12"
```
