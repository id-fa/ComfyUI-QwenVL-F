# PyTorch / llama-cpp-python install helper
#
# Prints the pip commands needed to bring a target Python environment up to date
# with a CUDA build of PyTorch and a vision-capable llama-cpp-python wheel
# (JamePeng fork). Nothing is installed unless --run is passed; when everything
# is already current the tool says so instead of emitting a command.
#
# Self-contained (standard library only) so it can be dropped into any ComfyUI
# custom node pack that depends on llama-cpp-python.
#
# All user-facing output is bilingual (Japanese / English).
#
# Usage:
#   python tools/install_helper.py
#   python tools/install_helper.py --python "C:/AI/ComfyUI/python_embeded/python.exe"
#   python tools/install_helper.py --cuda cu130 --run
#
# This script follows GPL-3.0 License.

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

TORCH_INDEX_ROOT = "https://download.pytorch.org/whl"
TORCH_PACKAGES = ["torch", "torchvision", "torchaudio"]
DEFAULT_LLAMA_REPO = "JamePeng/llama-cpp-python"
USER_AGENT = "pytorch-llama-cpp-install-helper"
HTTP_TIMEOUT = 30

# Runs inside the *target* interpreter so the report describes that environment,
# not the one this script happens to be launched with.
_PROBE = r"""
import json, platform, sys, sysconfig

def dist_version(*names):
    from importlib.metadata import version
    for name in names:
        try:
            return version(name)
        except Exception:
            continue
    return None

torch_cuda = None
try:
    import torch
    torch_cuda = getattr(torch.version, "cuda", None)
except Exception:
    pass

free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
info = {
    "executable": sys.executable,
    "version": "%d.%d.%d" % sys.version_info[:3],
    "py_tag": "cp%d%d" % sys.version_info[:2],
    "abi_tag": "cp%d%d%s" % (sys.version_info[0], sys.version_info[1], "t" if free_threaded else ""),
    "system": platform.system(),
    "machine": platform.machine(),
    "torch_cuda": torch_cuda,
    "installed": {
        "torch": dist_version("torch"),
        "torchvision": dist_version("torchvision"),
        "torchaudio": dist_version("torchaudio"),
        "llama_cpp_python": dist_version("llama_cpp_python", "llama-cpp-python"),
    },
}
print("<<ENV_PROBE>>" + json.dumps(info))
"""

_WHEEL_RE = re.compile(
    r"^(?P<name>[^-]+)-(?P<version>[^-]+)"
    r"(?:-(?P<build>\d[^-]*))?"
    r"-(?P<py>[^-]+)-(?P<abi>[^-]+)-(?P<plat>[^-]+)\.whl$",
    re.IGNORECASE,
)
_HREF_RE = re.compile(r'href="([^"]+)"', re.IGNORECASE)
_CUDA_TAG_RE = re.compile(r"^cu(\d+)$")


# --------------------------------------------------------------------------- #
# bilingual output helpers
# --------------------------------------------------------------------------- #

def log(message: str = "") -> None:
    print(message)


def bi(ja: str, en: str) -> str:
    """Inline pair for short fragments: "最新 / up to date"."""
    return f"{ja} / {en}"


def log_lines(ja: str, en: str, indent: str = "  ") -> None:
    """Sentence-length messages get one line per language."""
    log(f"{indent}{ja}")
    log(f"{indent}{en}")


def log_summary(ja: str, en: str, indent: str = "  ") -> None:
    log(f"{indent}=> {ja}")
    log(f"{indent}   {en}")


def log_warn(ja: str, en: str, indent: str = "  ") -> None:
    log(f"{indent}[!] {ja}")
    log(f"{indent}    {en}")


ERROR_HEAD = "[エラー / Error] "


def display_width(text: str) -> int:
    """Console cells, so a head containing 全角 lines up with its continuation."""
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def log_error(ja: str, en: str, detail: str | None = None, indent: str = "  ") -> None:
    pad = " " * display_width(ERROR_HEAD)
    log(f"{indent}{ERROR_HEAD}{ja}")
    log(f"{indent}{pad}{en}")
    if detail:
        log(f"{indent}{pad}{detail}")


def fail(ja: str, en: str, detail: str | None = None) -> "SystemExit":
    pad = " " * display_width(ERROR_HEAD)
    message = f"{ERROR_HEAD}{ja}\n{pad}{en}"
    if detail:
        message += f"\n{detail}"
    return SystemExit(message)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def http_get(url: str, accept: str | None = None) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if accept:
        request.add_header("Accept", accept)
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token and "api.github.com" in url:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
        return response.read()


def version_key(version: str):
    """Sort key that ignores the local part ("2.9.1+cu130" == "2.9.1")."""
    base = (version or "0").split("+", 1)[0]
    match = re.match(r"^(\d+(?:\.\d+)*)(.*)$", base)
    if not match:
        return ((0,), 0, base)
    numbers = tuple(int(part) for part in match.group(1).split("."))
    suffix = match.group(2)
    # A bare release outranks any rc/a/b suffix carrying the same numbers.
    return (numbers, 0 if suffix else 1, suffix)


def local_tag(version: str | None) -> str:
    if not version or "+" not in version:
        return ""
    return version.split("+", 1)[1].lower()


def cuda_tag_value(tag: str):
    """cu130 -> (13, 0), cu121 -> (12, 1), cu92 -> (9, 2)."""
    match = _CUDA_TAG_RE.match(tag or "")
    if not match:
        return None
    digits = match.group(1)
    if len(digits) >= 3:
        return (int(digits[:-1]), int(digits[-1]))
    if len(digits) == 2:
        return (int(digits[0]), int(digits[1]))
    return None


def cuda_tag_from_version(text: str | None) -> str | None:
    """"13.0" / "12.6.3" -> "cu130" / "cu126"."""
    if not text:
        return None
    match = re.match(r"^(\d+)\.(\d+)", str(text).strip())
    if not match:
        return None
    return f"cu{int(match.group(1))}{int(match.group(2))}"


def pick_cuda_tag(available, wanted: str | None) -> str | None:
    """Highest available tag that does not exceed the detected CUDA version."""
    wanted_value = cuda_tag_value(wanted or "")
    if wanted_value is None:
        return None
    usable = []
    for tag in available:
        value = cuda_tag_value(tag)
        if value is not None and value <= wanted_value:
            usable.append((value, tag))
    if not usable:
        return None
    return max(usable)[1]


def tag_matches(field: str, wanted: str) -> bool:
    return wanted in field.split(".")


def platform_matches(field: str, wanted: str) -> bool:
    # manylinux_2_28_x86_64 must satisfy a linux_x86_64 request.
    return any(wanted in part for part in field.split("."))


def quote(value: str) -> str:
    return f'"{value}"' if (" " in value or "+" in value) else value


# --------------------------------------------------------------------------- #
# environment probe
# --------------------------------------------------------------------------- #

class Environment:
    def __init__(self, data: dict):
        self.executable = data["executable"]
        self.version = data["version"]
        self.py_tag = data["py_tag"]
        self.abi_tag = data["abi_tag"]
        self.system = data["system"]
        self.machine = data["machine"]
        self.torch_cuda = data.get("torch_cuda")
        self.installed = data.get("installed") or {}

    @property
    def platform_tag(self) -> str | None:
        system = (self.system or "").lower()
        machine = (self.machine or "").lower()
        if system == "windows":
            return "win_amd64" if machine in ("amd64", "x86_64") else None
        if system == "linux":
            return "linux_x86_64" if machine in ("x86_64", "amd64") else None
        return None


def probe_environment(python_exe: str) -> Environment:
    try:
        output = subprocess.run(
            [python_exe, "-c", _PROBE],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except FileNotFoundError:
        raise fail(
            f"Python が見つかりません: {python_exe}",
            f"Python executable not found: {python_exe}",
        )
    if output.returncode != 0:
        raise fail(
            f"環境情報の取得に失敗しました ({python_exe})",
            f"Failed to inspect the target environment ({python_exe})",
            output.stderr.strip(),
        )
    for line in output.stdout.splitlines():
        if line.startswith("<<ENV_PROBE>>"):
            return Environment(json.loads(line[len("<<ENV_PROBE>>"):]))
    raise fail(
        f"環境情報を解釈できませんでした ({python_exe})",
        f"Could not parse the environment report ({python_exe})",
    )


def detect_cuda_tag() -> tuple[str | None, str]:
    """Driver-reported CUDA first: wheels ship their own runtime, the driver caps it."""
    try:
        output = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, errors="replace", timeout=30
        )
        if output.returncode == 0:
            match = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", output.stdout)
            if match:
                return cuda_tag_from_version(match.group(1)), "nvidia-smi"
    except Exception:
        pass

    try:
        output = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, errors="replace", timeout=30
        )
        if output.returncode == 0:
            match = re.search(r"release\s+([0-9]+\.[0-9]+)", output.stdout)
            if match:
                return cuda_tag_from_version(match.group(1)), "nvcc"
    except Exception:
        pass

    return None, bi("検出できず", "not detected")


# --------------------------------------------------------------------------- #
# PyTorch index
# --------------------------------------------------------------------------- #

def fetch_torch_cuda_tags() -> list[str]:
    html = http_get(f"{TORCH_INDEX_ROOT}/").decode("utf-8", "replace")
    tags = set()
    for href in _HREF_RE.findall(html):
        name = href.strip("/").split("/")[-1]
        if _CUDA_TAG_RE.match(name):
            tags.add(name)
    return sorted(tags)


def fetch_latest_torch_version(index_tag: str, package: str, env: Environment) -> str | None:
    url = f"{TORCH_INDEX_ROOT}/{index_tag}/{package}/"
    try:
        html = http_get(url).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise

    platform_tag = env.platform_tag
    best = None
    for href in _HREF_RE.findall(html):
        filename = urllib.parse.unquote(href.split("#", 1)[0].strip("/").split("/")[-1])
        parsed = _WHEEL_RE.match(filename)
        if not parsed or parsed.group("name").lower() != package.lower():
            continue
        if not tag_matches(parsed.group("py"), env.py_tag):
            continue
        if not tag_matches(parsed.group("abi"), env.abi_tag):
            continue
        if platform_tag and not platform_matches(parsed.group("plat"), platform_tag):
            continue
        version = parsed.group("version")
        if best is None or version_key(version) > version_key(best):
            best = version
    return best


# --------------------------------------------------------------------------- #
# llama-cpp-python releases
# --------------------------------------------------------------------------- #

def fetch_llama_assets(repo: str, per_page: int, env: Environment) -> list[dict]:
    url = f"https://api.github.com/repos/{repo}/releases?per_page={per_page}"
    releases = json.loads(http_get(url, accept="application/vnd.github+json").decode("utf-8"))
    platform_tag = env.platform_tag

    candidates = []
    for release in releases:
        for asset in release.get("assets", []):
            filename = asset.get("name", "")
            if not filename.lower().endswith(".whl"):
                continue
            parsed = _WHEEL_RE.match(filename)
            if not parsed:
                continue
            if not tag_matches(parsed.group("py"), env.py_tag):
                continue
            if not tag_matches(parsed.group("abi"), env.abi_tag):
                continue
            if platform_tag and not platform_matches(parsed.group("plat"), platform_tag):
                continue
            version = parsed.group("version")
            build_tag = local_tag(version)
            candidates.append(
                {
                    "filename": filename,
                    "version": version,
                    "cuda": build_tag if _CUDA_TAG_RE.match(build_tag) else "",
                    "url": asset.get("browser_download_url"),
                }
            )
    return candidates


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #

def describe(installed: str | None) -> str:
    return installed if installed else bi("未インストール", "not installed")


def plan_torch(env: Environment, cuda_tag: str | None, force: bool) -> list[str]:
    log("[PyTorch]")

    if env.platform_tag is None:
        log_lines(
            f"このプラットフォームは非対応です ({env.system}/{env.machine})。手動で導入してください。",
            f"Unsupported platform ({env.system}/{env.machine}); install it manually.",
        )
        log("")
        return []

    try:
        available = fetch_torch_cuda_tags()
    except Exception as exc:
        log_error(
            "PyTorch インデックスを取得できませんでした",
            "Failed to fetch the PyTorch index",
            str(exc),
        )
        log("")
        return []

    index_tag = pick_cuda_tag(available, cuda_tag) if cuda_tag else None
    if index_tag is None:
        index_tag = "cpu"
        if cuda_tag:
            log_warn(
                f"{cuda_tag} 以下の CUDA ビルドが見つからないため CPU 版を対象にします。",
                f"No CUDA build at or below {cuda_tag}; targeting the CPU build instead.",
            )
    elif cuda_tag and index_tag != cuda_tag:
        log_warn(
            f"{cuda_tag} 用のビルドが無いため {index_tag} を使用します。",
            f"No {cuda_tag} build is published; using {index_tag} instead.",
        )

    index_url = f"{TORCH_INDEX_ROOT}/{index_tag}"
    log(f"  Index       : {index_url}")

    # torch は必ず対象。vision/audio は既に入っているものだけ追随させる。
    targets = [name for name in TORCH_PACKAGES if name == "torch" or env.installed.get(name)]

    needs_update = False
    unavailable: list[str] = []
    for package in targets:
        installed = env.installed.get(package)
        try:
            latest = fetch_latest_torch_version(index_tag, package, env)
        except Exception as exc:
            note = bi("最新版の確認に失敗", "version check failed")
            log(f"  {package:<12}: {describe(installed)} ({note}: {exc})")
            continue

        if latest is None:
            # このインデックスに存在しない以上、-U の対象には含められない。
            note = bi(f"{index_tag} 用の配布なし", f"not published for {index_tag}")
            log(f"  {package:<12}: {describe(installed)} ({note})")
            unavailable.append(package)
            continue

        if installed is None:
            log(f"  {package:<12}: {describe(installed)} -> {latest}")
            needs_update = True
            continue

        # PyPI 版はローカルタグを持たないので torch.version.cuda を代用する。
        build = local_tag(installed) or (cuda_tag_from_version(env.torch_cuda) or "")
        if version_key(installed) < version_key(latest):
            note = bi("更新あり", "update available")
            log(f"  {package:<12}: {installed} -> {latest} ({note})")
            needs_update = True
        elif build and build != index_tag:
            note = bi("ビルド不一致", "build mismatch")
            log(f"  {package:<12}: {installed} ({note}: {build} != {index_tag})")
            needs_update = True
        else:
            log(f"  {package:<12}: {installed} ({bi('最新', 'up to date')})")

    if not needs_update and not force:
        log_summary(
            "更新はありません。最新版が導入済みです。",
            "No update needed; the latest build is already installed.",
        )
        log("")
        return []

    if force and not needs_update:
        log_summary(
            "更新は不要ですが --force が指定されたためコマンドを出力します。",
            "Already current, but --force was given, so a command is printed anyway.",
        )
    else:
        log_summary("更新が必要です。", "An update is required.")

    installable = [package for package in targets if package not in unavailable]
    if not installable:
        log_warn(
            f"{index_tag} 用に導入できるパッケージがありません。",
            f"No package can be installed from the {index_tag} index.",
        )
        log("")
        return []

    command = (
        f"{quote(env.executable)} -m pip install -U "
        f"{' '.join(installable)} --index-url {index_url}"
    )
    log("")
    return [command]


def plan_llama(env: Environment, cuda_tag: str | None, repo: str, per_page: int, force: bool) -> list[str]:
    log(f"[llama-cpp-python ({repo})]")

    installed = env.installed.get("llama_cpp_python")
    log(f"  Installed   : {describe(installed)}")

    if env.platform_tag is None:
        log_lines(
            f"このプラットフォーム向けの wheel は配布されていません ({env.system}/{env.machine})。",
            f"No wheel is published for this platform ({env.system}/{env.machine}).",
        )
        log("")
        return []

    try:
        candidates = fetch_llama_assets(repo, per_page, env)
    except Exception as exc:
        log_error(
            "リリース情報を取得できませんでした",
            "Failed to fetch the release list",
            str(exc),
        )
        log("")
        return []

    if not candidates:
        log_warn(
            f"{env.py_tag}/{env.platform_tag} に一致する wheel が見つかりませんでした。",
            f"No wheel matches {env.py_tag}/{env.platform_tag}.",
        )
        log_lines(
            f"https://github.com/{repo}/releases を直接確認してください。",
            f"Check https://github.com/{repo}/releases directly.",
            indent="      ",
        )
        log("")
        return []

    build_tag = pick_cuda_tag({item["cuda"] for item in candidates if item["cuda"]}, cuda_tag)
    if build_tag is None:
        cpu_candidates = [item for item in candidates if not item["cuda"]]
        if not cpu_candidates:
            log_warn(
                f"{cuda_tag or 'CPU'} に適合する wheel が見つかりませんでした。",
                f"No wheel matches {cuda_tag or 'CPU'}.",
            )
            log("")
            return []
        if cuda_tag:
            log_warn(
                f"{cuda_tag} 以下の CUDA ビルドが無いため CPU 版を対象にします。",
                f"No CUDA build at or below {cuda_tag}; targeting the CPU build instead.",
            )
        pool = cpu_candidates
        build_tag = "cpu"
    else:
        if cuda_tag and build_tag != cuda_tag:
            log_warn(
                f"{cuda_tag} 用のビルドが無いため {build_tag} を使用します。",
                f"No {cuda_tag} build is published; using {build_tag} instead.",
            )
        pool = [item for item in candidates if item["cuda"] == build_tag]

    best = max(pool, key=lambda item: version_key(item["version"]))
    log(f"  Latest      : {best['version']}  ({best['filename']})")

    reason_ja = reason_en = ""
    if installed is None:
        needs_update = True
        reason_ja, reason_en = "未インストールです。", "It is not installed."
    elif version_key(installed) < version_key(best["version"]):
        needs_update = True
        reason_ja, reason_en = "新しいバージョンがあります。", "A newer version is available."
    elif local_tag(installed) and local_tag(installed) != local_tag(best["version"]):
        needs_update = True
        detail = f"({local_tag(installed)} != {build_tag})"
        reason_ja = f"バージョンは同じですがビルドが異なります {detail}"
        reason_en = f"Same version, different build {detail}"
    else:
        needs_update = False

    if not needs_update and not force:
        log_summary(
            "更新はありません。最新版が導入済みです。",
            "No update needed; the latest build is already installed.",
        )
        log("")
        return []

    if needs_update:
        log_summary(reason_ja, reason_en)
    else:
        log_summary(
            "更新は不要ですが --force が指定されたためコマンドを出力します。",
            "Already current, but --force was given, so a command is printed anyway.",
        )

    command = (
        f"{quote(env.executable)} -m pip install -U --force-reinstall --no-cache-dir "
        f"{quote(best['url'])}"
    )
    log("")
    return [command]


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CUDA 版 PyTorch と vision 対応 llama-cpp-python の "
                    "pip コマンドを取得します（既に最新ならその旨を表示します）。\n"
                    "Works out the pip commands this environment needs for a CUDA build of "
                    "PyTorch and a vision-capable llama-cpp-python wheel (and says so when "
                    "everything is already current).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="例 / Examples:\n"
               "  python tools/install_helper.py\n"
               '  python tools/install_helper.py --python "C:/AI/ComfyUI/python_embeded/python.exe"\n'
               "  python tools/install_helper.py --cuda cu130 --run\n",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="対象の Python 実行ファイル（既定: このスクリプトを実行中の Python）。"
             "ComfyUI portable なら python_embeded/python.exe を指定します。"
             "指定した場合その環境を activate しておく必要はありません。"
             "省略時は実行中の環境が対象になるため、venv は activate してから実行してください。"
             " / Target Python executable (default: the one running this script). "
             "For ComfyUI portable, point it at python_embeded/python.exe. "
             "No need to activate that environment when this is given; without it the "
             "currently active environment is inspected instead.",
    )
    parser.add_argument(
        "--cuda",
        help="CUDA タグを手動指定します（例: cu130、CPU 版なら cpu）。既定は自動検出。"
             " / Force a CUDA tag (e.g. cu130, or cpu for CPU builds). Default: auto-detect.",
    )
    parser.add_argument(
        "--no-torch",
        action="store_true",
        help="PyTorch のチェックを行いません。 / Skip the PyTorch check.",
    )
    parser.add_argument(
        "--no-llama",
        action="store_true",
        help="llama-cpp-python のチェックを行いません。 / Skip the llama-cpp-python check.",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_LLAMA_REPO,
        help=f"llama-cpp-python の wheel 配布リポジトリ（既定: {DEFAULT_LLAMA_REPO}）。"
             f" / Repository publishing the llama-cpp-python wheels (default: {DEFAULT_LLAMA_REPO}).",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=100,
        help="GitHub から取得するリリース数（既定: 100）。"
             " / Number of GitHub releases to scan (default: 100).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="最新版が導入済みでもコマンドを出力します（再インストール用）。"
             " / Print a command even when everything is current (for a reinstall).",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="表示した pip コマンドをそのまま実行します（ComfyUI を停止してから使用）。"
             " / Execute the printed pip commands (stop ComfyUI first).",
    )
    return parser


def main(argv=None) -> int:
    # cp932 コンソールでも日本語出力で落ちないようにする。
    reconfigure = getattr(sys.stdout, "reconfigure", None)  # type: ignore[attr-defined]
    if reconfigure is not None:
        try:
            reconfigure(errors="replace")
        except Exception:
            pass

    args = build_parser().parse_args(argv)

    log("=" * 66)
    log(" PyTorch / llama-cpp-python インストールヘルパー")
    log(" PyTorch / llama-cpp-python install helper")
    log("=" * 66)

    env = probe_environment(args.python)

    if args.cuda:
        cuda_tag = None if args.cuda.lower() == "cpu" else args.cuda.lower()
        cuda_source = bi("手動指定", "manual")
    else:
        cuda_tag, cuda_source = detect_cuda_tag()

    log("[環境 / Environment]")
    log(f"  Python      : {env.version} ({env.py_tag}/{env.abi_tag}, {env.platform_tag or env.machine})")
    log(f"  Executable  : {env.executable}")
    log(f"  CUDA        : {cuda_tag or bi('なし', 'none')}  ({cuda_source})")
    if not cuda_tag:
        log_lines(
            "CPU 版を対象にします。",
            "CPU builds will be targeted.",
            indent="                ",
        )
    if env.torch_cuda:
        log(f"  torch build : cuda {env.torch_cuda}")
    log("")

    commands: list[str] = []
    if not args.no_torch:
        commands += plan_torch(env, cuda_tag, args.force)
    if not args.no_llama:
        commands += plan_llama(env, cuda_tag, args.repo, args.search_limit, args.force)

    log("-" * 66)
    if not commands:
        if args.no_torch and args.no_llama:
            log_lines(
                "チェック対象がありません（--no-torch と --no-llama の両方が指定されています）。",
                "Nothing to check (both --no-torch and --no-llama were given).",
                indent="",
            )
        else:
            log_lines(
                "実行が必要なコマンドはありません。すべて最新版が導入済みです。",
                "No command to run; everything is already up to date.",
                indent="",
            )
        return 0

    log_lines(
        "以下の pip コマンドを実行してください:",
        "Run the following pip command(s):",
        indent="",
    )
    log("")
    for command in commands:
        log(f"  {command}")
    log("")
    if os.name == "nt":
        log_lines(
            '※ PowerShell では先頭に & を付けてください（例: & "C:\\...\\python.exe" -m pip ...）。',
            '   In PowerShell, prefix the command with & (e.g. & "C:\\...\\python.exe" -m pip ...).',
            indent="",
        )
    log_lines(
        "※ ComfyUI を終了してから実行してください。",
        "   Stop ComfyUI before running it.",
        indent="",
    )

    if not args.run:
        log_lines(
            "※ --run を付けて実行すると、上記コマンドをこのツールが実行します。",
            "   Re-run with --run to have this tool execute the commands above.",
            indent="",
        )
        return 0

    log("")
    log_lines(
        "--run が指定されました。コマンドを実行します。",
        "--run was given; executing the commands now.",
        indent="",
    )
    for command in commands:
        log("")
        log(f"$ {command}")
        result = subprocess.run(command, shell=True)
        if result.returncode != 0:
            log_error(
                f"コマンドが失敗しました (exit={result.returncode})",
                f"The command failed (exit={result.returncode})",
                indent="",
            )
            return result.returncode
    log("")
    log_lines("完了しました。", "Done.", indent="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
