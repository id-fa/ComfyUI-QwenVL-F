# PyTorch / llama-cpp-python install helper
#
# Prints the pip commands needed to bring a target Python environment up to date
# with a CUDA build of PyTorch and a vision-capable llama-cpp-python wheel
# (JamePeng fork). Nothing is installed unless --run is passed; when everything
# is already current the tool says so instead of emitting a command.
#
# torch / torchvision / torchaudio are always planned as one matching set. When the
# official index publishes no wheel for a package/CUDA/platform combination
# (currently torchaudio on cu132 + Windows), a third-party build hosted on Hugging
# Face is used; --no-fallback turns that off, in which case the plan steps down to
# the newest CUDA index where every package lines up on the same release.
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
HF_API_ROOT = "https://huggingface.co/api"
HF_RESOLVE_ROOT = "https://huggingface.co"
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


def usable_cuda_tags(available, wanted: str | None) -> list[str]:
    """wanted 以下の CUDA タグを新しい順に。1 段下げる先の候補になる。"""
    wanted_value = cuda_tag_value(wanted or "")
    if wanted_value is None:
        return []
    pairs = []
    for tag in available:
        value = cuda_tag_value(tag)
        if value is not None and value <= wanted_value:
            pairs.append((value, tag))
    return [tag for _, tag in sorted(pairs, reverse=True)]


def fetch_index_wheels(index_tag: str, package: str, env: Environment) -> list[dict]:
    """このインデックスにある、この環境に載る wheel を列挙する。未配布なら空。"""
    page = f"{TORCH_INDEX_ROOT}/{index_tag}/{package}/"
    try:
        html = http_get(page).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return []
        raise

    platform_tag = env.platform_tag
    wheels = []
    for href in _HREF_RE.findall(html):
        target = href.split("#", 1)[0]
        filename = urllib.parse.unquote(target.strip("/").split("/")[-1])
        parsed = _WHEEL_RE.match(filename)
        if not parsed or parsed.group("name").lower() != package.lower():
            continue
        if not tag_matches(parsed.group("py"), env.py_tag):
            continue
        if not tag_matches(parsed.group("abi"), env.abi_tag):
            continue
        if platform_tag and not platform_matches(parsed.group("plat"), platform_tag):
            continue
        url = urllib.parse.urljoin(page, target)
        wheels.append(
            {
                "source": "index",
                "filename": filename,
                "version": parsed.group("version"),
                "url": url,
                # PEP 658。PyTorch のインデックスは data-core-metadata を出しており、
                # wheel 本体を落とさずに Requires-Dist を読める。
                "metadata_url": f"{url}.metadata",
            }
        )
    return wheels


def base_version(version: str | None) -> str:
    """"2.11.0+cu130" -> "2.11.0"。リリース番号の突き合わせに使う。"""
    return (version or "").split("+", 1)[0]


def latest_entry(entries: list[dict]) -> dict | None:
    if not entries:
        return None
    return max(entries, key=lambda entry: version_key(entry["version"]))


def index_by_base(entries: list[dict]) -> dict:
    """リリース番号 -> その番号のうち最新のエントリ。"""
    table: dict[str, dict] = {}
    for entry in entries:
        key = base_version(entry["version"])
        current = table.get(key)
        if current is None or version_key(entry["version"]) > version_key(current["version"]):
            table[key] = entry
    return table


_REQUIRES_TORCH_RE = re.compile(
    r"^Requires-Dist:\s*torch\s*\(?\s*==\s*([0-9][^\s,)\];]*)",
    re.IGNORECASE | re.MULTILINE,
)


def required_torch_version(metadata_text: str) -> str | None:
    match = _REQUIRES_TORCH_RE.search(metadata_text)
    return match.group(1) if match else None


def resolve_companion(entries: list[dict], release: str, limit: int = 12) -> dict | None:
    """torch==<release> を要求する最新の wheel を PEP 658 メタデータから探す。

    torchvision は torch と番号体系が違う（torch 2.11.0 <-> torchvision 0.26.0）ので、
    対応版はメタデータを読まないと特定できない。
    """
    wanted = base_version(release)
    ordered = sorted(entries, key=lambda entry: version_key(entry["version"]), reverse=True)
    for entry in ordered[:limit]:
        metadata_url = entry.get("metadata_url")
        if not metadata_url:
            continue
        try:
            text = http_get(metadata_url).decode("utf-8", "replace")
        except Exception:
            return None
        needed = required_torch_version(text)
        if needed is None:
            continue
        if base_version(needed) == wanted:
            return entry
        if version_key(needed) < version_key(wanted):
            # これより古い wheel は更に古い torch を要求するので打ち切る。
            break
    return None


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
# fallback wheels (Hugging Face)
# --------------------------------------------------------------------------- #

# 公式インデックスに wheel が置かれない組み合わせの受け皿。現状は cu132 の
# Windows 版 torchaudio だけが該当する（cu132 インデックスには torchaudio の
# CUDA ビルドが 1 つも無い）。いずれも非公式ビルドなので、明示的に列挙した
# 組み合わせにしか使わない。
FALLBACK_WHEEL_SOURCES = [
    {
        "package": "torchaudio",
        "cuda": "cu132",
        "platform": "win_amd64",
        "repo": "ussoewwin/torchaudio-built-on-cu132-for-windows",
    },
]


def find_fallback_source(package: str, index_tag: str, platform_tag: str | None):
    for source in FALLBACK_WHEEL_SOURCES:
        if source["package"] != package or source["cuda"] != index_tag:
            continue
        if platform_tag and source["platform"] != platform_tag:
            continue
        return source
    return None


def fetch_hf_wheels(repo: str, package: str, env: Environment) -> list[dict]:
    """Hugging Face リポジトリ内の wheel から、この環境に載るものを列挙する。"""
    data = json.loads(
        http_get(f"{HF_API_ROOT}/models/{repo}", accept="application/json").decode("utf-8")
    )
    platform_tag = env.platform_tag

    candidates = []
    for sibling in data.get("siblings") or []:
        path = sibling.get("rfilename") or ""
        if not path.lower().endswith(".whl"):
            continue
        filename = path.rsplit("/", 1)[-1]
        parsed = _WHEEL_RE.match(filename)
        if not parsed or parsed.group("name").lower() != package.lower():
            continue
        if not tag_matches(parsed.group("py"), env.py_tag):
            continue
        if not tag_matches(parsed.group("abi"), env.abi_tag):
            continue
        if platform_tag and not platform_matches(parsed.group("plat"), platform_tag):
            continue
        candidates.append(
            {
                "source": "fallback",
                "repo": repo,
                "filename": filename,
                "version": parsed.group("version"),
                # ファイル名の "+" はそのままでも通るが、念のためエスケープする。
                "url": f"{HF_RESOLVE_ROOT}/{repo}/resolve/main/{urllib.parse.quote(path)}",
            }
        )
    return candidates


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #

def describe(installed: str | None) -> str:
    return installed if installed else bi("未インストール", "not installed")


# 同じリリース番号で揃っていないと ABI が合わないパッケージ。torchvision も
# torch に固定されるが番号体系が違うため、resolve_companion で別途解決する。
TORCH_ALIGNED_PACKAGES = ("torch", "torchaudio")

# 1 段下げを繰り返しすぎないための上限（1 タグあたり数回の HTTP が走る）。
MAX_INDEX_ATTEMPTS = 4


def collect_pool(env: Environment, index_tag: str, package: str, use_fallback: bool) -> dict:
    """公式インデックスと（あれば）代替ビルドの wheel をまとめて集める。"""
    pool = {"index": [], "fallback": [], "repo": None, "error": None}
    try:
        pool["index"] = fetch_index_wheels(index_tag, package, env)
    except Exception as exc:
        pool["error"] = str(exc)

    if use_fallback:
        source = find_fallback_source(package, index_tag, env.platform_tag)
        if source is not None:
            pool["repo"] = source["repo"]
            try:
                pool["fallback"] = fetch_hf_wheels(source["repo"], package, env)
            except Exception as exc:
                pool["error"] = pool["error"] or str(exc)
    return pool


def merged_entries(pool: dict) -> dict:
    """リリース番号 -> エントリ。同じ番号なら公式インデックスを優先する。"""
    table = index_by_base(pool["fallback"])
    table.update(index_by_base(pool["index"]))
    return table


def choose_release(pools: dict, aligned: list[str]) -> str | None:
    """揃えるべきパッケージ全部が出せる最大のリリース番号。"""
    common: set | None = None
    for package in aligned:
        keys = set(merged_entries(pools[package]))
        common = keys if common is None else (common & keys)
    if not common:
        return None
    return max(common, key=version_key)


def resolve_index(env: Environment, targets: list[str], candidate_tags: list[str], use_fallback: bool):
    """torch と torchaudio が揃うインデックスを、新しい順に探す。

    見つからなければ最初に試したインデックスをそのまま返す（従来どおり
    「取れるものだけ最新にする」動作にフォールバックする）。
    """
    aligned = [name for name in targets if name in TORCH_ALIGNED_PACKAGES]
    first = None
    for index_tag in candidate_tags[:MAX_INDEX_ATTEMPTS]:
        pools = {name: collect_pool(env, index_tag, name, use_fallback) for name in targets}
        attempt = {"index_tag": index_tag, "pools": pools, "release": choose_release(pools, aligned)}
        first = first or attempt
        if attempt["release"] is not None:
            return attempt, first
    return first, first


def select_wheels(attempt: dict, targets: list[str]) -> tuple[dict, bool]:
    """パッケージごとに入れるべき wheel を決める。戻り値は (選択, ピン留めが必要か)。"""
    release = attempt["release"]
    pools = attempt["pools"]

    torch_entries = pools["torch"]["index"] + pools["torch"]["fallback"]
    torch_latest = latest_entry(torch_entries)
    # 最新をそのまま入れられないなら == で固定しないと pip がダウングレードしない。
    pinning = bool(
        release
        and torch_latest is not None
        and base_version(torch_latest["version"]) != release
    )

    chosen: dict[str, dict | None] = {}
    for package in targets:
        pool = pools[package]
        if package in TORCH_ALIGNED_PACKAGES and release:
            chosen[package] = merged_entries(pool).get(release)
        elif pinning and package == "torchvision":
            chosen[package] = resolve_companion(pool["index"], release)
        else:
            chosen[package] = latest_entry(pool["index"] + pool["fallback"])
    return chosen, pinning


def log_package(env: Environment, package: str, entry: dict | None) -> bool:
    """1 行分の状態表示。戻り値は「導入・更新が必要か」。"""
    installed = env.installed.get(package)
    if entry is None:
        note = bi("配布なし", "not published")
        log(f"  {package:<12}: {describe(installed)} ({note})")
        return False

    target = entry["version"]
    suffix = f"  [{bi('代替', 'fallback')}]" if entry.get("source") == "fallback" else ""

    if installed is None:
        log(f"  {package:<12}: {describe(installed)} -> {target}{suffix}")
        return True

    # PyPI 版はローカルタグを持たないので torch.version.cuda を代用する。
    installed_build = local_tag(installed) or (cuda_tag_from_version(env.torch_cuda) or "")
    target_build = local_tag(target)

    if version_key(installed) < version_key(target):
        note = bi("更新あり", "update available")
    elif version_key(installed) > version_key(target):
        note = bi("ダウングレード", "downgrade")
    elif installed_build and target_build and installed_build != target_build:
        note = bi("ビルド不一致", "build mismatch")
    else:
        log(f"  {package:<12}: {installed} ({bi('最新', 'up to date')}){suffix}")
        return False

    log(f"  {package:<12}: {installed} -> {target} ({note}){suffix}")
    return True


def plan_torch(
    env: Environment, cuda_tag: str | None, force: bool, use_fallback: bool = True
) -> list[str]:
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

    candidate_tags = usable_cuda_tags(available, cuda_tag) if cuda_tag else []
    if not candidate_tags:
        candidate_tags = ["cpu"]
        if cuda_tag:
            log_warn(
                f"{cuda_tag} 以下の CUDA ビルドが見つからないため CPU 版を対象にします。",
                f"No CUDA build at or below {cuda_tag}; targeting the CPU build instead.",
            )
    elif cuda_tag and candidate_tags[0] != cuda_tag:
        log_warn(
            f"{cuda_tag} 用のビルドが無いため {candidate_tags[0]} を使用します。",
            f"No {cuda_tag} build is published; using {candidate_tags[0]} instead.",
        )

    # torch は必ず対象。vision/audio は既に入っているものだけ追随させる。
    targets = [name for name in TORCH_PACKAGES if name == "torch" or env.installed.get(name)]

    attempt, first = resolve_index(env, targets, candidate_tags, use_fallback)
    if attempt is None:
        log("")
        return []

    index_tag = attempt["index_tag"]
    index_url = f"{TORCH_INDEX_ROOT}/{index_tag}"
    chosen, pinning = select_wheels(attempt, targets)

    if index_tag != first["index_tag"]:
        # torchaudio だけ古いインデックスに取り残される（ABI 不一致で import が
        # 落ちる）のを避けるため、全部が揃うインデックスまで下げる。
        blocked = [
            name
            for name in targets
            if name in TORCH_ALIGNED_PACKAGES and not merged_entries(first["pools"][name])
        ]
        if blocked:
            missing = "/".join(blocked)
            log_warn(
                f"{first['index_tag']} には {missing} の wheel がありません。",
                f"No {missing} wheel is published for {first['index_tag']}.",
            )
        else:
            log_warn(
                f"{first['index_tag']} では torch と torchaudio のバージョンが揃いません。",
                f"torch and torchaudio cannot be matched on {first['index_tag']}.",
            )
        log_lines(
            f"バージョンまで揃う最新の組み合わせは {index_tag} の {attempt['release']} 系です。",
            f"The newest matching set is {attempt['release']} on {index_tag}.",
            indent="      ",
        )
    elif pinning:
        # 同じインデックス内でも torchaudio の方が先に打ち止めになることがある
        # （公式の Windows 版 torchaudio は 2.11.0 で止まっている）。
        capped = []
        for package in targets:
            if package == "torch" or package not in TORCH_ALIGNED_PACKAGES:
                continue
            pool = attempt["pools"][package]
            newest = latest_entry(pool["index"] + pool["fallback"])
            if newest is not None and base_version(newest["version"]) == attempt["release"]:
                capped.append(package)
        holder = "/".join(capped) or "torchaudio"
        log_warn(
            f"{holder} は {attempt['release']} までしか配布されていないため、全体を "
            f"{attempt['release']} 系に揃えます。",
            f"{holder} is published only up to {attempt['release']}, so the whole set is "
            f"held at {attempt['release']}.",
        )

    log(f"  Index       : {index_url}")

    needs_update = False
    for package in targets:
        pool = attempt["pools"][package]
        if pool["error"]:
            note = bi("最新版の確認に失敗", "version check failed")
            log(f"  {package:<12}: {describe(env.installed.get(package))} ({note}: {pool['error']})")
            continue
        if pinning and package == "torchvision" and chosen.get(package) is None:
            # メタデータから対応版を特定できなかった場合。torchvision は torch== を
            # 宣言しているので、名前だけ渡せば pip 側で辻褄を合わせられる。
            note = bi("pip の依存解決に任せます", "left to pip's resolver")
            log(f"  {package:<12}: {describe(env.installed.get(package))} -> ({note})")
            needs_update = True
            continue
        if log_package(env, package, chosen.get(package)):
            needs_update = True

    for package in targets:
        entry = chosen.get(package)
        if entry is None or entry.get("source") != "fallback":
            continue
        log(f"  [{bi('代替', 'fallback')}] {package} <- huggingface.co/{entry['repo']}")
        log_lines(
            "非公式のコミュニティビルドです。内容を確認のうえ自己責任で導入してください。",
            "This is an unofficial community build; review it and install at your own risk.",
            indent="      ",
        )

    missing = [
        name
        for name in targets
        if chosen.get(name) is None
        and not attempt["pools"][name]["error"]
        and not (pinning and name == "torchvision")
    ]
    if missing:
        log_warn(
            f"{index_tag} で導入できないパッケージがあります: {'/'.join(missing)}",
            f"Some packages cannot be installed from {index_tag}: {'/'.join(missing)}",
        )

    commands = build_torch_commands(env, targets, chosen, pinning, index_url)

    if not commands:
        log_summary(
            "更新はありません。最新版が導入済みです。",
            "No update needed; the latest build is already installed.",
        )
        log("")
        return []

    if not needs_update:
        if not force:
            log_summary(
                "更新はありません。最新版が導入済みです。",
                "No update needed; the latest build is already installed.",
            )
            log("")
            return []
        log_summary(
            "更新は不要ですが --force が指定されたためコマンドを出力します。",
            "Already current, but --force was given, so a command is printed anyway.",
        )
    else:
        log_summary("更新が必要です。", "An update is required.")

    log("")
    return commands


def build_torch_commands(
    env: Environment, targets: list[str], chosen: dict, pinning: bool, index_url: str
) -> list[str]:
    specs = []
    fallback_commands = []
    for package in TORCH_PACKAGES:
        if package not in targets:
            continue
        entry = chosen.get(package)
        if entry is None:
            # ピン留め中に torchvision を特定できなかった場合だけ、名前だけ渡して
            # pip の依存解決に任せる（torchvision は torch== を宣言している）。
            if pinning and package == "torchvision":
                specs.append(package)
            continue
        if entry.get("source") == "fallback":
            # 現行ビルドは依存を宣言していないが、将来 torch== 付きの版が来ても CUDA 版
            # torch を PyPI 版で上書きされないよう --no-deps を付けておく。
            fallback_commands.append(
                f"{quote(env.executable)} -m pip install -U --no-deps --force-reinstall "
                f"--no-cache-dir {quote(entry['url'])}"
            )
            continue
        specs.append(quote(f"{package}=={entry['version']}") if pinning else package)

    commands = []
    if specs:
        commands.append(
            f"{quote(env.executable)} -m pip install -U {' '.join(specs)} --index-url {index_url}"
        )
    return commands + fallback_commands


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
        "--no-fallback",
        action="store_true",
        help="公式インデックスに wheel が無い場合でも Hugging Face の非公式ビルドを使いません"
             "（例: cu132 + Windows の torchaudio）。この場合は全パッケージが同一リリースで"
             "揃う下位の CUDA インデックスに切り替わります。"
             " / Do not use third-party Hugging Face builds when the official index "
             "publishes no wheel (e.g. torchaudio on cu132 + Windows); the plan then "
             "steps down to a CUDA index where every package matches.",
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
        commands += plan_torch(env, cuda_tag, args.force, not args.no_fallback)
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
