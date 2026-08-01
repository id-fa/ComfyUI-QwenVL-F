# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ComfyUI-QwenVL is a ComfyUI custom node plugin providing multimodal AI capabilities via Alibaba's Qwen-VL vision-language models. It supports two backends: HuggingFace Transformers and GGUF (llama-cpp-python), with 6 ComfyUI nodes total.

## Setup & Installation

```bash
pip install -r requirements.txt
# GGUF backend (optional): pip install -r gguf_requirements.txt
# SageAttention (optional): pip install sageattention
```

`tools/install_helper.py` は CUDA 版 PyTorch と JamePeng fork の `llama-cpp-python` wheel を対象に、
「今の環境に必要な pip コマンド」を出力する CLI（`--python` で ComfyUI portable の python.exe を指定可能、
`--run` で実行、最新導入済みならその旨を表示）。ルート直下の `.py` は `__init__.py` に全て import されるため
`tools/` 配下に置いてある。

No test suite or linter is configured. Publishing to the ComfyUI registry is handled automatically via `.github/workflows/publish.yml` when `pyproject.toml` changes on main.

## Architecture

### Node Registration

`__init__.py` dynamically scans all `.py` files, imports them, and collects `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` for ComfyUI. Each node module exports these two dicts.

### Two Backend Hierarchies

**Transformers backend** (`AILab_QwenVL.py`):
- `QwenVLBase` — base class handling model resolution (`get_local_model`), loading (`load_model`), device/VRAM management, quantization (4-bit/8-bit/FP16 via BitsAndBytes), and attention backend selection.
- `AILab_QwenVL` (simple) and `AILab_QwenVL_Advanced` (full control) inherit from it.
- Advancedノードは `image`, `image2`, `image3` の3つのoptional画像入力を持ち、複数画像の同時参照が可能。Simpleノードは `image` のみ。

**GGUF backend** (`AILab_QwenVL_GGUF.py`):
- `QwenVLGGUFBase` — base class for llama-cpp-python models with output cleaning.
- `AILab_QwenVL_GGUF` and `AILab_QwenVL_GGUF_Advanced` inherit from it.
- Advancedノードは `image`, `image2`, `image3` の3つのoptional画像入力を持ち、複数画像の同時参照が可能。Simpleノードは `image` のみ。
- **Chat handler自動選択**: モデル名（`base_dir` からの相対パス）に `gemma` が含まれる場合は `Gemma4ChatHandler` を使用（JamePeng fork v0.3.35+ 必須）。それ以外は `Qwen3VLChatHandler` → `Qwen25VLChatHandler` の順でフォールバック。判定は `_is_gemma_model_name()` によるファイル名部分一致。

**Prompt enhancers** (text-only, no vision):
- `AILab_QwenVL_PromptEnhancer.py` — Transformers-based。`keep_model_loaded` スイッチあり（HF text model用の `_invoke_text` パスと、VLモデル流用の `_invoke_qwen` パスの両方でアンロード対応）。
- `AILab_QwenVL_GGUF_PromptEnhancer.py` — GGUF-based。`keep_model_loaded` スイッチあり（`process()` 完了後に `self.clear()` でアンロード）。モデル名に `gemma` を含む場合は `chat_format="qwen"` を渡さず、GGUFに埋め込まれた chat template を llama_cpp に使わせる（それ以外は従来通り `chat_format="qwen"` を強制）。

**Output cleaning** (`AILab_OutputCleaner.py`):
- `OutputCleanConfig` dataclass and utilities to strip thinking tags and leaked tokens from model output.
- HF側 (`generate()`) でも GGUF側 (`_invoke()`) でも `clean_model_output()` を通して出力をクリーニングする。

### Key Subsystems

- **Attention resolution**: `resolve_attention_mode()` auto-selects SageAttention → Flash-Attn → SDPA with GPU architecture-aware kernel selection (`set_sage_attention`, `get_sage_attention_config`).
- **Memory management**: `enforce_memory()` auto-downgrades quantization if VRAM is insufficient; `clear()` releases models and clears CUDA cache。各バックエンドでメモリリーク対策を実施済み：
  - **中間テンソル解放**: HF側 `generate()` と `_invoke_text()` で `processed`, `model_inputs`, `outputs`, `inputs` を使用後に `del`。
  - **PIL/BytesIO解放**: GGUF側 `_pil_to_base64_png()` で BytesIO を `finally` で close。`run()` で `pil_images` をエンコード後に `del`。
  - **モデルパス切替時の解放**: HF PromptEnhancer の `process()` で textモデル⇔QwenVLパス切替時に他方のモデルを解放。`_load_text_model()` でモデル入替時に `gc.collect()` / `empty_cache()` を実行。
  - **GGUF例外安全**: `_load_model()` で Llama 初期化失敗時に `chat_handler` をクリーンアップ。
  - **GGUF PromptEnhancer**: `clear()` に `gc.collect()` / `torch.cuda.empty_cache()` を追加（VLベースクラスと同等）。
- **Device handling**: `get_device_info()` detects CUDA/MPS/CPU; `normalize_device_choice()` validates device strings.

### Local Model Discovery (カスタム改造)

**自動ダウンロードは全ノードで撤去済み。** `huggingface_hub` の `snapshot_download` / `hf_hub_download` 呼び出し、バックグラウンドDLスレッド、`_downloading_files` フラグはすべて削除。ドロップダウンは複数のベースフォルダの実スキャン結果のみで構成され、モデルが無ければ全候補フォルダを列挙した `FileNotFoundError` を投げる。

**参照フォルダ（複数ルート）** — デフォルトは `["text_encoders", "LLM"]`（先頭が優先）。`text_encoders` は ComfyUI 標準の `folder_paths` キーなので、`extra_model_paths.yaml` で追加された全ルートが自動的に含まれる。同じ相対パス名が複数ルートに存在する場合は**先頭ルートが勝ち**、無視した側をコンソールに出力する（ComfyUI 本体のマージ挙動に合わせた）。

**共通モジュール** (`AILab_ModelScan.py`) — HF・GGUF 4モジュールが共有する：
- `read_base_dirs(config_path)` — カタログJSONの `"base_dirs"`（リスト）を読む。旧 `"base_dir"`（文字列）も1要素として受け付ける。どちらも無ければ `DEFAULT_BASE_DIRS`。
- `resolve_base_dirs(values)` — 各エントリを次の優先順で解決し、重複を除いた `list[Path]` を返す：
  1. 絶対パス → そのまま
  2. `folder_paths.folder_names_and_paths` にキーがあれば `get_folder_paths()` の**全パス**（`extra_model_paths.yaml` 設定を含む）
  3. フォールバック → `{ComfyUI}/models/{value}`
- `scan_gguf_files()` / `scan_mmproj_files()` — 各ルート以下を再帰スキャンし、`.gguf` を「そのルートからの相対パス（posix区切り）」をキーに返す。`mmproj` を含むファイル名で本体/プロジェクタを振り分け、隠しディレクトリ配下は除外。
- `scan_hf_models()` — `config.json` と重みファイル（`*.safetensors` 優先、無ければ `*.bin`）が同居するディレクトリを Transformers チェックポイントとみなし、`HFLocalModel`（`is_vision` / `is_prequantized` / `weight_bytes`）を返す。`is_vision` は `config.json` の `vision_config` 等のキー、または `model_type`/`architectures`/フォルダ名のヒント語で判定。
- `lookup(entries, name)` — 旧ワークフロー互換。`[local] ` プレフィクスとバックスラッシュを剥がし、完全一致→ファイル名一致の順で探す。
- `dropdown(names)` — 空リストをプレースホルダに差し替える（ComfyUIのcomboは空にできないため）。

**各ノードでのスキャン**:
- スキャンは `INPUT_TYPES()` 呼び出しのたびに全ルートに対して実行される（`refresh_local_models()` / `refresh_local_gguf()`）。ComfyUI再読み込みで新規配置ファイルが反映される。解決時にキャッシュミスした場合も一度だけ再スキャンしてからエラーにする。
- **HF VLノード**: `is_vision=True` のチェックポイントのみ列挙。**HF PromptEnhancer**: 全チェックポイントを列挙し、`is_vision` で `_invoke_qwen`（VL流用）と `_invoke_text`（`AutoModelForCausalLM`）を分岐。
- **GGUF VLノード**: Advancedに `mmproj_name` ドロップダウン（`auto` + スキャンされたmmproj一覧）を追加。`auto` は従来どおりモデルと同ディレクトリの `*mmproj*.gguf` を自動検出。Simpleは常に `auto`。
- **GGUF PromptEnhancer**: mmproj不要。カタログ由来の `context_length` が無くなったため `ctx` ウィジェット（デフォルト8192）を追加。
- GGUFのパラメータ既定値（ctx=8192, gpu_layers=-1 等）は `GGUFVLResolved` のdataclassデフォルトが供給し、AdvancedノードのUIで上書きされる。
- `enforce_memory()` はカタログの `vram_requirement` が無くなったため、ディスク上の重みファイル合計サイズをFP16相当とみなし、8bit=1/2・4bit=1/4 で見積もる。

### Configuration Files

- `hf_models.json` — `"base_dirs"`（デフォルト `["text_encoders", "LLM"]`）のみコードが参照。`hf_vl_models` / `hf_text_models` は**未使用**（手動ダウンロード先のrepo_id台帳として残置）
- `gguf_models.json` — `"base_dirs"` のみコードが参照。`Qwen_model` / `qwenVL_model` は**未使用**（同上）
- `custom_models_example.json`, `docs/custom_models.md` — **未使用**（`custom_models.json` のマージ機能はダウンロード撤去に伴い削除済み）
- `AILab_System_Prompts.json` — system prompt presets

### Max Tokens 設定

`max_tokens` の上限値は全ノード（Simple/Advanced × HF/GGUF）で **32,768** に設定。Qwen3.5 の推奨 max_new_tokens（標準タスク 32K、複雑タスク 80K）に基づく。Thinkingモデルでは `<think>思考</think>回答` の全体がこの予算内で生成されるため、低い値では思考部分で打ち切られる。

GGUF側のデフォルト値:
- `ctx`: 32768（画像トークン + テキスト + 生成出力を収容するため。テキスト専用なら8192でも可）
- `n_batch`: 8192（`image_max_tokens` 以上が必要な制約あり）
- `image_max_tokens`: 8192（高解像度画像対応）
- `image_min_tokens`: 1024（グラウンディング/OCRタスクの最低推奨値）

GGUF PromptEnhancer（テキスト専用）の `ctx` デフォルトは 8192。

### 画像自動リサイズ（GGUF側）

GGUF側の `run()` で、入力画像のトークン数が `ctx` 予算を超える場合に自動縮小する仕組みを持つ。Qwen2VLのMROPE（n_pos_per_embd=3）ではKVキャッシュの `seq_add` がサポートされないため、コンテキストシフトが発生するとプロセスが強制終了する。これを防ぐためのガード。**Gemma 4 はMROPEを使わないためこの自動リサイズをスキップする**（`is_gemma=True` の場合 `run()` 内で分岐）。

- **トークン推定**: `_estimate_image_tokens(w, h)` = `ceil(H/28) * ceil(W/28)`（14pxパッチ × 2x2マージ）
- **予算計算**: `ctx - max_tokens - テキストオーバーヘッド(256)` を画像枚数で均等分割し、`image_max_tokens` とのmin値を各画像の上限とする
- **リサイズ**: `_resize_image_to_token_budget()` が28px単位にアラインしつつLANCZOS縮小。リサイズ時はコンソールにログ出力

### Thinking モード制御

全ノード（Simple/Advanced × HF/GGUF）に `enable_thinking` (BOOLEAN, default=False) ウィジェットがある。Qwen3-VL-*-Thinking モデル向け。

**HF Transformers側** (`AILab_QwenVL.py`):
- `apply_chat_template` に `chat_template_kwargs={"enable_thinking": enable_thinking}` を渡す。テンプレート側が `enable_thinking=False` のとき `/no_think` トークンを自動挿入する。
- `enable_thinking=None`（非Thinkingモデル）の場合は `chat_template_kwargs` 自体を送らず、テンプレート側のデフォルト動作に任せる。

**GGUF側** (`AILab_QwenVL_GGUF.py`):
- **Qwen系**: llama-cpp-python にはテンプレート制御フラグがないため、ユーザープロンプト先頭に `/think` または `/no_think` トークンを直接挿入する。ハンドラ init には `force_reasoning=False` を渡す。
- **Gemma 4系**: プレフィックス注入はスキップし、`Gemma4ChatHandler` にも thinking 関連 kwargs を渡さない（`force_reasoning` / `enable_thinking` はフォーク側 v0.3.35 の実装で親クラスが `TypeError` を投げるため）。現状 `enable_thinking` トグルは Gemma 経路では無効。将来フォーク側が安定したkwarg名を公開したら追加予定。Gemma 4 のthinkingはフォーク側の仕様で 31B / 26BA4B バリアントのみ対応（E2B / E4B は非対応）。

> **注意**: `Gemma4ChatHandler.__init__` は `**kwargs` を受け取り親クラスで runtime 検証するため、`inspect.signature` ベースの `_filter_kwargs_for_callable` ではフィルタできない。ハンドラ毎に必要最小限の kwargs だけ明示的に構築すること。

**出力側**（共通）:
- `AILab_OutputCleaner.py` の `clean_model_output()` が `<think>...</think>` ブロック、不完全な `<think>` / `</think>` タグを正規表現で除去する。
- 閉じタグのない `<think>`（`max_tokens` 不足で途中打ち切り時に発生）は `<think>` 以降のテキストをすべて除去する。
- **Gemma 4 チャンネル除去**: Gemma 4 は reasoning を `<|channel|>thought ... <|channel|>` で囲むため、`_GEMMA_CHANNEL_BLOCK_RE` でペアマッチしてブロックごと除去する。パイプの位置が揺れるレンダリング（`<|channel>`, `<channel|>`, `<|channel|>`）すべてにマッチ。ペア除去後に残った単独マーカーは、先頭にある場合は途中打ち切りの opener とみなして以降を全削除、そうでなければ closer とみなして最終マーカー以降のテキストを残す。`<|turn|>` / `<|start_of_turn|>` / `<|end_of_turn|>` 系リークトークンは `_GEMMA_TURN_TOKEN_RE` で個別に除去。

### Stop Words

Advancedノード（HF/GGUF両方）に `stop_words` (STRING) ウィジェットがある。カンマ区切りで複数のストップシーケンスを指定可能。空欄ならデフォルト動作。

- **HF側**: 各ストップワードをトークナイズし、末尾トークンIDを `eos_token_id` リストに追加。
- **GGUF側**: `create_chat_completion` の `stop` リストにそのまま文字列として追加。デフォルト値はハンドラ別に切替: Qwen系は `["<|im_end|>", "<|im_start|>"]`、Gemma 4系は `["<|turn>", "<|channel>", "<end_of_turn>", "<start_of_turn>"]`。

### UI

`web/js/appearance.js` registers a ComfyUI extension for custom node colors and sizing.

## Conventions

- All node classes follow ComfyUI's `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, `CATEGORY` class-variable pattern.
- Model files の配置先は `models/text_encoders/` と `models/LLM/`（HF・GGUF共通、両方スキャンされる）。`base_dirs` や `extra_model_paths.yaml` で変更可能。**モデルはユーザーが手動で配置する**（自動ダウンロードなし）。
- License: GPL-3.0.
