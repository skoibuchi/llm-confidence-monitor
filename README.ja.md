# LLM Confidence Monitor — LLM内部状態分析ツール

本プロジェクトは、LLMの中間層の状態（hidden states）から、モデルの不確実性を推定できる可能性を検証する実験的なツールです。

LLMは必要な知識を持っていない場合でも、もっともらしい回答を生成することがあります。
本プロジェクトでは、中間層の表現に不確実性を推定したり、ハルシネーションの兆候を検出したりするために利用可能なシグナルが含まれているかを検証します。

> English version: [README.md](./README.md)

## 概要

<img alt="inference" src="./images/inference.png" width="800">

```
入力テキスト → LLM推論 → 中間層の隠れ状態を抽出 → 線形プローブ → 確信度スコア
```

トークンを生成しながらリアルタイムで確信度を計算し、Gradio UI 上で色分け表示します。

## アーキテクチャ

| コンポーネント | ファイル | 役割 |
|---|---|---|
| モデルロード | `src/models/model_loader.py` | HF モデル・トークナイザーの管理（M1/CUDA/CPU 自動選択） |
| 中間層抽出 | `src/models/hidden_extractor.py` | 指定層の隠れ状態を抽出・プーリング |
| 線形プローブ | `src/probes/linear_probe.py` | `Linear + Sigmoid` で確信度を出力（単一層 / 複数層統合） |
| 確信度スコア | `src/probes/confidence_scorer.py` | プローブの結果を解釈付きで返す |
| 学習 | `src/training/trainer.py` | プローブの学習・チェックポイント管理 |
| リアルタイム推論 | `src/inference/realtime_generator.py` | トークン生成と確信度計算をストリーミングで実行 |
| データセット | `src/data/dataset_generator.py` | 学習用データ（Q&A + 確信度ラベル）の生成 |
| Gradio UI | `experiments/demo/gradio_app.py` | 色分けテキスト・Plotly グラフのリアルタイム表示 |

## 技術スタック

- Python 3.9+
- PyTorch 2.0+
- Hugging Face Transformers
- scikit-learn
- matplotlib / seaborn / plotly（インタラクティブ可視化）
- MLflow（実験管理）
- Gradio（Web UI）

## インストール

### 1. リポジトリのクローン

```bash
git clone https://github.com/yourusername/brain-llm.git
cd brain-llm
```

### 2. 仮想環境の作成

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# または
venv\Scripts\activate  # Windows
```

### 3. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 4. 開発モードでインストール

```bash
pip install -e .
```

## クイックスタート

### ステップ 1: データセット生成

学習用の Q&A データを生成し、train/val/test に分割します（`src/data/dataset_generator.py`）。

```bash
# 英語データ（2,000サンプル）
python experiments/scripts/generate_large_dataset.py --num_samples 2000

# 日本語データ（2,000サンプル）
python experiments/scripts/generate_japanese_dataset.py --num_samples 2000
```

生成されるファイル:
- `data/processed/train.jsonl` / `val.jsonl` / `test.jsonl`（英語）
- `data/japanese/processed/train.jsonl` / `val.jsonl` / `test.jsonl`（日本語）

主なオプション（`generate_large_dataset.py` / `generate_japanese_dataset.py`）:

| オプション | 説明 | デフォルト |
|---|---|---|
| `--num_samples` | 生成するサンプル数 | `2000` |
| `--output_path` | 出力ファイルパス | `data/raw/large_dataset.jsonl` |
| `--high_ratio` | 高確信度サンプルの割合 | `0.3` |
| `--medium_ratio` | 中確信度サンプルの割合 | `0.4` |
| `--low_ratio` | 低確信度サンプルの割合 | `0.3` |
| `--train_ratio` | 学習データの割合 | `0.7` |
| `--val_ratio` | 検証データの割合 | `0.15` |
| `--test_ratio` | テストデータの割合 | `0.15` |
| `--seed` | 乱数シード | `42` |

> **注意: 現在のデータセットの制限**
>
> 現在の生成スクリプトは、**テンプレートのカテゴリ（一般常識 / 架空の情報など）をコードで分類し、カテゴリごとに決めた範囲内でラベル値をランダム生成**しています（例: 一般常識 → `uniform(0.85, 1.0)`、架空の情報 → `uniform(0.0, 0.25)`）。
> これはモデルが実際に「知っているか」を計測したものではないため、プローブは「架空っぽいキーワードが含まれているか」という表面的な文体パターンを学習している可能性があります。
>
> より厳密なデータセット生成の方法は検討が必要です。

### ステップ 2: プローブの学習

抽出した中間層の状態で線形プローブを学習します（`experiments/scripts/train_multi_layer_probe.py`）。

```bash
# 英語モデル（GPT-2）
python experiments/scripts/train_multi_layer_probe.py \
    --model_name gpt2 \
    --layers 0,6,11 \
    --aggregation weighted \
    --data_dir data/processed \
    --experiment_name gpt2_weighted

# 日本語モデル（rinna/japanese-gpt2-medium）
python experiments/scripts/train_multi_layer_probe.py \
    --model_name rinna/japanese-gpt2-medium \
    --layers 0,12,23 \
    --aggregation weighted \
    --data_dir data/japanese/processed \
    --experiment_name rinna_weighted

# 大きなモデルや epoch 数が多い場合はキャッシュを使うと高速化できます
python experiments/scripts/train_multi_layer_probe.py \
    --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --layers 0,11,21 \
    --aggregation weighted \
    --data_dir data/processed \
    --experiment_name tinyllama_weighted \
    --num_epochs 10 \
    --cache_hidden_states
```

学習済みチェックポイントは `results/experiments/<実験名>_<タイムスタンプ>/best_model.pt` に保存されます。

- `--experiment_name gpt2_weighted` を指定した場合: `results/experiments/gpt2_weighted_20250612_143022/`
- 省略した場合 (`train_multi_layer_probe.py`): `results/experiments/multi_layer_weighted_20250612_143022/`
- 省略した場合 (`train_probe.py`): `results/experiments/20250612_143022/`

主なオプション:

| オプション | 説明 | 例 |
|---|---|---|
| `--model_name` | HF モデル名 | `gpt2`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| `--layers` | 使用する層（カンマ区切り） | `0,6,11` |
| `--aggregation` | 複数層の統合方法 | `weighted` / `concat` / `mean` |
| `--pooling` | 隠れ状態のプーリング方法 | `last` / `mean` / `cls` |
| `--num_epochs` | 学習エポック数 | `20` |
| `--experiment_name` | 実験名（ディレクトリ名のプレフィックス） | `gpt2_weighted`, `tinyllama_jp` |
| `--cache_hidden_states` | 隠れ状態を事前計算してキャッシュ（大きなモデルで高速化） | フラグ |

### ステップ 3: Gradio デモの起動

学習済みプローブを使って確信度をリアルタイム可視化します（`experiments/demo/gradio_app.py`）。

```bash
# 英語モデル（GPT-2）
python experiments/demo/gradio_app.py \
    --model gpt2 \
    --checkpoint results/experiments/gpt2_weighted_20250612_143022/best_model.pt

# 日本語モデル（rinna）
python experiments/demo/gradio_app.py \
    --model rinna/japanese-gpt2-medium \
    --checkpoint results/experiments/rinna_weighted_20250612_143022/best_model.pt
```

ブラウザで http://localhost:7860 を開いて使用します。

主なオプション:

| オプション | 説明 | デフォルト |
|---|---|---|
| `--model` | HF モデル名 | `gpt2` |
| `--checkpoint` | 学習済みプローブのパス | `None`（ダミープローブ） |
| `--port` | ポート番号 | `7860` |
| `--share` | Gradio の公開リンクを作成 | フラグ |

### ステップ 4: 可視化レポートの生成（任意）

学習済みプローブの評価結果をインタラクティブな HTML レポートとして出力します。

```bash
# 複数層プローブの可視化
python experiments/scripts/visualize_multi_layer.py \
    --checkpoint results/experiments/gpt2_weighted_20250612_143022/best_model.pt \
    --output_dir results/experiments/gpt2_weighted_20250612_143022/visualizations

# インタラクティブ可視化
python experiments/scripts/visualize_interactive.py \
    --checkpoint results/experiments/gpt2_weighted_20250612_143022/best_model.pt \
    --output_dir results/experiments/gpt2_weighted_20250612_143022/visualizations
```

主なオプション（`visualize_multi_layer.py` / `visualize_interactive.py`）:

| オプション | 説明 | デフォルト |
|---|---|---|
| `--checkpoint` | 学習済みプローブのパス | 必須 |
| `--output_dir` | HTML の出力先ディレクトリ | `results/experiments` |
| `--model_name` | HF モデル名 | `gpt2` |
| `--layers` | 使用する層（カンマ区切り） | `0,6,11` |
| `--num_samples` | 可視化するサンプル数 | `100` |

生成される HTML（ブラウザで開いて確認）:

| ファイル | 内容 |
|---|---|
| `dashboard.html` | 全グラフをまとめたダッシュボード |
| `confidence_distribution.html` | 確信度スコアの分布 |
| `confidence_scatter.html` | 予測値 vs 正解の散布図 |
| `sample_predictions.html` | サンプルごとの予測結果一覧 |
| `error_analysis.html` | 誤分類サンプルの詳細 |

## プロジェクト構造

```
brain-llm/
├── src/
│   ├── models/          # モデル管理
│   ├── probes/          # プローブ実装
│   ├── data/            # データ処理
│   ├── training/        # 学習・評価
│   ├── visualization/   # 可視化
│   └── utils/           # ユーティリティ
├── experiments/
│   ├── demo/            # Gradio デモ
│   └── scripts/         # 実験スクリプト
├── data/
│   ├── raw/             # 英語生データ
│   ├── processed/       # 英語 train/val/test
│   └── japanese/        # 日本語データ
│       ├── raw/
│       └── processed/
├── results/
│   └── experiments/     # 学習済みプローブ・評価結果・可視化HTML
│       └── <experiment_name>_<timestamp>/   # 例: gpt2_weighted_20250612_143022
│           ├── best_model.pt
│           ├── config.json
│           ├── history.json
│           ├── test_metrics.json
│           ├── training_history.png
│           └── visualizations/  # インタラクティブHTML（任意）
├── tests/               # テストコード（未作成）
└── docs/                # ドキュメント（未作成）
```

## 対応モデル

`--model_name` / `--model` 引数を変えるだけで切り替えられます。

| モデル | 言語 | パラメータ | 層数 | hidden_dim | 推奨使用層 |
|---|---|---|---|---|---|
| `gpt2` | 英語 | 124M | 12 | 768 | `0,6,11` |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 多言語 | 1.1B | 22 | 2048 | `0,11,21` |
| `rinna/japanese-gpt2-medium` | 日本語 | 336M | 24 | 1024 | `0,12,23` |
| `cyberagent/open-calm-small` | 日本語 | 160M | 16 | 1024 | `0,8,15` |

## 確信度スコアをスクリプトから取得する

`src/probes/confidence_scorer.py` の `ConfidenceScorer` を使うと、任意のテキストに対して確信度スコアを数値で取得できます。

```python
from src.models.model_loader import ModelLoader
from src.models.hidden_extractor import HiddenStateExtractor
from src.probes.linear_probe import MultiLayerProbe
from src.probes.confidence_scorer import ConfidenceScorer
import torch

loader = ModelLoader("gpt2")
model, tokenizer = loader.load()
extractor = HiddenStateExtractor(model, tokenizer)

probe = MultiLayerProbe(input_dim=768, num_layers=3, aggregation="weighted")
checkpoint = torch.load("results/experiments/gpt2_weighted_20250612_143022/best_model.pt")
probe.load_state_dict(checkpoint["probe_state_dict"])

scorer = ConfidenceScorer(probe)
hidden_states = extractor.extract_and_pool("富士山の高さは？", layers=[0, 6, 11])
result = scorer.score_with_interpretation(hidden_states)

print(f"確信度: {result['confidence_score']:.3f}")
print(f"判定  : {result['confidence_level']}")   # 例: "おそらく知っている"
print(f"知識有: {result['knows']}")               # True / False
```

## モデルキャッシュの管理

`from_pretrained()` を初めて呼んだタイミングで、Hugging Face ライブラリが自動的にモデルをダウンロード・保存します。

**ゲートモデルの認証（Gemma、LLaMA など）**

一部のモデルは HuggingFace のモデルページでライセンスへの同意が必要です。同意後、以下のいずれかの方法で認証してください。コードの変更は不要です。

```bash
# 方法 1: 対話式ログイン（トークンが ~/.huggingface/token に保存される）
pip install huggingface_hub
huggingface-cli login

# 方法 2: 環境変数（~/.zshrc や ~/.bash_profile に追記）
export HF_TOKEN=hf_xxxxxxxxxxxx
```

`from_pretrained()` は `~/.huggingface/token` と環境変数 `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` を自動的に参照します。

**デフォルトの保存先**

```
~/.cache/huggingface/hub/
├── models--gpt2/                        # GPT-2（約 500MB）
├── models--rinna--japanese-gpt2-medium/ # rinna（約 1.4GB）
└── models--TinyLlama--...               # TinyLlama（約 4GB）
```

複数モデルを試すと数十GB になることがあります。

**保存先を変更する**

```bash
# ~/.zshrc や ~/.bash_profile に追記
export HF_HOME=~/your/preferred/path
```

**不要なモデルを削除する**

```bash
# 対話形式でキャッシュ一覧を確認しながら削除
huggingface-cli delete-cache

# 特定モデルをディレクトリごと削除
rm -rf ~/.cache/huggingface/hub/models--gpt2
```

## テスト

`tests/` ディレクトリにテストファイルを追加することで `pytest` で実行できます。現時点ではテストファイルは未作成です。

```bash
pytest tests/
pytest --cov=src tests/
```

## ライセンス

MIT License


## 参考文献

- Petroni et al. (2019) — "Language Models as Knowledge Bases?"
- Hewitt & Manning (2019) — "A Structural Probe for Finding Syntax"
- Clark et al. (2019) — "What Does BERT Look At?"
- Belinkov (2022) — "Probing Classifiers: Promises, Shortcomings, and Advances"

---

**注意**: このプロジェクトは研究目的で開発されています。本番環境での使用には追加の検証が必要です。
