# Gemini Prompt Optimizer (Streamlit)

9案 → ランキング → 4案 → ランキング → 1案（最終） という流れで、
画像生成プロンプトを絞り込む Streamlit アプリです。

## 1) セットアップ

```bash
python -m venv .venv
source .venv/bin/activate  # Windowsは .venv\Scripts\activate
pip install -r requirements.txt
```

## 2) APIキー

Google AI Studio の Gemini API Key を用意して、環境変数に設定します。

```bash
export GEMINI_API_KEY="YOUR_KEY"
# あるいは
export GOOGLE_API_KEY="YOUR_KEY"
```

アプリのサイドバーから直接入力もできます（ただし運用は secrets / env 推奨）。

## 3) 起動

```bash
streamlit run app.py
```

## 4) 使い方

1. 作りたい画像のイメージを入力
2. Round 1 を実行（9案 + 9枚）
3. ランキング（上ほど良い）
4. Round 2 を実行（4案 + 4枚）
5. ランキング
6. 最終生成

## メモ

- 画像生成モデル: `gemini-2.5-flash-image` (Nano Banana 🍌) をデフォルトにしています。
- 必要なら `gemini-3-pro-image-preview` に切り替え可能です（コスト/品質のバランスは用途次第）。
- 生成画像には SynthID ウォーターマークが付与される場合があります（モデル仕様）。
