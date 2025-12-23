import io
import json
import os
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import streamlit as st
from PIL import Image

# Gemini SDK
from google import genai
from google.genai import types

# Optional: drag & drop ranking
try:
    from streamlit_sortables import sort_items  # pip install streamlit-sortables
    _HAS_SORTABLES = True
except Exception:
    _HAS_SORTABLES = False


# -----------------------------
# Data models (lightweight)
# -----------------------------
@dataclass
class PromptCandidate:
    id: int
    label: str
    prompt: str


# -----------------------------
# Utilities
# -----------------------------
def _truncate(s: str, n: int = 60) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _ensure_state() -> None:
    """Initialize session_state keys."""
    defaults = {
        "user_intent": "",
        "must_include": "",
        "must_avoid": "",
        "extra_feedback": "",
        "settings": {
            "text_model": "gemini-3-flash-preview",
            "image_model": "gemini-2.5-flash-image",
            "aspect_ratio": "1:1",
            "prompt_language": "日本語",
            "temperature_round1": 0.9,
            "temperature_round2": 0.7,
            "temperature_round3": 0.5,
            "use_multimodal_feedback": False,
        },
        "round1": {
            "candidates": [],  # List[PromptCandidate as dict]
            "images": {},      # id -> bytes (png)
            "errors": {},      # id -> str
            "ranking": [],     # List[int]
        },
        "round2": {
            "candidates": [],
            "images": {},
            "errors": {},
            "ranking": [],
        },
        "final": {
            "candidate": None,  # PromptCandidate as dict
            "image": None,      # bytes
            "error": None,      # str
        },
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_all() -> None:
    for k in ["round1", "round2", "final"]:
        if k in st.session_state:
            del st.session_state[k]
    _ensure_state()


def _get_api_key_from_env_or_ui(ui_key: str) -> Optional[str]:
    if ui_key and ui_key.strip():
        return ui_key.strip()
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


@st.cache_resource(show_spinner=False)
def _get_client(api_key: str) -> genai.Client:
    # Gemini Developer API (Google AI Studio key)
    return genai.Client(api_key=api_key)


def _json_schema_for_prompts(n: int) -> Dict[str, Any]:
    # Standard JSON Schema subset supported by Gemini structured output mode.
    return {
        "type": "object",
        "properties": {
            "prompts": {
                "type": "array",
                "minItems": n,
                "maxItems": n,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "minimum": 1},
                        "label": {"type": "string", "minLength": 1},
                        "prompt": {"type": "string", "minLength": 1},
                    },
                    "required": ["id", "label", "prompt"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["prompts"],
        "additionalProperties": False,
    }


def _json_schema_for_final_prompt() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "minimum": 1},
            "label": {"type": "string", "minLength": 1},
            "prompt": {"type": "string", "minLength": 1},
        },
        "required": ["id", "label", "prompt"],
        "additionalProperties": False,
    }


def _build_system_instruction(language: str) -> str:
    lang_hint = "日本語" if language == "日本語" else "English"
    return textwrap.dedent(
        f'''
        You are a senior prompt engineer for image generation.
        Produce high-quality prompts that work well for Gemini 2.5 Flash Image (Nano Banana).
        Write prompts in {lang_hint}.
        Each prompt must be self-contained and describe:
        - subject, environment, composition, lighting, style/medium
        - optionally camera details (lens, angle) when relevant
        - clear constraints and what to avoid (no watermarks, no unreadable text, etc.) when useful
        Avoid referencing copyrighted characters, trademarks, living artists, or brand names unless the user explicitly requests it.
        Do NOT output markdown. Only output valid JSON that matches the provided schema.
        '''
    ).strip()


def _call_text_model_for_prompts(
    client: genai.Client,
    model: str,
    schema: Dict[str, Any],
    system_instruction: str,
    user_prompt: Any,
    temperature: float,
) -> Dict[str, Any]:
    config = {
        "response_mime_type": "application/json",
        "response_json_schema": schema,
        "system_instruction": system_instruction,
        "temperature": float(temperature),
        "max_output_tokens": 4096,
    }
    resp = client.models.generate_content(model=model, contents=user_prompt, config=config)

    # Prefer native parsed object (SDK convenience)
    if getattr(resp, "parsed", None) is not None:
        return resp.parsed  # type: ignore[return-value]

    # Fallback: parse text manually
    try:
        return json.loads(resp.text)
    except Exception as e:
        raise ValueError(f"JSON parse failed. Raw text: {resp.text[:500]}") from e


def _candidates_from_payload(payload: Dict[str, Any]) -> List[PromptCandidate]:
    raw = payload.get("prompts", [])
    raw_sorted = sorted(raw, key=lambda x: int(x.get("id", 0)))
    out: List[PromptCandidate] = []
    for item in raw_sorted:
        out.append(
            PromptCandidate(
                id=int(item["id"]),
                label=str(item["label"]).strip(),
                prompt=str(item["prompt"]).strip(),
            )
        )
    return out


def _final_candidate_from_payload(payload: Dict[str, Any]) -> PromptCandidate:
    return PromptCandidate(
        id=int(payload["id"]),
        label=str(payload["label"]).strip(),
        prompt=str(payload["prompt"]).strip(),
    )


def _generate_image_bytes(
    client: genai.Client,
    model: str,
    prompt: str,
    aspect_ratio: str,
) -> bytes:
    """Generate a single PNG image as bytes using Gemini image model."""
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
        ),
    )

    for part in resp.parts:
        if getattr(part, "inline_data", None):
            img: Image.Image = part.as_image()
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

    raise RuntimeError("Model response did not include an image.")


def _render_gallery(candidates: List[Dict[str, Any]], images: Dict[str, bytes], cols: int = 3) -> None:
    if not candidates:
        return
    cols = max(1, min(cols, 5))
    rows = (len(candidates) + cols - 1) // cols

    idx = 0
    for _ in range(rows):
        cs = st.columns(cols)
        for c in cs:
            if idx >= len(candidates):
                break
            cand = candidates[idx]
            cid = str(cand["id"])
            with c:
                st.markdown(f"### #{cid} — {_truncate(cand['label'], 40)}")
                if cid in images:
                    st.image(images[cid], use_container_width=True)
                else:
                    st.info("まだ画像がありません。")
                with st.expander("プロンプトを見る"):
                    st.code(cand["prompt"], language="text")
            idx += 1


def _rank_ui(
    title: str,
    candidates: List[Dict[str, Any]],
    default_ranking: Optional[List[int]] = None,
) -> Optional[List[int]]:
    if not candidates:
        return None

    st.subheader(title)
    st.caption("上ほど『想像に近い(=良い)』です。")

    ids = [int(c["id"]) for c in candidates]
    labels = {int(c["id"]): c["label"] for c in candidates}

    methods = ["順位を数字で入力"]
    if _HAS_SORTABLES:
        methods.insert(0, "ドラッグ&ドロップ (おすすめ)")

    method = st.radio("ランキング方法", methods, horizontal=True, key=f"rank_method_{title}")

    if method.startswith("ドラッグ") and _HAS_SORTABLES:
        items: List[str] = []
        if default_ranking and len(default_ranking) == len(ids):
            for i in default_ranking:
                items.append(f"{i}: {_truncate(labels[i], 60)}")
        else:
            for i in ids:
                items.append(f"{i}: {_truncate(labels[i], 60)}")

        custom_style = '''
        .sortable-container { counter-reset: item; }
        .sortable-item::before { content: counter(item) ". "; counter-increment: item; }
        '''
        sorted_items = sort_items(items, custom_style=custom_style)
        ranking = [int(x.split(":", 1)[0].strip()) for x in sorted_items]
        st.write("現在の順番:", ranking)
        return ranking

    import pandas as pd

    if default_ranking and len(default_ranking) == len(ids):
        initial_rank = {cid: i + 1 for i, cid in enumerate(default_ranking)}
    else:
        initial_rank = {cid: i + 1 for i, cid in enumerate(ids)}

    df = pd.DataFrame(
        [{"id": cid, "label": _truncate(labels[cid], 80), "rank(1=best)": initial_rank[cid]} for cid in ids]
    )
    edited = st.data_editor(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "id": st.column_config.NumberColumn(disabled=True),
            "label": st.column_config.TextColumn(disabled=True),
            "rank(1=best)": st.column_config.NumberColumn(min_value=1, max_value=len(ids), step=1),
        },
        key=f"rank_editor_{title}",
    )

    ranks = list(edited["rank(1=best)"])
    if len(set(ranks)) != len(ranks):
        st.warning("rank が重複しています。1〜N をユニークにしてください。")
        return None
    if any((r < 1 or r > len(ids)) for r in ranks):
        st.warning("rank の範囲が不正です。")
        return None

    edited_sorted = edited.sort_values("rank(1=best)")
    ranking = [int(x) for x in edited_sorted["id"].tolist()]
    st.write("現在の順番:", ranking)
    return ranking


def _build_round1_user_prompt(user_intent: str, must_include: str, must_avoid: str, language: str) -> str:
    return textwrap.dedent(
        f'''
        The user wants prompts for image generation.

        User intent:
        {user_intent.strip()}

        Must include (if any):
        {must_include.strip() or "None"}

        Must avoid (if any):
        {must_avoid.strip() or "None"}

        Task:
        Create 9 distinct image-generation prompts exploring different directions (composition, style, lighting, perspective)
        while staying faithful to the user intent.

        Constraints:
        - Keep each prompt concise (1-3 sentences), but concrete.
        - Prefer scene descriptions over keyword salad.
        - If you include “avoid/negative” instructions, embed them naturally at the end.
        - Do not mention any model names.
        - Output strictly JSON matching the schema.
        '''
    ).strip()


def _build_round2_user_prompt(
    user_intent: str,
    must_include: str,
    must_avoid: str,
    extra_feedback: str,
    round1_candidates: List[Dict[str, Any]],
    round1_ranking: List[int],
) -> str:
    lines = []
    for c in round1_candidates:
        cid = int(c["id"])
        rank = round1_ranking.index(cid) + 1 if cid in round1_ranking else None
        lines.append(f"- id={cid}, rank={rank}, label={c['label']}\n  prompt={c['prompt']}")
    blob = "\n".join(lines)

    return textwrap.dedent(
        f'''
        We are iteratively optimizing an image generation prompt.

        User intent:
        {user_intent.strip()}

        Must include (if any):
        {must_include.strip() or "None"}

        Must avoid (if any):
        {must_avoid.strip() or "None"}

        Additional user feedback (optional):
        {extra_feedback.strip() or "None"}

        Round 1 results:
        The user ranked the candidates from best (rank=1) to worst.

        Candidates:
        {blob}

        Task:
        Create 4 new prompts that move closer to what the user prefers.
        - Preserve the strongest qualities of the top-ranked prompts.
        - Avoid traits likely responsible for the bottom-ranked prompts being worse.
        - Keep prompts diverse but clearly improved vs round 1.
        - Output strictly JSON matching the schema.
        '''
    ).strip()


def _build_round3_user_prompt(
    user_intent: str,
    must_include: str,
    must_avoid: str,
    extra_feedback: str,
    round2_candidates: List[Dict[str, Any]],
    round2_ranking: List[int],
) -> str:
    lines = []
    for c in round2_candidates:
        cid = int(c["id"])
        rank = round2_ranking.index(cid) + 1 if cid in round2_ranking else None
        lines.append(f"- id={cid}, rank={rank}, label={c['label']}\n  prompt={c['prompt']}")
    blob = "\n".join(lines)

    return textwrap.dedent(
        f'''
        We are finishing an iterative prompt optimization process.

        User intent:
        {user_intent.strip()}

        Must include (if any):
        {must_include.strip() or "None"}

        Must avoid (if any):
        {must_avoid.strip() or "None"}

        Additional user feedback (optional):
        {extra_feedback.strip() or "None"}

        Round 2 results:
        The user ranked 4 candidates from best (rank=1) to worst.

        Candidates:
        {blob}

        Task:
        Produce ONE final best prompt that most likely matches the user's preference.
        - Make it specific and unambiguous.
        - Keep it concise (1-3 sentences).
        - If useful, include a short “avoid” clause at the end.
        - Output strictly JSON matching the schema.
        '''
    ).strip()


def main() -> None:
    st.set_page_config(page_title="Gemini Prompt Optimizer", layout="wide")
    _ensure_state()

    st.title("🧪 画像生成プロンプト最適化 (Gemini + Streamlit)")
    st.caption("9案 → ランキング → 4案 → ランキング → 1案（最終） という流れでプロンプトを絞り込みます。🍌")

    with st.sidebar:
        st.header("設定")
        api_key_ui = st.text_input("Gemini API Key (GEMINI_API_KEY)", type="password")
        api_key = _get_api_key_from_env_or_ui(api_key_ui)

        st.markdown("---")
        st.subheader("モデル")
        st.session_state["settings"]["text_model"] = st.selectbox(
            "テキスト生成モデル (プロンプト作成)",
            options=["gemini-3-flash-preview", "gemini-3-pro-preview"],
            index=0 if st.session_state["settings"]["text_model"] == "gemini-3-flash-preview" else 1,
        )
        st.session_state["settings"]["image_model"] = st.selectbox(
            "画像生成モデル",
            options=["gemini-2.5-flash-image", "gemini-3-pro-image-preview"],
            index=0 if st.session_state["settings"]["image_model"] == "gemini-2.5-flash-image" else 1,
        )

        st.markdown("---")
        st.subheader("出力")
        st.session_state["settings"]["aspect_ratio"] = st.selectbox(
            "アスペクト比",
            options=["1:1", "16:9", "9:16", "4:3", "3:4"],
            index=["1:1", "16:9", "9:16", "4:3", "3:4"].index(st.session_state["settings"]["aspect_ratio"]),
        )
        st.session_state["settings"]["prompt_language"] = st.selectbox(
            "プロンプト言語",
            options=["日本語", "English"],
            index=0 if st.session_state["settings"]["prompt_language"] == "日本語" else 1,
        )

        st.markdown("---")
        st.subheader("温度 (多様性)")
        st.session_state["settings"]["temperature_round1"] = st.slider("Round 1 (9案)", 0.0, 1.5, float(st.session_state["settings"]["temperature_round1"]), 0.05)
        st.session_state["settings"]["temperature_round2"] = st.slider("Round 2 (4案)", 0.0, 1.5, float(st.session_state["settings"]["temperature_round2"]), 0.05)
        st.session_state["settings"]["temperature_round3"] = st.slider("Round 3 (最終)", 0.0, 1.5, float(st.session_state["settings"]["temperature_round3"]), 0.05)

        st.markdown("---")
        st.session_state["settings"]["use_multimodal_feedback"] = st.toggle(
            "ランキング生成時に画像も渡す (精度↑/コスト↑)",
            value=bool(st.session_state["settings"]["use_multimodal_feedback"]),
            help="Round 2 / Final のプロンプト作成時、上位/下位の画像を Gemini 3 に渡して分析させます。",
        )

        st.markdown("---")
        if st.button("🧹 全リセット"):
            _reset_all()
            st.rerun()

        st.caption(
            "APIキーは環境変数 GEMINI_API_KEY / GOOGLE_API_KEY でもOK。\n"
            "Streamlit Cloudなら st.secrets で管理推奨。"
        )

    st.header("① 作りたい画像のイメージ")
    st.session_state["user_intent"] = st.text_area(
        "どんな画像を作りたい？（例：『雨の夜の東京、ネオンが反射する路地、シネマティックな写真』）",
        value=st.session_state["user_intent"],
        height=120,
    )

    cols = st.columns(2)
    with cols[0]:
        st.session_state["must_include"] = st.text_input("必ず入れて欲しい要素（任意）", value=st.session_state["must_include"])
    with cols[1]:
        st.session_state["must_avoid"] = st.text_input("避けたい要素（任意）", value=st.session_state["must_avoid"])

    st.session_state["extra_feedback"] = st.text_area(
        "補足フィードバック（任意：色味、雰囲気、構図、画風など）",
        value=st.session_state["extra_feedback"],
        height=80,
    )

    if not st.session_state["user_intent"].strip():
        st.warning("まずは『作りたい画像のイメージ』を入力してください。")
        return

    if not api_key:
        st.error("Gemini APIキーが見つかりません。サイドバーに入力するか、環境変数 GEMINI_API_KEY を設定してください。")
        return

    client = _get_client(api_key)

    # Round 1
    st.header("②③ Round 1 — 9案生成 → 画像生成")
    r1 = st.session_state["round1"]

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("🚀 Round 1 を実行 (9案 + 画像9枚)", disabled=bool(r1["candidates"])):
            with st.spinner("Round 1: 9案のプロンプトを生成中…"):
                sys_inst = _build_system_instruction(st.session_state["settings"]["prompt_language"])
                schema = _json_schema_for_prompts(9)
                user_prompt = _build_round1_user_prompt(
                    st.session_state["user_intent"],
                    st.session_state["must_include"],
                    st.session_state["must_avoid"],
                    st.session_state["settings"]["prompt_language"],
                )
                payload = _call_text_model_for_prompts(
                    client=client,
                    model=st.session_state["settings"]["text_model"],
                    schema=schema,
                    system_instruction=sys_inst,
                    user_prompt=user_prompt,
                    temperature=st.session_state["settings"]["temperature_round1"],
                )
                candidates = _candidates_from_payload(payload)
                # Force stable ids 1..9 to avoid duplicates from the model
                for i, c in enumerate(candidates, start=1):
                    c.id = i
                r1["candidates"] = [c.__dict__ for c in candidates]
                r1["images"] = {}
                r1["errors"] = {}
                r1["ranking"] = []

            with st.spinner("Round 1: 画像を生成中…（9枚）"):
                prog = st.progress(0.0)
                for i, cand in enumerate(r1["candidates"]):
                    cid = str(cand["id"])
                    try:
                        img_bytes = _generate_image_bytes(
                            client=client,
                            model=st.session_state["settings"]["image_model"],
                            prompt=cand["prompt"],
                            aspect_ratio=st.session_state["settings"]["aspect_ratio"],
                        )
                        r1["images"][cid] = img_bytes
                    except Exception as e:
                        r1["errors"][cid] = str(e)
                    prog.progress((i + 1) / max(1, len(r1["candidates"])))
                prog.empty()

            st.success("Round 1 完了。下でランキングしてください。")
            st.rerun()

    with colB:
        if r1["candidates"] and st.button("♻️ Round 1 画像を再生成 (失敗分のみ)"):
            with st.spinner("失敗分のみ再生成中…"):
                missing = [c for c in r1["candidates"] if str(c["id"]) not in r1["images"]]
                prog = st.progress(0.0)
                for i, cand in enumerate(missing):
                    cid = str(cand["id"])
                    try:
                        img_bytes = _generate_image_bytes(
                            client=client,
                            model=st.session_state["settings"]["image_model"],
                            prompt=cand["prompt"],
                            aspect_ratio=st.session_state["settings"]["aspect_ratio"],
                        )
                        r1["images"][cid] = img_bytes
                        r1["errors"].pop(cid, None)
                    except Exception as e:
                        r1["errors"][cid] = str(e)
                    prog.progress((i + 1) / max(1, len(missing)))
                prog.empty()
            st.rerun()

    if r1["candidates"]:
        _render_gallery(r1["candidates"], r1["images"], cols=3)

        if r1["errors"]:
            st.warning("一部の画像生成に失敗しました。必要なら『失敗分のみ再生成』を押してください。")
            with st.expander("エラー詳細"):
                st.json(r1["errors"])

        st.header("④ Round 1 — ランキング")
        ranking1 = _rank_ui("Round 1 ランキング", r1["candidates"], default_ranking=r1["ranking"] if r1["ranking"] else None)
        if ranking1 and st.button("✅ Round 1 ランキング確定"):
            r1["ranking"] = ranking1
            st.success("Round 1 ランキングを保存しました。次に Round 2 を実行できます。")
            st.rerun()

    # Round 2
    st.header("⑤③ Round 2 — 4案生成 → 画像生成")
    r2 = st.session_state["round2"]

    if not r1["ranking"]:
        st.info("Round 2 に進むには、まず Round 1 のランキングを確定してください。")
        return

    colC, colD = st.columns([1, 1])
    with colC:
        if st.button("🚀 Round 2 を実行 (4案 + 画像4枚)", disabled=bool(r2["candidates"])):
            with st.spinner("Round 2: 4案のプロンプトを生成中…"):
                sys_inst = _build_system_instruction(st.session_state["settings"]["prompt_language"])
                schema = _json_schema_for_prompts(4)

                use_imgs = bool(st.session_state["settings"]["use_multimodal_feedback"])
                if use_imgs:
                    top_id = str(r1["ranking"][0])
                    bottom_id = str(r1["ranking"][-1])

                    contents: List[Any] = []
                    contents.append(
                        _build_round2_user_prompt(
                            st.session_state["user_intent"],
                            st.session_state["must_include"],
                            st.session_state["must_avoid"],
                            st.session_state["extra_feedback"],
                            r1["candidates"],
                            r1["ranking"],
                        )
                    )
                    if top_id in r1["images"]:
                        contents.append("Best-ranked image (rank=1):")
                        contents.append(types.Part.from_bytes(data=r1["images"][top_id], mime_type="image/png"))
                    if bottom_id in r1["images"]:
                        contents.append("Worst-ranked image (last rank):")
                        contents.append(types.Part.from_bytes(data=r1["images"][bottom_id], mime_type="image/png"))

                    user_prompt_any: Any = contents
                else:
                    user_prompt_any = _build_round2_user_prompt(
                        st.session_state["user_intent"],
                        st.session_state["must_include"],
                        st.session_state["must_avoid"],
                        st.session_state["extra_feedback"],
                        r1["candidates"],
                        r1["ranking"],
                    )

                payload = _call_text_model_for_prompts(
                    client=client,
                    model=st.session_state["settings"]["text_model"],
                    schema=schema,
                    system_instruction=sys_inst,
                    user_prompt=user_prompt_any,
                    temperature=st.session_state["settings"]["temperature_round2"],
                )
                candidates = _candidates_from_payload(payload)

                # Re-id them as 1..4 for UI clarity
                for i, c in enumerate(candidates, start=1):
                    c.id = i

                r2["candidates"] = [c.__dict__ for c in candidates]
                r2["images"] = {}
                r2["errors"] = {}
                r2["ranking"] = []

            with st.spinner("Round 2: 画像を生成中…（4枚）"):
                prog = st.progress(0.0)
                for i, cand in enumerate(r2["candidates"]):
                    cid = str(cand["id"])
                    try:
                        img_bytes = _generate_image_bytes(
                            client=client,
                            model=st.session_state["settings"]["image_model"],
                            prompt=cand["prompt"],
                            aspect_ratio=st.session_state["settings"]["aspect_ratio"],
                        )
                        r2["images"][cid] = img_bytes
                    except Exception as e:
                        r2["errors"][cid] = str(e)
                    prog.progress((i + 1) / max(1, len(r2["candidates"])))
                prog.empty()

            st.success("Round 2 完了。下でランキングしてください。")
            st.rerun()

    with colD:
        if r2["candidates"] and st.button("♻️ Round 2 画像を再生成 (失敗分のみ)"):
            with st.spinner("失敗分のみ再生成中…"):
                missing = [c for c in r2["candidates"] if str(c["id"]) not in r2["images"]]
                prog = st.progress(0.0)
                for i, cand in enumerate(missing):
                    cid = str(cand["id"])
                    try:
                        img_bytes = _generate_image_bytes(
                            client=client,
                            model=st.session_state["settings"]["image_model"],
                            prompt=cand["prompt"],
                            aspect_ratio=st.session_state["settings"]["aspect_ratio"],
                        )
                        r2["images"][cid] = img_bytes
                        r2["errors"].pop(cid, None)
                    except Exception as e:
                        r2["errors"][cid] = str(e)
                    prog.progress((i + 1) / max(1, len(missing)))
                prog.empty()
            st.rerun()

    if r2["candidates"]:
        _render_gallery(r2["candidates"], r2["images"], cols=2)

        if r2["errors"]:
            st.warning("一部の画像生成に失敗しました。必要なら『失敗分のみ再生成』を押してください。")
            with st.expander("エラー詳細"):
                st.json(r2["errors"])

        st.header("⑥ Round 2 — ランキング")
        ranking2 = _rank_ui("Round 2 ランキング", r2["candidates"], default_ranking=r2["ranking"] if r2["ranking"] else None)
        if ranking2 and st.button("✅ Round 2 ランキング確定"):
            r2["ranking"] = ranking2
            st.success("Round 2 ランキングを保存しました。次に最終生成へ進めます。")
            st.rerun()

    # Final
    st.header("⑦⑧ Final — 1案生成 → 画像生成 → 表示")

    if not r2["ranking"]:
        st.info("最終生成に進むには、Round 2 のランキングを確定してください。")
        return

    if st.button("🏁 最終プロンプト + 画像を生成", disabled=bool(st.session_state["final"]["candidate"])):
        with st.spinner("最終プロンプトを生成中…"):
            sys_inst = _build_system_instruction(st.session_state["settings"]["prompt_language"])
            schema = _json_schema_for_final_prompt()

            use_imgs = bool(st.session_state["settings"]["use_multimodal_feedback"])
            if use_imgs:
                top_id = str(r2["ranking"][0])
                bottom_id = str(r2["ranking"][-1])
                contents: List[Any] = []
                contents.append(
                    _build_round3_user_prompt(
                        st.session_state["user_intent"],
                        st.session_state["must_include"],
                        st.session_state["must_avoid"],
                        st.session_state["extra_feedback"],
                        r2["candidates"],
                        r2["ranking"],
                    )
                )
                if top_id in r2["images"]:
                    contents.append("Best-ranked image (rank=1):")
                    contents.append(types.Part.from_bytes(data=r2["images"][top_id], mime_type="image/png"))
                if bottom_id in r2["images"]:
                    contents.append("Worst-ranked image (last rank):")
                    contents.append(types.Part.from_bytes(data=r2["images"][bottom_id], mime_type="image/png"))
                user_prompt_any: Any = contents
            else:
                user_prompt_any = _build_round3_user_prompt(
                    st.session_state["user_intent"],
                    st.session_state["must_include"],
                    st.session_state["must_avoid"],
                    st.session_state["extra_feedback"],
                    r2["candidates"],
                    r2["ranking"],
                )

            payload = _call_text_model_for_prompts(
                client=client,
                model=st.session_state["settings"]["text_model"],
                schema=schema,
                system_instruction=sys_inst,
                user_prompt=user_prompt_any,
                temperature=st.session_state["settings"]["temperature_round3"],
            )
            final_cand = _final_candidate_from_payload(payload)
            final_cand.id = 1
            st.session_state["final"]["candidate"] = final_cand.__dict__
            st.session_state["final"]["image"] = None
            st.session_state["final"]["error"] = None

        with st.spinner("最終画像を生成中…"):
            try:
                img_bytes = _generate_image_bytes(
                    client=client,
                    model=st.session_state["settings"]["image_model"],
                    prompt=st.session_state["final"]["candidate"]["prompt"],
                    aspect_ratio=st.session_state["settings"]["aspect_ratio"],
                )
                st.session_state["final"]["image"] = img_bytes
            except Exception as e:
                st.session_state["final"]["error"] = str(e)

        st.rerun()

    final = st.session_state["final"]
    if final["candidate"]:
        st.subheader("最終結果")
        st.markdown(f"### ✅ {_truncate(final['candidate']['label'], 80)}")
        if final["image"]:
            st.image(final["image"], use_container_width=True)
        else:
            st.error("画像生成に失敗しました。")
            if final["error"]:
                st.code(final["error"])

        st.markdown("#### 最終プロンプト")
        st.code(final["candidate"]["prompt"], language="text")

        st.download_button(
            "📥 プロンプトをダウンロード (.txt)",
            data=final["candidate"]["prompt"].encode("utf-8"),
            file_name="final_prompt.txt",
            mime="text/plain",
        )
        if final["image"]:
            st.download_button(
                "📥 画像をダウンロード (.png)",
                data=final["image"],
                file_name="final_image.png",
                mime="image/png",
            )


if __name__ == "__main__":
    main()
