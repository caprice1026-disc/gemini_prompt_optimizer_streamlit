from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st
from google.genai import types

from src.gemini_api import (
    call_text_model_for_prompts,
    generate_image_bytes,
    get_api_key_from_env_or_ui,
    get_client,
)
from src.models import PromptCandidate
from src.prompts import (
    build_round1_user_prompt,
    build_round2_user_prompt,
    build_round3_user_prompt,
    build_system_instruction,
    json_schema_for_final_prompt,
    json_schema_for_prompts,
)
from src.runtime import is_running_with_streamlit
from src.state import ensure_state, reset_all
from src.ui_components import render_gallery, rank_ui, truncate


def candidates_from_payload(payload: Dict[str, Any]) -> List[PromptCandidate]:
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


def final_candidate_from_payload(payload: Dict[str, Any]) -> PromptCandidate:
    return PromptCandidate(
        id=int(payload["id"]),
        label=str(payload["label"]).strip(),
        prompt=str(payload["prompt"]).strip(),
    )


def render_app() -> None:
    st.set_page_config(page_title="Gemini Prompt Optimizer", layout="wide")
    ensure_state()

    st.title("🧪 画像生成プロンプト最適化 (Gemini + Streamlit)")
    st.caption("9件→ランキング → 4件→ランキング → 1案（最終）という流れでプロンプトを絞り込みます。🍃")

    with st.sidebar:
        st.header("設定")
        api_key_ui = st.text_input("Gemini API Key (GEMINI_API_KEY)", type="password")
        api_key = get_api_key_from_env_or_ui(api_key_ui)

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
        st.session_state["settings"]["temperature_round1"] = st.slider(
            "Round 1 (9件)", 0.0, 1.5, float(st.session_state["settings"]["temperature_round1"]), 0.05
        )
        st.session_state["settings"]["temperature_round2"] = st.slider(
            "Round 2 (4件)", 0.0, 1.5, float(st.session_state["settings"]["temperature_round2"]), 0.05
        )
        st.session_state["settings"]["temperature_round3"] = st.slider(
            "Round 3 (最終)", 0.0, 1.5, float(st.session_state["settings"]["temperature_round3"]), 0.05
        )

        st.markdown("---")
        st.session_state["settings"]["use_multimodal_feedback"] = st.toggle(
            "ランキング生成時に画像も渡す (精度↑/コスト↑)",
            value=bool(st.session_state["settings"]["use_multimodal_feedback"]),
            help="Round 2 / Final のプロンプト作成時、上位・下位の画像を Gemini に渡して改善させます。",
        )

        st.markdown("---")
        if st.button("🧹 全リセット"):
            reset_all()
            st.rerun()

        st.caption(
            "APIキーは環境変数 GEMINI_API_KEY / GOOGLE_API_KEY でもOK。\n"
            "Streamlit Cloudなら st.secrets で管理推奨。"
        )

    st.header("① 作りたい画像のイメージ")
    st.session_state["user_intent"] = st.text_area(
        "どんな画像を作りたいですか？（例：『雨の夜の東京、ネオンが反射する路地、シネマティックな写真』）",
        value=st.session_state["user_intent"],
        height=120,
    )

    cols = st.columns(2)
    with cols[0]:
        st.session_state["must_include"] = st.text_input(
            "必ず入れてほしい要素（任意）",
            value=st.session_state["must_include"],
        )
    with cols[1]:
        st.session_state["must_avoid"] = st.text_input(
            "避けたい要素（任意）",
            value=st.session_state["must_avoid"],
        )

    st.session_state["extra_feedback"] = st.text_area(
        "補足フィードバック（任意）：色味、雰囲気、構図、画風など",
        value=st.session_state["extra_feedback"],
        height=80,
    )

    if not st.session_state["user_intent"].strip():
        st.warning("まず『作りたい画像のイメージ』を入力してください。")
        return

    if not api_key:
        st.error("Gemini APIキーが見つかりません。サイドバーに入力するか、環境変数 GEMINI_API_KEY を設定してください。")
        return

    client = get_client(api_key)

    # Round 1
    st.header("②③ Round 1 — 9案生成 → 画像生成")
    r1 = st.session_state["round1"]

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("🚀 Round 1 を実行 (9案 + 画像生成)", disabled=bool(r1["candidates"])):
            with st.spinner("Round 1: 9案のプロンプトを生成中…"):
                sys_inst = build_system_instruction(st.session_state["settings"]["prompt_language"])
                schema = json_schema_for_prompts(9)
                user_prompt = build_round1_user_prompt(
                    st.session_state["user_intent"],
                    st.session_state["must_include"],
                    st.session_state["must_avoid"],
                    st.session_state["settings"]["prompt_language"],
                )
                payload = call_text_model_for_prompts(
                    client=client,
                    model=st.session_state["settings"]["text_model"],
                    schema=schema,
                    system_instruction=sys_inst,
                    user_prompt=user_prompt,
                    temperature=st.session_state["settings"]["temperature_round1"],
                )
                candidates = candidates_from_payload(payload)
                # Force stable ids 1..9 to avoid duplicates from the model
                for i, c in enumerate(candidates, start=1):
                    c.id = i
                r1["candidates"] = [c.__dict__ for c in candidates]
                r1["images"] = {}
                r1["errors"] = {}
                r1["ranking"] = []

            with st.spinner("Round 1: 画像を生成中…(9枚)"):
                prog = st.progress(0.0)
                for i, cand in enumerate(r1["candidates"]):
                    cid = str(cand["id"])
                    try:
                        img_bytes = generate_image_bytes(
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

    with col_b:
        if r1["candidates"] and st.button("♻️ Round 1 画像を再生成 (失敗分のみ)"):
            with st.spinner("失敗分のみ再生成中…"):
                missing = [c for c in r1["candidates"] if str(c["id"]) not in r1["images"]]
                prog = st.progress(0.0)
                for i, cand in enumerate(missing):
                    cid = str(cand["id"])
                    try:
                        img_bytes = generate_image_bytes(
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
        render_gallery(r1["candidates"], r1["images"], cols=3)

        if r1["errors"]:
            st.warning("一部の画像生成に失敗しました。必要なら『失敗分のみ再生成』を押してください。")
            with st.expander("エラー詳細"):
                st.json(r1["errors"])

        st.header("④ Round 1 — ランキング")
        ranking1 = rank_ui("Round 1 ランキング", r1["candidates"], default_ranking=r1["ranking"] or None)
        if ranking1 and st.button("✅ Round 1 ランキング確定"):
            r1["ranking"] = ranking1
            st.success("Round 1 ランキングを保存しました。次に Round 2 を実行できます。")
            st.rerun()

    # Round 2
    st.header("⑤⑥ Round 2 — 4案生成 → 画像生成")
    r2 = st.session_state["round2"]

    if not r1["ranking"]:
        st.info("Round 2 に進むには、まず Round 1 のランキングを確定してください。")
        return

    col_c, col_d = st.columns([1, 1])
    with col_c:
        if st.button("🚀 Round 2 を実行 (4案 + 画像生成)", disabled=bool(r2["candidates"])):
            with st.spinner("Round 2: 4案のプロンプトを生成中…"):
                sys_inst = build_system_instruction(st.session_state["settings"]["prompt_language"])
                schema = json_schema_for_prompts(4)

                use_imgs = bool(st.session_state["settings"]["use_multimodal_feedback"])
                if use_imgs:
                    top_id = str(r1["ranking"][0])
                    bottom_id = str(r1["ranking"][-1])

                    contents: List[Any] = []
                    contents.append(
                        build_round2_user_prompt(
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
                    user_prompt_any = build_round2_user_prompt(
                        st.session_state["user_intent"],
                        st.session_state["must_include"],
                        st.session_state["must_avoid"],
                        st.session_state["extra_feedback"],
                        r1["candidates"],
                        r1["ranking"],
                    )

                payload = call_text_model_for_prompts(
                    client=client,
                    model=st.session_state["settings"]["text_model"],
                    schema=schema,
                    system_instruction=sys_inst,
                    user_prompt=user_prompt_any,
                    temperature=st.session_state["settings"]["temperature_round2"],
                )
                candidates = candidates_from_payload(payload)

                # Re-id them as 1..4 for UI clarity
                for i, c in enumerate(candidates, start=1):
                    c.id = i

                r2["candidates"] = [c.__dict__ for c in candidates]
                r2["images"] = {}
                r2["errors"] = {}
                r2["ranking"] = []

            with st.spinner("Round 2: 画像を生成中…(4枚)"):
                prog = st.progress(0.0)
                for i, cand in enumerate(r2["candidates"]):
                    cid = str(cand["id"])
                    try:
                        img_bytes = generate_image_bytes(
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

    with col_d:
        if r2["candidates"] and st.button("♻️ Round 2 画像を再生成 (失敗分のみ)"):
            with st.spinner("失敗分のみ再生成中…"):
                missing = [c for c in r2["candidates"] if str(c["id"]) not in r2["images"]]
                prog = st.progress(0.0)
                for i, cand in enumerate(missing):
                    cid = str(cand["id"])
                    try:
                        img_bytes = generate_image_bytes(
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
        render_gallery(r2["candidates"], r2["images"], cols=2)

        if r2["errors"]:
            st.warning("一部の画像生成に失敗しました。必要なら『失敗分のみ再生成』を押してください。")
            with st.expander("エラー詳細"):
                st.json(r2["errors"])

        st.header("⑦ Round 2 — ランキング")
        ranking2 = rank_ui("Round 2 ランキング", r2["candidates"], default_ranking=r2["ranking"] or None)
        if ranking2 and st.button("✅ Round 2 ランキング確定"):
            r2["ranking"] = ranking2
            st.success("Round 2 ランキングを保存しました。次に最終生成へ進めます。")
            st.rerun()

    # Final
    st.header("⑧⑨ Final — 1案生成 → 画像生成 → 表示")

    if not r2["ranking"]:
        st.info("最終生成に進むには、Round 2 のランキングを確定してください。")
        return

    if st.button("🏁 最終プロンプト + 画像を生成", disabled=bool(st.session_state["final"]["candidate"])):
        with st.spinner("最終プロンプトを生成中…"):
            sys_inst = build_system_instruction(st.session_state["settings"]["prompt_language"])
            schema = json_schema_for_final_prompt()

            use_imgs = bool(st.session_state["settings"]["use_multimodal_feedback"])
            if use_imgs:
                top_id = str(r2["ranking"][0])
                bottom_id = str(r2["ranking"][-1])
                contents: List[Any] = []
                contents.append(
                    build_round3_user_prompt(
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
                user_prompt_any = build_round3_user_prompt(
                    st.session_state["user_intent"],
                    st.session_state["must_include"],
                    st.session_state["must_avoid"],
                    st.session_state["extra_feedback"],
                    r2["candidates"],
                    r2["ranking"],
                )

            payload = call_text_model_for_prompts(
                client=client,
                model=st.session_state["settings"]["text_model"],
                schema=schema,
                system_instruction=sys_inst,
                user_prompt=user_prompt_any,
                temperature=st.session_state["settings"]["temperature_round3"],
            )
            final_cand = final_candidate_from_payload(payload)
            final_cand.id = 1
            st.session_state["final"]["candidate"] = final_cand.__dict__
            st.session_state["final"]["image"] = None
            st.session_state["final"]["error"] = None

        with st.spinner("最終画像を生成中…"):
            try:
                img_bytes = generate_image_bytes(
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
        st.markdown(f"### ✅ {truncate(final['candidate']['label'], 80)}")
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


def main() -> None:
    if not is_running_with_streamlit():
        print("This app must be run with: streamlit run app.py")
        return
    render_app()


if __name__ == "__main__":
    main()
