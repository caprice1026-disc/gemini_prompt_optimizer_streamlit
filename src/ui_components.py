from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

try:
    from streamlit_sortables import sort_items  # pip install streamlit-sortables

    HAS_SORTABLES = True
except Exception:
    HAS_SORTABLES = False


def truncate(s: str, n: int = 60) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def render_gallery(candidates: List[Dict[str, Any]], images: Dict[str, bytes], cols: int = 3) -> None:
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
                st.markdown(f"### #{cid} — {truncate(cand['label'], 40)}")
                if cid in images:
                    st.image(images[cid], use_container_width=True)
                else:
                    st.info("まだ画像がありません。")
                with st.expander("プロンプトを見る"):
                    st.code(cand["prompt"], language="text")
            idx += 1


def rank_ui(
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

    methods = ["並び順を数字で入力"]
    if HAS_SORTABLES:
        methods.insert(0, "ドラッグ&ドロップ (おすすめ)")

    method = st.radio("ランキング方法", methods, horizontal=True, key=f"rank_method_{title}")

    if method.startswith("ドラッグ") and HAS_SORTABLES:
        items: List[str] = []
        if default_ranking and len(default_ranking) == len(ids):
            for i in default_ranking:
                items.append(f"{i}: {truncate(labels[i], 60)}")
        else:
            for i in ids:
                items.append(f"{i}: {truncate(labels[i], 60)}")

        custom_style = """
        .sortable-container { counter-reset: item; }
        .sortable-item::before { content: counter(item) ". "; counter-increment: item; }
        """
        sorted_items = sort_items(items, custom_style=custom_style)
        ranking = [int(x.split(":", 1)[0].strip()) for x in sorted_items]
        st.write("現在のランキング:", ranking)
        return ranking

    import pandas as pd

    if default_ranking and len(default_ranking) == len(ids):
        initial_rank = {cid: i + 1 for i, cid in enumerate(default_ranking)}
    else:
        initial_rank = {cid: i + 1 for i, cid in enumerate(ids)}

    df = pd.DataFrame(
        [{"id": cid, "label": truncate(labels[cid], 80), "rank(1=best)": initial_rank[cid]} for cid in ids]
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
    st.write("現在のランキング:", ranking)
    return ranking
