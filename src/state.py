from __future__ import annotations

import streamlit as st


def ensure_state() -> None:
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
            "images": {},  # id -> bytes (png)
            "errors": {},  # id -> str
            "ranking": [],  # List[int]
        },
        "round2": {
            "candidates": [],
            "images": {},
            "errors": {},
            "ranking": [],
        },
        "final": {
            "candidate": None,  # PromptCandidate as dict
            "image": None,  # bytes
            "error": None,  # str
        },
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_all() -> None:
    for k in ["round1", "round2", "final"]:
        if k in st.session_state:
            del st.session_state[k]
    ensure_state()
