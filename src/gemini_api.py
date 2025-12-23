from __future__ import annotations

import io
import json
import os
from typing import Any, Dict, Optional

import streamlit as st
from PIL import Image

from google import genai
from google.genai import types


def get_api_key_from_env_or_ui(ui_key: str) -> Optional[str]:
    if ui_key and ui_key.strip():
        return ui_key.strip()
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


@st.cache_resource(show_spinner=False)
def get_client(api_key: str) -> genai.Client:
    # Gemini Developer API (Google AI Studio key)
    return genai.Client(api_key=api_key)


def call_text_model_for_prompts(
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


def generate_image_bytes(
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
