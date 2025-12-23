from __future__ import annotations

import io
import json
import os
from typing import Any, Dict, Optional

import streamlit as st
from PIL import Image

from google import genai
from google.genai import types


def load_dotenv_if_present(path: str = ".env") -> None:
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        return


def get_api_key_from_env() -> Optional[str]:
    load_dotenv_if_present()
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
        inline_data = getattr(part, "inline_data", None)
        if inline_data is not None and getattr(inline_data, "data", None):
            return inline_data.data
        if hasattr(part, "as_image"):
            img = part.as_image()
            buf = io.BytesIO()
            if isinstance(img, Image.Image):
                img.save(buf, "PNG")
            else:
                try:
                    img.save(buf, "PNG")
                except Exception as e:
                    raise RuntimeError("Unsupported image object from model response.") from e
            return buf.getvalue()

    raise RuntimeError("Model response did not include an image.")
