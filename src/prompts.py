from __future__ import annotations

import textwrap
from typing import Any, Dict, List


def json_schema_for_prompts(n: int) -> Dict[str, Any]:
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


def json_schema_for_final_prompt() -> Dict[str, Any]:
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


def build_system_instruction(language: str) -> str:
    lang_hint = "日本語" if language == "日本語" else "English"
    return textwrap.dedent(
        f"""
        You are a senior prompt engineer for image generation.
        Produce high-quality prompts that work well for Gemini 2.5 Flash Image (Nano Banana).
        Write prompts in {lang_hint}.
        Each prompt must be self-contained and describe:
        - subject, environment, composition, lighting, style/medium
        - optionally camera details (lens, angle) when relevant
        - clear constraints and what to avoid (no watermarks, no unreadable text, etc.) when useful
        Avoid referencing copyrighted characters, trademarks, living artists, or brand names unless the user explicitly requests it.
        Do NOT output markdown. Only output valid JSON that matches the provided schema.
        """
    ).strip()


def build_round1_user_prompt(
    user_intent: str,
    must_include: str,
    must_avoid: str,
    language: str,
) -> str:
    return textwrap.dedent(
        f"""
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
        """
    ).strip()


def build_round2_user_prompt(
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
        f"""
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
        """
    ).strip()


def build_round3_user_prompt(
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
        f"""
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
        """
    ).strip()
