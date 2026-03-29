"""Chat-mode evaluation via LM Studio's OpenAI-compatible API.

Use this for behavioral testing. For mechanistic work, load models directly.

Supports:
- Thinking models (qwen3-*-thinking): reasoning in `reasoning_content` field
- Instruct models: reasoning in <think> tags within `content`
"""

import re
import time
from dataclasses import dataclass, asdict
from openai import OpenAI

from lib.utils.config import (
    LMSTUDIO_BASE_URL,
    LMSTUDIO_API_KEY,
    ExperimentConfig,
)

INSTRUCT_SYSTEM = "Think step by step in <think> tags, then answer with just the location name."
THINKING_SYSTEM = "Answer concisely with just the location name."


@dataclass
class ChatResponse:
    """Full response data — nothing truncated."""

    thinking: str
    answer: str
    raw: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def get_client() -> OpenAI:
    return OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)


def extract_location(text: str, valid_locations: list[str]) -> str | None:
    """Extract a known location from model output.

    Instead of crude substring matching, check against the actual locations
    in the stimulus. Returns the matched location or None.
    """
    text_lower = text.lower().strip()
    for loc in valid_locations:
        if loc.lower() in text_lower:
            return loc
    return None


def parse_response(message, raw_content: str) -> tuple[str, str]:
    """Extract thinking and answer from a model response."""
    # Thinking models: separate reasoning_content field
    reasoning_content = getattr(message, "reasoning_content", None) or ""
    if reasoning_content:
        return reasoning_content, (raw_content or "").strip()

    # Instruct models: <think> tags in content
    if "<think>" in raw_content and "</think>" in raw_content:
        start = raw_content.index("<think>") + len("<think>")
        end = raw_content.index("</think>")
        thinking = raw_content[start:end].strip()
        answer = raw_content[end + len("</think>"):].strip()
        return thinking, answer

    return "", raw_content.strip()


def run_scenario(
    scenario_text: str,
    question: str,
    model: str = "",
    config: ExperimentConfig | None = None,
    system_prompt: str | None = None,
) -> ChatResponse:
    """Run a single scenario through LM Studio.

    Auto-selects system prompt based on model type if not provided.
    """
    if config is None:
        config = ExperimentConfig()

    if system_prompt is None:
        is_thinking = "thinking" in model.lower()
        system_prompt = THINKING_SYSTEM if is_thinking else INSTRUCT_SYSTEM

    client = get_client()
    user_content = f"{scenario_text}\n\n{question} Answer with just the location name."

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=config.max_new_tokens,
        temperature=config.temperature,
    )
    elapsed = time.time() - start

    choice = response.choices[0]
    raw = choice.message.content or ""
    thinking, answer = parse_response(choice.message, raw)

    return ChatResponse(
        thinking=thinking,
        answer=answer,
        raw=raw,
        model=response.model or model,
        prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
        completion_tokens=response.usage.completion_tokens if response.usage else 0,
        finish_reason=choice.finish_reason or "",
        elapsed_seconds=round(elapsed, 2),
    )


def score_response(
    response: ChatResponse,
    correct_answer: str,
    all_locations: list[str],
) -> dict:
    """Score a response against the correct answer and all possible locations.

    Returns dict with extracted_answer, is_correct, error_type, etc.
    """
    # Check answer field first, then full raw output
    for text_to_check in [response.answer, response.raw, response.thinking]:
        matched = extract_location(text_to_check, all_locations)
        if matched is not None:
            break

    is_correct = matched is not None and matched.lower() == correct_answer.lower()

    # Classify error type
    if is_correct:
        error_type = None
    elif matched is not None:
        error_type = "wrong_location"
    elif response.finish_reason == "length":
        error_type = "truncated"
    else:
        error_type = "no_location_found"

    return {
        "extracted_answer": matched,
        "is_correct": is_correct,
        "error_type": error_type,
    }
