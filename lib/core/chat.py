"""Chat-mode evaluation via LM Studio's OpenAI-compatible API.

Use this for all behavioral testing (accuracy measurements, response analysis).
For mechanistic work (activations, ablation), use direct model loading instead.
"""

from dataclasses import dataclass
from openai import OpenAI

from lib.utils.config import (
    LMSTUDIO_BASE_URL,
    LMSTUDIO_API_KEY,
    SYSTEM_PROMPT,
    ExperimentConfig,
)


@dataclass
class ChatResponse:
    """Parsed response from a chat completion."""

    raw: str
    thinking: str  # Content inside <think> tags
    answer: str  # Content after </think>
    model: str


def get_client() -> OpenAI:
    """Get an OpenAI client pointed at LM Studio."""
    return OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)


def parse_response(text: str) -> tuple[str, str]:
    """Extract thinking and answer from a response with <think> tags."""
    if "<think>" in text and "</think>" in text:
        start = text.index("<think>") + len("<think>")
        end = text.index("</think>")
        thinking = text[start:end].strip()
        answer = text[end + len("</think>"):].strip()
    else:
        thinking = ""
        answer = text.strip()
    return thinking, answer


def run_scenario(
    scenario_text: str,
    question: str,
    model: str = "",
    config: ExperimentConfig | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> ChatResponse:
    """Run a single scenario through LM Studio and parse the response.

    Args:
        scenario_text: The narrative/scenario
        question: The question to ask
        model: LM Studio model identifier (leave empty to use whatever's loaded)
        config: Experiment config (uses defaults if None)
        system_prompt: System prompt to use

    Returns:
        ChatResponse with parsed thinking and answer
    """
    if config is None:
        config = ExperimentConfig()

    client = get_client()

    user_content = f"{scenario_text}\n\n{question}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=config.max_new_tokens,
        temperature=config.temperature,
    )

    raw = response.choices[0].message.content or ""
    thinking, answer = parse_response(raw)

    return ChatResponse(
        raw=raw,
        thinking=thinking,
        answer=answer,
        model=response.model or model,
    )


def run_batch(
    scenarios: list[dict],
    model: str = "",
    config: ExperimentConfig | None = None,
) -> list[ChatResponse]:
    """Run a batch of scenarios sequentially.

    Args:
        scenarios: List of dicts with 'text' and 'question' keys
        model: LM Studio model identifier
        config: Experiment config

    Returns:
        List of ChatResponse objects
    """
    results = []
    for s in scenarios:
        resp = run_scenario(
            scenario_text=s["text"],
            question=s["question"],
            model=model,
            config=config,
        )
        results.append(resp)
    return results
