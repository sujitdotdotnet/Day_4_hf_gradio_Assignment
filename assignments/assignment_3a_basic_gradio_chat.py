from __future__ import annotations

import os
from typing import Any

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

from assignment_2_multimodal_messages import build_multimodal_messages


load_dotenv()


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"
APP_TITLE = "Basic Multimodal Chat"


def stream_basic_chat(
    message: dict[str, Any],
    history: list[dict[str, Any]],
    api_key: str,
):
    """Stream a multimodal response into Gradio."""

    api_key = (
        api_key
        or os.getenv("OPENROUTER_API_KEY")
        or ""
    ).strip()

    if not api_key:
        yield "Add your OpenRouter API key first."
        return

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )

    # Build messages first so we can validate/debug them.
    messages = build_multimodal_messages(
        history,
        message,
    )

    print("Current Gradio message:")
    print(message)

    print("Messages being sent to OpenRouter:")
    print(messages)

    if not messages:
        yield "No messages were generated for OpenRouter."
        return

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            stream=True,
            extra_body={
                "provider": {
                    "data_collection": "deny"
                }
            },
        )

        answer = ""

        for chunk in response:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta.content

            if delta:
                answer += delta
                yield answer

    except Exception as exc:
        yield f"OpenRouter error: {exc}"


def build_demo() -> gr.ChatInterface:
    """Create the basic Gradio app."""

    return gr.ChatInterface(
        fn=stream_basic_chat,
        multimodal=True,
        title=APP_TITLE,
        textbox=gr.MultimodalTextbox(
            file_types=["image"],
            placeholder="Type a message or upload an image...",
        ),
        additional_inputs=[
            gr.Textbox(
                label="OpenRouter API Key",
                type="password",
            )
        ],
    )


demo = build_demo()


if __name__ == "__main__":
    demo.launch()