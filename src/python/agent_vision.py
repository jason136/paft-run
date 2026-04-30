#!/usr/bin/env python3
import base64
import json
import os
import signal

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp

from openai import OpenAI

from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import (
    Conversation,
    ClientTools,
    ConversationInitiationData,
)
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

Gst.init(None)

# -----------------------------
# Image / camera config
# -----------------------------
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "paft_capture.jpg")

openai_client = OpenAI()


def capture_image_bytes() -> bytes:
    pipeline = Gst.parse_launch(
        "qtiqmmfsrc camera=0 ! "
        "video/x-raw,format=NV12,width=1280,height=720,framerate=30/1 ! "
        "jpegenc ! "
        "appsink name=sink emit-signals=false max-buffers=1 drop=true"
    )
    pipeline.set_state(Gst.State.PLAYING)

    try:
        sink = pipeline.get_by_name("sink")
        sample = None

        # Skip the first 9 frames and keep the 10th one.
        for _ in range(10):
            sample = sink.try_pull_sample(5 * Gst.SECOND)
            if sample is None:
                continue

        if sample is None:
            raise RuntimeError("camera failed to capture an image")

        buf = sample.get_buffer()
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("failed to read camera buffer")

        try:
            return bytes(mapinfo.data)
        finally:
            buf.unmap(mapinfo)
    finally:
        pipeline.set_state(Gst.State.NULL)


# -----------------------------
# Tool
# -----------------------------
def use_vision(parameters: dict) -> str:
    """
    Client tool called by the ElevenLabs agent.

    Expected parameter:
      - question: the user's camera / vision question
    """
    print("use_vision called with:", parameters)

    question = parameters.get("question") or "Describe what you see in this image concisely."

    try:
        jpeg_bytes = capture_image_bytes()

        # Save locally so you can inspect the frame later.
        with open(IMAGE_PATH, "wb") as f:
            f.write(jpeg_bytes)

        b64 = base64.b64encode(jpeg_bytes).decode("utf-8")

        # Ask the vision backend the user's exact question.
        response = openai_client.chat.completions.create(
            model="gpt-5.4-nano",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
            reasoning_effort="none",
        )

        answer = response.choices[0].message.content or "I could not determine an answer from the image."

        return json.dumps(
            {
                "status": "ok",
                "image_path": IMAGE_PATH,
                "answer": answer,
            }
        )

    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": str(e),
            }
        )


# -----------------------------
# Main
# -----------------------------
def main():
    agent_id = "agent_0801kggw2vmxeh1vbv5y7yr3hj7n"
    api_key = os.environ.get("ELEVENLABS_API_KEY")

    if not api_key:
        print("ERROR: ELEVENLABS_API_KEY not set")
        return

    client = ElevenLabs(api_key=api_key)

    client_tools = ClientTools()
    client_tools.register("use_vision", use_vision)

    # Text-only mode for debugging the agent/tool flow
    config = ConversationInitiationData(
        conversation_config_override={
            "conversation": {
                "text_only": True
            }
        }
    )

    conversation = Conversation(
        client=client,
        agent_id=agent_id,
        requires_auth=bool(api_key),
        audio_interface=DefaultAudioInterface(),  # kept for SDK compatibility
        config=config,
        client_tools=client_tools,
        callback_agent_response=lambda r: print(f"Agent: {r}"),
        callback_agent_response_correction=lambda o, c: print(f"Agent: {o} -> {c}"),
        callback_user_transcript=lambda t: print(f"User: {t}"),
    )

    signal.signal(signal.SIGINT, lambda sig, frame: conversation.end_session())

    print("Text-only vision debug agent running. Type 'exit' to quit.\n")
    conversation.start_session()
    print("Session started.\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            if user_input:
                conversation.send_user_message(user_input)
    except KeyboardInterrupt:
        print("\nStopping session...")
    finally:
        conversation.end_session()
        conversation.wait_for_session_end()


if __name__ == "__main__":
    main()