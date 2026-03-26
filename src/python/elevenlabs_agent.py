import base64
import os
import signal
import subprocess
import tempfile

from openai import OpenAI

from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation, ClientTools
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

IMAGE_PATH = os.path.join(tempfile.gettempdir(), "paft_capture.jpg")

openai_client = OpenAI()


def use_vision() -> str:
    try:
        subprocess.run(
            [
                "gst-launch-1.0",
                "-e",
                "qtiqmmfsrc",
                "camera=0",
                "!",
                "video/x-raw,format=NV12,width=1280,height=720,framerate=30/1",
                "!",
                "jpegenc",
                "!",
                "multifilesink",
                f"location={IMAGE_PATH}",
                "max-files=1",
            ],
            timeout=3,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        with open(IMAGE_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        response = openai_client.chat.completions.create(
            model="gpt-5.4-nano",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe what you see in this image concisely.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
            reasoning={"effort": "none"},
            max_tokens=128,
        )
        return response.choices[0].message.content

    except subprocess.TimeoutExpired:
        return "Error: camera capture timed out"
    except Exception as e:
        return f"Error: {e}"


def main():
    agent_id = "agent_0801kggw2vmxeh1vbv5y7yr3hj7n"
    api_key = os.environ.get("ELEVENLABS_API_KEY")

    client = ElevenLabs(api_key=api_key)

    client_tools = ClientTools()
    client_tools.register("use_vision", use_vision)

    conversation = Conversation(
        client,
        agent_id,
        requires_auth=bool(api_key),
        audio_interface=DefaultAudioInterface(),
        client_tools=client_tools,
        callback_agent_response=lambda r: print(f"Agent: {r}"),
        callback_agent_response_correction=lambda o, c: print(f"Agent: {o} -> {c}"),
        callback_user_transcript=lambda t: print(f"User: {t}"),
    )

    conversation.start_session()
    signal.signal(signal.SIGINT, lambda sig, frame: conversation.end_session())

    conversation_id = conversation.wait_for_session_end()
    print(f"Conversation ID: {conversation_id}")


if __name__ == "__main__":
    main()
