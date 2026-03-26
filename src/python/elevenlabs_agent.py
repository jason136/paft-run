import base64
import os
import signal
import tempfile

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp

from openai import OpenAI

from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation, ClientTools
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

Gst.init(None)

IMAGE_PATH = os.path.join(tempfile.gettempdir(), "paft_capture.jpg")

openai_client = OpenAI()


def use_vision() -> str:
    try:
        pipeline = Gst.parse_launch(
            "qtiqmmfsrc camera=0 ! "
            "video/x-raw,format=NV12,width=1280,height=720,framerate=30/1 ! "
            "jpegenc ! "
            "appsink name=sink emit-signals=false max-buffers=1 drop=true"
        )
        pipeline.set_state(Gst.State.PLAYING)

        sink = pipeline.get_by_name("sink")

        for _ in range(10):
            sample = sink.try_pull_sample(5 * Gst.SECOND)
            if sample is None:
                pipeline.set_state(Gst.State.NULL)
                return "Error: camera failed to capture an image"

        buf = sample.get_buffer()
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        jpeg_bytes = bytes(mapinfo.data)
        buf.unmap(mapinfo)

        pipeline.set_state(Gst.State.NULL)

        b64 = base64.b64encode(jpeg_bytes).decode()

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
            reasoning_effort="none",
        )
        return response.choices[0].message.content

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
    # main()

    print(use_vision())
