import json
import os
import signal
import socket

from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation, ClientTools
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

SOCKET_PATH = "/tmp/paft-agent.sock"


def call_rust(method: str, **params) -> str:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(SOCKET_PATH)
    try:
        request = json.dumps({"method": method, **params}) + "\n"
        sock.sendall(request.encode())

        buf = b""
        while not buf.endswith(b"\n"):
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk

        response = json.loads(buf.decode())
        if response.get("result") == "error":
            raise RuntimeError(response.get("message", "unknown error"))

        return json.dumps(response.get("value", {}))
    finally:
        sock.close()


def use_vision(parameters: dict) -> str:
    return call_rust("use_vision")


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
