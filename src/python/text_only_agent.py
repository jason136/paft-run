#!/usr/bin/env python3
import json
import os
import signal
import socket
import tempfile

from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import (
    Conversation,
    ClientTools,
    ConversationInitiationData,
)
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

SOCKET_PATH = "/tmp/paft-agent.sock"

# -----------------------------
# Humor file config
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HUMOR_FILE = os.path.join(BASE_DIR, "humor_setting.txt")
DEFAULT_HUMOR = 5


def load_humor():
    try:
        with open(HUMOR_FILE, "r", encoding="utf-8") as f:
            line = f.read().strip()
        if not line:
            return DEFAULT_HUMOR
        if "=" in line:
            _, val = line.split("=", 1)
            val = val.strip()
        else:
            val = line.strip()
        return max(0, min(10, int(val)))
    except Exception:
        return DEFAULT_HUMOR


def save_humor(level: int):
    level = max(0, min(10, int(level)))
    tmp_fd, tmp_path = tempfile.mkstemp(dir=BASE_DIR)
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(f"humor setting = {level}\n")
        os.replace(tmp_path, HUMOR_FILE)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


HUMOR_LEVEL = load_humor()


# -----------------------------
# Tools
# -----------------------------
def set_humor_tool(parameters: dict) -> str:
    global HUMOR_LEVEL
    print("set_humor called with:", parameters)

    raw = parameters.get("level")
    try:
        lvl = int(raw)
    except Exception:
        return json.dumps({
            "status": "error",
            "message": "invalid or missing 'level'"
        })

    lvl = max(0, min(10, lvl))
    HUMOR_LEVEL = lvl

    try:
        save_humor(lvl)
    except Exception as e:
        print("ERROR saving humor:", e)
        return json.dumps({
            "status": "error",
            "message": "failed to persist humor level"
        })

    print("[tool] humor set ->", HUMOR_LEVEL)
    return json.dumps({
        "status": "ok",
        "humor": HUMOR_LEVEL
    })


def get_humor_tool(parameters: dict) -> str:
    global HUMOR_LEVEL
    print("get_humor called with:", parameters)

    HUMOR_LEVEL = load_humor()
    print("[tool] humor get ->", HUMOR_LEVEL)

    return json.dumps({
        "status": "ok",
        "humor": HUMOR_LEVEL
    })


# -----------------------------
# Rust bridge (optional)
# -----------------------------
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
    client_tools.register("set_humor", set_humor_tool)
    client_tools.register("get_humor", get_humor_tool)

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
        audio_interface=DefaultAudioInterface(),
        config=config,
        client_tools=client_tools,
        callback_agent_response=lambda r: print(f"Agent: {r}"),
        callback_user_transcript=lambda t: print(f"User: {t}"),
    )

    signal.signal(signal.SIGINT, lambda sig, frame: conversation.end_session())

    print("Text-only agent running. Type 'exit' to quit.\n")

    conversation.start_session()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input:
            conversation.send_user_message(user_input)

    conversation.end_session()
    conversation.wait_for_session_end()


if __name__ == "__main__":
    main()