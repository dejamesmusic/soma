"""
soma_logos_bridge.py — bridge between a local soma GUI and logOS remote chat.

Launched as a subprocess by the GUI. Communicates via stdout lines.
Polls the logOS API for incoming prompts, forwards them to the GUI,
and posts model responses back.

Protocol (stdout lines, read by ManagedProcess):
    LOGOS:started                     — bridge is running
    LOGOS:connected                   — first successful poll
    LOGOS:prompt:<json_string>        — incoming prompt from web user
    LOGOS:thinking                    — server has an unanswered prompt
    LOGOS:posted                      — response successfully posted
    LOGOS:error:<message>             — recoverable error
    LOGOS:stopped                     — clean shutdown

Protocol (stdin lines, written by GUI):
    RESPONSE:<json_string>            — model response to post back

The pass_prompt is a shared secret. Anyone who knows it can send
prompts to your model. Treat it like a password.
"""

import sys
import time
import json
import signal
import select
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from urllib.error import URLError, HTTPError

POLL_INTERVAL = 1.8
MAX_RESPONSE_CHARS = 16_000
POST_RETRIES = 3
POST_RETRY_SLEEP = 1.5


def _post_json(url, data, timeout=10):
    """POST JSON, return parsed response and HTTP status."""
    body = json.dumps(data).encode("utf-8")
    req = Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        parsed = json.loads(raw) if raw else {}
        return parsed, resp.status


def _get_json(url, params=None, timeout=5):
    """GET with query params, return parsed response and HTTP status."""
    if params:
        url = f"{url}?{urlencode(params)}"
    req = Request(url)
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        parsed = json.loads(raw) if raw else {}
        return parsed, resp.status


def _read_response_line():
    """Read one RESPONSE line from stdin, preserving newlines via JSON."""
    line = sys.stdin.readline()
    if not line:
        return None
    line = line.rstrip("\n")
    if not line.startswith("RESPONSE:"):
        return None

    payload = line[len("RESPONSE:"):]
    try:
        value = json.loads(payload)
        return value if isinstance(value, str) else str(value)
    except json.JSONDecodeError:
        return payload


def _post_response(api, pass_prompt, response_text):
    """Post the completed model response with capped retries."""
    response_text = (response_text or "")[:MAX_RESPONSE_CHARS]
    url = f"{api}/api/chat/log-os/response"
    body = {
        "pass_prompt": pass_prompt,
        "response": response_text,
    }

    last_error = None
    for attempt in range(POST_RETRIES):
        try:
            _data, status = _post_json(url, body, timeout=10)
            if status == 200:
                print("LOGOS:posted", flush=True)
                return True
            last_error = f"http {status}"
        except HTTPError as e:
            # 404 means the browser/session has expired or moved on.
            last_error = f"http {e.code}: {e.reason}"
        except URLError as e:
            last_error = str(e)
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

        if attempt < POST_RETRIES - 1:
            time.sleep(POST_RETRY_SLEEP)

    print(f"LOGOS:error:post failed: {last_error}", flush=True)
    return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass-prompt", required=True)
    parser.add_argument("--api-base", default="https://logossoma.com")
    args = parser.parse_args()

    pass_prompt = args.pass_prompt.strip()
    api = args.api_base.rstrip("/")

    if not pass_prompt:
        print("LOGOS:error:pass prompt required", flush=True)
        return 2
    if len(pass_prompt) > 200:
        print("LOGOS:error:pass prompt too long", flush=True)
        return 2

    _stop = False

    def on_signal(sig, frame):
        nonlocal _stop
        _stop = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    print("LOGOS:started", flush=True)

    connected = False
    last_prompt_id = None       # every prompt handed to the GUI already
    pending_prompt_id = None    # prompt currently waiting for GUI response

    while not _stop:
        try:
            # Non-blocking stdin check for completed GUI/model response.
            if select.select([sys.stdin], [], [], 0)[0]:
                response_text = _read_response_line()
                if response_text is not None:
                    if pending_prompt_id is not None:
                        _post_response(api, pass_prompt, response_text)
                        pending_prompt_id = None
                    else:
                        print("LOGOS:error:response with no pending prompt",
                              flush=True)

            # One in-flight prompt per pass prompt. Do not poll for another
            # prompt while the GUI/model is still generating.
            if pending_prompt_id is not None:
                time.sleep(POLL_INTERVAL)
                continue

            data, _status = _get_json(
                f"{api}/api/chat/log-os/poll",
                params={"pass_prompt": pass_prompt},
                timeout=5,
            )

            if not connected:
                connected = True
                print("LOGOS:connected", flush=True)

            prompt = data.get("prompt")
            prompt_id = data.get("prompt_id")

            # Commit before generation starts so repeated polls/retries cannot
            # double-fire the same prompt.
            if prompt and prompt_id and prompt_id != last_prompt_id:
                last_prompt_id = prompt_id
                pending_prompt_id = prompt_id
                print(f"LOGOS:prompt:{json.dumps(prompt)}", flush=True)
            elif data.get("thinking"):
                print("LOGOS:thinking", flush=True)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"LOGOS:error:{type(e).__name__}: {e}", flush=True)
            time.sleep(3)
            continue

        time.sleep(POLL_INTERVAL)

    print("LOGOS:stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
