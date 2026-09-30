"""
Prints the Drive OAuth token JSON that rclone's config needs, taken from the
RCLONE_DRIVE_TOKEN secret. `rclone authorize "drive"` prints it either as JSON
({"access_token": ...}) or, in newer rclone versions, as a base64 blob between
"--->" and "<---End paste"; both are accepted, with any text pasted around them.
"""
import base64
import json
import os
import re
import sys


def as_token(text):
    try:
        v = json.loads(text)
    except ValueError:
        return None
    if isinstance(v, dict) and "token" in v:  # {"token": "{...}"} wrapper
        v = v["token"]
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except ValueError:
                return None
    if isinstance(v, dict) and "refresh_token" in v:
        return json.dumps(v)
    return None


def candidates(raw):
    yield raw
    yield from re.findall(r"\{.*\}", raw, re.S)
    for blob in re.findall(r"[A-Za-z0-9+/_=-]{40,}", raw):
        padded = blob + "=" * (-len(blob) % 4)
        for decode in (base64.b64decode, base64.urlsafe_b64decode):
            try:
                yield decode(padded).decode()
            except Exception:
                pass


def main():
    raw = os.environ.get("RCLONE_DRIVE_TOKEN", "").strip()
    for c in candidates(raw):
        token = as_token(c)
        if token:
            print(token)
            return 0
    print("::error::RCLONE_DRIVE_TOKEN is not an rclone Drive token: paste the whole "
          'output of `rclone authorize "drive"` (the {...} line, or the text between '
          '"--->" and "<---End paste")', file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
