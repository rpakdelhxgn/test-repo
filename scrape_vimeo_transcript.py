#!/usr/bin/env python3
"""Fetch the transcript/captions from a Vimeo video."""

import re
import sys
import json
import urllib.request
import urllib.error


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://vimeo.com/",
}


def fetch(url, extra_headers=None):
    h = {**HEADERS, **(extra_headers or {})}
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8")


def extract_video_id(url):
    m = re.search(r"vimeo\.com/(?:video/)?(\d+)", url)
    if not m:
        raise ValueError(f"Cannot parse Vimeo video ID from: {url}")
    return m.group(1)


def get_player_config(video_id):
    """Fetch config from the Vimeo player endpoint (no login required)."""
    config_url = f"https://player.vimeo.com/video/{video_id}/config"
    print(f"Fetching player config: {config_url}", file=sys.stderr)
    data = fetch(config_url, {"Referer": "https://player.vimeo.com/"})
    return json.loads(data)


def get_text_tracks(config):
    try:
        return config["request"]["text_tracks"]
    except (KeyError, TypeError):
        pass
    try:
        return config["request"]["files"]["text_tracks"]
    except (KeyError, TypeError):
        return []


def parse_vtt(vtt_text):
    """Parse WebVTT, return deduplicated plain-text lines."""
    lines = vtt_text.splitlines()
    transcript = []
    current = []

    for line in lines:
        if re.match(r"^\d{2}:\d{2}[:\.\d]* -->", line):
            if current:
                transcript.append(" ".join(current))
                current = []
            continue
        stripped = re.sub(r"<[^>]+>", "", line).strip()
        if stripped:
            current.append(stripped)
        elif current:
            transcript.append(" ".join(current))
            current = []

    if current:
        transcript.append(" ".join(current))

    # Remove consecutive duplicates
    deduped = []
    for line in transcript:
        if not deduped or line != deduped[-1]:
            deduped.append(line)
    return deduped


def get_transcript(video_url):
    video_id = extract_video_id(video_url)
    config = get_player_config(video_id)

    tracks = get_text_tracks(config)
    if not tracks:
        # Provide a helpful hint
        title = config.get("video", {}).get("title", "unknown")
        raise RuntimeError(
            f'No captions/subtitles found for "{title}". '
            "The video owner may not have uploaded captions, or the video may be private."
        )

    # Prefer English; fall back to first track
    track = next((t for t in tracks if (t.get("lang") or "").startswith("en")), tracks[0])
    label = track.get("label") or track.get("lang") or "unknown"
    print(f"Using caption track: {label}", file=sys.stderr)

    url = track["url"]
    if not url.startswith("http"):
        url = "https://vimeo.com" + url

    print(f"Fetching VTT: {url}", file=sys.stderr)
    vtt = fetch(url, {"Referer": "https://player.vimeo.com/"})
    return parse_vtt(vtt)


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "https://vimeo.com/1179709645"
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        lines = get_transcript(url)
        transcript = "\n".join(lines)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(transcript + "\n")
            print(f"Saved transcript to {output_file}", file=sys.stderr)
        else:
            print(transcript)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
