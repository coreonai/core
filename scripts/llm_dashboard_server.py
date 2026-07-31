#!/usr/bin/env python3
"""Static dashboard server + live /status.json for llm.coreon.build.

Serves the dashboard (SITE/index.html, a symlink to docs/dashboard.html) and a
read-only /status.json snapshot the page polls for live state: GPU memory/util,
active phase-22 runs, and git head/sync. Stdlib only; bound to localhost and
exposed via the dedicated cloudflared tunnel (systemd: llm-dashboard.service).
"""
import http.server
import subprocess
import json
import re
import datetime

SITE = "/raid/users/paul/workLLM/scratch-7b-sft/llm_site"
REPO = "/raid/users/paul/workLLM"
PORT = 8137


def sh(cmd):
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=8, cwd=REPO
        ).stdout.strip()
    except Exception:
        return ""


def status():
    gpus = []
    out = sh(["nvidia-smi",
              "--query-gpu=index,memory.used,utilization.gpu",
              "--format=csv,noheader,nounits"])
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 3 and p[0].isdigit():
            gpus.append({"i": int(p[0]), "mem": int(p[1]), "util": int(p[2])})
    busy = sum(1 for g in gpus if g["mem"] > 2000)

    runs, seen = [], set()
    for line in sh(["pgrep", "-af", "phase22_"]).splitlines():
        if "pgrep" in line or "dashboard_server" in line:
            continue
        m = re.search(r"(phase22_[a-z0-9_]+)", line)
        if not m:
            continue
        name = m.group(1)
        bits = []
        s = re.search(r"--seed (\d+)", line)
        if s:
            bits.append("seed " + s.group(1))
        r = re.search(r"--rl-steps (\d+)", line)
        if r:
            bits.append("rl-steps " + r.group(1))
        rr = re.search(r"--rounds (\d+)", line)
        if rr:
            bits.append("rounds " + rr.group(1))
        po = re.search(r"--pg-positive-only", line)
        if po:
            bits.append("posonly")
        key = (name, tuple(bits))
        if key in seen:
            continue
        seen.add(key)
        runs.append({"name": name, "desc": " · ".join(bits)})

    head = sh(["git", "rev-parse", "--short", "HEAD"])
    sb = sh(["git", "status", "-sb"])
    synced = ("[ahead" not in sb) and ("[behind" not in sb)
    return {
        "ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": head, "git_synced": synced,
        "gpus": gpus, "gpu_busy": busy, "gpu_total": len(gpus),
        "runs": runs, "run_count": len(runs),
    }


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=SITE, **k)

    def do_GET(self):
        if self.path.split("?")[0] == "/status.json":
            body = json.dumps(status()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
