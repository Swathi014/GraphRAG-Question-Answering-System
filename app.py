"""
PrivGraphAI Web UI — a local Flask app for running `ingest` and `ask`
through a browser instead of the command line, with live log streaming.

Everything here is local-only:
  - Flask's own dev server, bound to 127.0.0.1 (not reachable from outside
    this machine, let alone the internet)
  - No CDN, no external fonts, no external JS libraries -- static/app.js and
    static/style.css are the only front-end code, served from disk
  - A PDF you pick in the browser is uploaded over the loopback interface
    (127.0.0.1) straight to this process and saved locally -- it never
    leaves the machine

This matches the project's own premise: the UI itself must not depend on
internet access any more than the pipeline it's driving does.

Run with:
    python app.py
then open http://127.0.0.1:5000 in a browser.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from flask import Flask, Response, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB upload cap


class RunManager:
    """Owns at most one subprocess at a time, mirrors stdout+stderr into a
    buffer (so a browser tab connecting mid-run still sees everything from
    the start), and fans new lines out to any number of SSE listeners."""

    def __init__(self):
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._buffer: list[str] = []
        self._listeners: list["queue.Queue[str]"] = []
        self._status = "idle"  # idle | running | finished | failed
        self._exit_code: Optional[int] = None
        self._mode: Optional[str] = None
        self._started_at: Optional[float] = None

    def is_running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self._status,
                "exit_code": self._exit_code,
                "mode": self._mode,
                "running": self._proc is not None and self._proc.poll() is None,
            }

    def start(self, args: list[str], mode: str) -> None:
        if self.is_running():
            raise RuntimeError("A process is already running.")
        cmd = [sys.executable, "-u", "-m", "privgraphai.cli", *args]
        with self._lock:
            self._buffer = [f"$ {' '.join(cmd)}\n", f"(working directory: {BASE_DIR})\n"]
            self._status = "running"
            self._exit_code = None
            self._mode = mode
            self._started_at = time.time()
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        threading.Thread(target=self._reader_thread, daemon=True).start()

    def stop(self) -> bool:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
                self._push_line("\n--- Stopped by user ---\n")
                return True
            return False

    def subscribe(self) -> "queue.Queue[str]":
        """Register a new SSE listener; it first gets a replay of everything
        buffered so far, then live lines as they arrive."""
        q: "queue.Queue[str]" = queue.Queue()
        with self._lock:
            for line in self._buffer:
                q.put(line)
            if self._status != "running":
                q.put(f"__DONE__{self._exit_code}")
            self._listeners.append(q)
        return q

    def unsubscribe(self, q: "queue.Queue[str]") -> None:
        with self._lock:
            if q in self._listeners:
                self._listeners.remove(q)

    def _push_line(self, line: str) -> None:
        # Caller must hold self._lock.
        self._buffer.append(line)
        for q in self._listeners:
            q.put(line)

    def _reader_thread(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            for line in proc.stdout:
                with self._lock:
                    self._push_line(line)
        except Exception as exc:  # pragma: no cover - defensive
            with self._lock:
                self._push_line(f"\n[server] Error reading process output: {exc}\n")
        finally:
            exit_code = proc.wait()
            with self._lock:
                self._exit_code = exit_code
                self._status = "finished" if exit_code == 0 else "failed"
                sentinel = f"__DONE__{exit_code}"
                for q in self._listeners:
                    q.put(sentinel)


runner = RunManager()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify(runner.snapshot())


@app.route("/api/run/ingest", methods=["POST"])
def api_run_ingest():
    if runner.is_running():
        return jsonify({"error": "A process is already running."}), 409

    uploaded = request.files.get("pdf")
    existing_path = (request.form.get("existing_path") or "").strip()
    graph_path = (request.form.get("graph_path") or "graph.json").strip()
    elements_path = (request.form.get("elements_path") or "elements.json").strip()
    verbose = request.form.get("verbose") == "true"

    if uploaded and uploaded.filename:
        safe_name = Path(uploaded.filename).name  # strip any path components
        saved_path = UPLOAD_DIR / safe_name
        uploaded.save(saved_path)
        source = str(saved_path)
    elif existing_path:
        source = existing_path
    else:
        return jsonify({"error": "Upload a PDF or provide an existing file path."}), 400

    args = ["ingest", source, "--graph", graph_path, "--elements", elements_path]
    if verbose:
        args.append("-v")

    try:
        runner.start(args, mode="ingest")
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify({"started": True, "source": source, "graph": graph_path, "elements": elements_path})


@app.route("/api/run/ask", methods=["POST"])
def api_run_ask():
    if runner.is_running():
        return jsonify({"error": "A process is already running."}), 409

    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    graph_path = (data.get("graph_path") or "graph.json").strip()
    elements_path = (data.get("elements_path") or "elements.json").strip()
    verbose = bool(data.get("verbose"))

    if not question:
        return jsonify({"error": "Type a question first."}), 400

    args = ["ask", question, "--graph", graph_path, "--elements", elements_path]
    if verbose:
        args.append("-v")

    try:
        runner.start(args, mode="ask")
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify({"started": True})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    stopped = runner.stop()
    return jsonify({"stopped": stopped})


@app.route("/api/stream")
def api_stream():
    """Server-Sent Events endpoint. Plain browser-native EventSource on the
    client side -- no library needed, and nothing ever leaves localhost."""

    def generate():
        q = runner.subscribe()
        try:
            while True:
                line = q.get()
                if line.startswith("__DONE__"):
                    exit_code = line[len("__DONE__"):]
                    yield f"event: done\ndata: {exit_code}\n\n"
                    break
                yield f"data: {json.dumps(line)}\n\n"
        finally:
            runner.unsubscribe(q)

    return Response(generate(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.route("/api/browse")
def api_browse():
    """List a server-side directory so the browser can help pick existing
    graph.json/elements.json files without needing their full typed path."""
    raw_path = request.args.get("path", str(BASE_DIR))
    target = Path(raw_path).expanduser()
    if not target.exists() or not target.is_dir():
        target = BASE_DIR

    entries = []
    try:
        for p in sorted(target.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
            entries.append({"name": p.name, "path": str(p), "is_dir": p.is_dir()})
    except PermissionError:
        pass

    return jsonify({"path": str(target), "parent": str(target.parent), "entries": entries})


if __name__ == "__main__":
    print(f"PrivGraphAI Web UI starting -- open http://127.0.0.1:5000 in your browser")
    print(f"Project folder: {BASE_DIR}")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
