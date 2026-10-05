"""
PrivGraphAI UI — a small desktop app for running `ingest` and `ask` without
typing commands, and watching their output (including errors) live.

Uses only Python's standard library (tkinter) — no extra pip installs,
no Streamlit, no browser, no local web server. Just run:

    python privgraphai_ui.py

from inside your project folder (the one containing the `privgraphai`
package), with your venv active. The window that opens lets you:
  - Pick a PDF and run `ingest`, watching Docling/Ollama's log output live
  - Ask questions against the resulting graph, watching the QA graph's
    routing/retrieval/verification log output live
  - See errors highlighted in red as they happen, not just at the end
  - Stop a run partway through if it's taking too long

How it works: each "Run" click launches the exact same CLI you've been
running by hand (`python -m privgraphai.cli ...`) as a subprocess, using
the same Python interpreter this UI is running under — so it automatically
uses your active venv. A background thread reads that subprocess's output
line by line and hands it to the UI thread through a queue, which is the
standard safe way to update a Tkinter window from a background thread.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

APP_TITLE = "PrivGraphAI — Ingest & Ask"

# Lines containing any of these (case-insensitive) are colored as errors/warnings.
_ERROR_MARKERS = ("error", "traceback", "exception", "failed")
_WARNING_MARKERS = ("warning",)


class ProcessRunner:
    """Launches one subprocess and streams its merged stdout+stderr, line by
    line, into a queue.Queue the UI thread polls. Only one process runs at a
    time; starting a new one while another is active is rejected by the UI.
    """

    def __init__(self, on_line, on_done):
        self._proc: subprocess.Popen | None = None
        self._on_line = on_line  # callback(str) -- called on the UI thread
        self._on_done = on_done  # callback(int exit_code) -- called on the UI thread
        self._queue: "queue.Queue[tuple[str, object]]" = queue.Queue()

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, args: list[str], cwd: str) -> None:
        if self.is_running():
            raise RuntimeError("A process is already running.")
        # -u: unbuffered stdout, so lines appear live instead of batching
        # until the process exits. Without this, long Docling/Ollama runs
        # would look frozen even though they're progressing normally.
        cmd = [sys.executable, "-u", "-m", "privgraphai.cli", *args]
        self._proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self._queue.put(("line", f"$ {' '.join(cmd)}\n(working directory: {cwd})\n"))
        threading.Thread(target=self._reader_thread, daemon=True).start()

    def stop(self) -> None:
        if self.is_running():
            self._proc.terminate()
            self._queue.put(("line", "\n--- Stopped by user ---\n"))

    def _reader_thread(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            for line in proc.stdout:
                self._queue.put(("line", line))
        except Exception as exc:  # pragma: no cover - defensive
            self._queue.put(("line", f"\n[UI] Error reading process output: {exc}\n"))
        finally:
            exit_code = proc.wait()
            self._queue.put(("done", exit_code))

    def poll(self) -> None:
        """Call periodically from the UI thread (e.g. via root.after) to
        drain whatever output has arrived since the last poll."""
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "line":
                    self._on_line(payload)
                elif kind == "done":
                    self._on_done(payload)
        except queue.Empty:
            pass


class PrivGraphAIUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title(APP_TITLE)
        root.geometry("920x680")
        root.minsize(760, 520)

        self.project_dir = tk.StringVar(value=str(Path(__file__).resolve().parent))
        self.status_text = tk.StringVar(value="Idle")

        self.runner = ProcessRunner(on_line=self._append_log, on_done=self._on_process_done)

        self._build_widgets()
        self._poll_loop()

    # -- UI construction -----------------------------------------------

    def _build_widgets(self) -> None:
        pad = {"padx": 8, "pady": 4}

        # Project folder row
        proj_frame = ttk.Frame(self.root)
        proj_frame.pack(fill="x", **pad)
        ttk.Label(proj_frame, text="Project folder:").pack(side="left")
        ttk.Entry(proj_frame, textvariable=self.project_dir).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(proj_frame, text="Change...", command=self._choose_project_dir).pack(side="left")

        # Tabs: Ingest / Ask
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="x", padx=8, pady=4)

        ingest_tab = ttk.Frame(notebook)
        ask_tab = ttk.Frame(notebook)
        notebook.add(ingest_tab, text="Ingest a document")
        notebook.add(ask_tab, text="Ask a question")

        self._build_ingest_tab(ingest_tab)
        self._build_ask_tab(ask_tab)

        # Status + Stop row
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="x", **pad)
        ttk.Label(status_frame, text="Status:").pack(side="left")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_text, foreground="#444")
        self.status_label.pack(side="left", padx=6)
        self.stop_btn = ttk.Button(status_frame, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="right")
        ttk.Button(status_frame, text="Clear log", command=self._clear_log).pack(side="right", padx=6)

        # Log output
        log_frame = ttk.Frame(self.root)
        log_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.log_text = tk.Text(log_frame, wrap="word", state="disabled", bg="#111", fg="#ddd", insertbackground="#ddd")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.log_text.tag_configure("error", foreground="#ff6b6b")
        self.log_text.tag_configure("warning", foreground="#ffb86c")
        self.log_text.tag_configure("answer", foreground="#8be9fd")
        self.log_text.tag_configure("cmd", foreground="#8fff8f")

    def _build_ingest_tab(self, parent: ttk.Frame) -> None:
        pad = {"padx": 8, "pady": 4}

        self.source_pdf = tk.StringVar()
        self.ingest_graph = tk.StringVar(value="graph.json")
        self.ingest_elements = tk.StringVar(value="elements.json")
        self.ingest_verbose = tk.BooleanVar(value=False)

        row = ttk.Frame(parent)
        row.pack(fill="x", **pad)
        ttk.Label(row, text="PDF to ingest:", width=16).pack(side="left")
        ttk.Entry(row, textvariable=self.source_pdf).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row, text="Browse...", command=self._choose_pdf).pack(side="left")

        row2 = ttk.Frame(parent)
        row2.pack(fill="x", **pad)
        ttk.Label(row2, text="Save graph to:", width=16).pack(side="left")
        ttk.Entry(row2, textvariable=self.ingest_graph).pack(side="left", fill="x", expand=True, padx=6)

        row3 = ttk.Frame(parent)
        row3.pack(fill="x", **pad)
        ttk.Label(row3, text="Save elements to:", width=16).pack(side="left")
        ttk.Entry(row3, textvariable=self.ingest_elements).pack(side="left", fill="x", expand=True, padx=6)

        row4 = ttk.Frame(parent)
        row4.pack(fill="x", **pad)
        ttk.Checkbutton(row4, text="Verbose (-v)", variable=self.ingest_verbose).pack(side="left")
        self.ingest_run_btn = ttk.Button(row4, text="Run Ingest", command=self._run_ingest)
        self.ingest_run_btn.pack(side="right")

        ttk.Label(
            parent,
            text="First run is slow (Docling + model loading). Large PDFs can take minutes to hours on CPU-only hardware — this is expected.",
            foreground="#666", wraplength=820, justify="left",
        ).pack(fill="x", padx=8, pady=(0, 4))

    def _build_ask_tab(self, parent: ttk.Frame) -> None:
        pad = {"padx": 8, "pady": 4}

        self.question = tk.StringVar()
        self.ask_graph = tk.StringVar(value="graph.json")
        self.ask_elements = tk.StringVar(value="elements.json")
        self.ask_verbose = tk.BooleanVar(value=False)

        row = ttk.Frame(parent)
        row.pack(fill="x", **pad)
        ttk.Label(row, text="Question:", width=16).pack(side="left")
        entry = ttk.Entry(row, textvariable=self.question)
        entry.pack(side="left", fill="x", expand=True, padx=6)
        entry.bind("<Return>", lambda _event: self._run_ask())

        row2 = ttk.Frame(parent)
        row2.pack(fill="x", **pad)
        ttk.Label(row2, text="Graph file:", width=16).pack(side="left")
        ttk.Entry(row2, textvariable=self.ask_graph).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row2, text="Browse...", command=lambda: self._choose_json(self.ask_graph)).pack(side="left")

        row3 = ttk.Frame(parent)
        row3.pack(fill="x", **pad)
        ttk.Label(row3, text="Elements file:", width=16).pack(side="left")
        ttk.Entry(row3, textvariable=self.ask_elements).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row3, text="Browse...", command=lambda: self._choose_json(self.ask_elements)).pack(side="left")

        row4 = ttk.Frame(parent)
        row4.pack(fill="x", **pad)
        ttk.Checkbutton(row4, text="Verbose (-v)", variable=self.ask_verbose).pack(side="left")
        self.ask_run_btn = ttk.Button(row4, text="Run Ask", command=self._run_ask)
        self.ask_run_btn.pack(side="right")

    # -- actions -----------------------------------------------------------

    def _choose_project_dir(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.project_dir.get())
        if chosen:
            self.project_dir.set(chosen)

    def _choose_pdf(self) -> None:
        chosen = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")])
        if chosen:
            self.source_pdf.set(chosen)

    def _choose_json(self, target_var: tk.StringVar) -> None:
        chosen = filedialog.askopenfilename(
            initialdir=self.project_dir.get(), filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if chosen:
            target_var.set(chosen)

    def _run_ingest(self) -> None:
        if self.runner.is_running():
            messagebox.showwarning(APP_TITLE, "A process is already running. Stop it first.")
            return
        if not self.source_pdf.get().strip():
            messagebox.showerror(APP_TITLE, "Pick a PDF to ingest first.")
            return

        args = ["ingest", self.source_pdf.get(), "--graph", self.ingest_graph.get(), "--elements", self.ingest_elements.get()]
        if self.ingest_verbose.get():
            args.append("-v")
        self._start(args, on_success_label="Ingest finished")

    def _run_ask(self) -> None:
        if self.runner.is_running():
            messagebox.showwarning(APP_TITLE, "A process is already running. Stop it first.")
            return
        if not self.question.get().strip():
            messagebox.showerror(APP_TITLE, "Type a question first.")
            return

        args = ["ask", self.question.get(), "--graph", self.ask_graph.get(), "--elements", self.ask_elements.get()]
        if self.ask_verbose.get():
            args.append("-v")
        self._start(args, on_success_label="Ask finished")

    def _start(self, args: list[str], on_success_label: str) -> None:
        self._last_success_label = on_success_label
        self._set_running(True)
        self._append_log(f"\n=== Starting: {' '.join(args[:1])} ===\n", tag="cmd")
        try:
            self.runner.start(args, cwd=self.project_dir.get())
        except Exception as exc:
            self._set_running(False)
            messagebox.showerror(APP_TITLE, f"Could not start process:\n{exc}")

    def _stop(self) -> None:
        self.runner.stop()

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    # -- process plumbing -----------------------------------------------

    def _poll_loop(self) -> None:
        self.runner.poll()
        self.root.after(80, self._poll_loop)

    def _append_log(self, line: str, tag: str | None = None) -> None:
        if tag is None:
            lowered = line.lower()
            if any(marker in lowered for marker in _ERROR_MARKERS):
                tag = "error"
            elif any(marker in lowered for marker in _WARNING_MARKERS):
                tag = "warning"
            elif line.startswith("Answer:") or "Strategy used" in line:
                tag = "answer"
        self.log_text.configure(state="normal")
        if tag:
            self.log_text.insert("end", line, tag)
        else:
            self.log_text.insert("end", line)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _on_process_done(self, exit_code: int) -> None:
        self._set_running(False)
        if exit_code == 0:
            self.status_text.set(f"{self._last_success_label} (exit code 0)")
            self.status_label.configure(foreground="#2a7")
            # Convenience: once an ingest succeeds, carry its output paths over to the Ask tab.
            if self._last_success_label.startswith("Ingest"):
                self.ask_graph.set(self.ingest_graph.get())
                self.ask_elements.set(self.ingest_elements.get())
        else:
            self.status_text.set(f"Failed (exit code {exit_code}) — see log for details")
            self.status_label.configure(foreground="#d33")

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        self.ingest_run_btn.configure(state=state)
        self.ask_run_btn.configure(state=state)
        self.stop_btn.configure(state=("normal" if running else "disabled"))
        if running:
            self.status_text.set("Running...")
            self.status_label.configure(foreground="#444")


def main() -> None:
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")  # native-looking on Windows
    except Exception:
        pass
    PrivGraphAIUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
