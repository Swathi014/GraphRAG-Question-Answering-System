// PrivGraphAI Web UI -- vanilla JS, no build step, no external libraries.
// Talks only to this same origin (127.0.0.1) via fetch() and EventSource.

const el = (id) => document.getElementById(id);

const statusBadge = el("status-badge");
const statusDetail = el("status-detail");
const stopBtn = el("stop-btn");
const logConsole = el("log-console");

let currentEventSource = null;

// ---------- Tabs ----------
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
    btn.classList.add("active");
    el("tab-" + btn.dataset.tab).classList.add("active");
  });
});

// ---------- Source mode toggle (upload vs existing path) ----------
document.querySelectorAll('input[name="source-mode"]').forEach((radio) => {
  radio.addEventListener("change", () => {
    const useUpload = document.querySelector('input[name="source-mode"]:checked').value === "upload";
    el("upload-row").classList.toggle("hidden", !useUpload);
    el("path-row").classList.toggle("hidden", useUpload);
  });
});

// ---------- Log helpers ----------
function classifyLine(line) {
  const lower = line.toLowerCase();
  if (lower.includes("error") || lower.includes("traceback") || lower.includes("exception")) return "log-line-error";
  if (lower.includes("warning")) return "log-line-warning";
  if (line.startsWith("Answer:") || line.includes("Strategy used")) return "log-line-answer";
  if (line.startsWith("$ ")) return "log-line-cmd";
  return "";
}

function appendLogLine(line) {
  const span = document.createElement("span");
  const cls = classifyLine(line);
  if (cls) span.className = cls;
  span.textContent = line;
  logConsole.appendChild(span);
  logConsole.scrollTop = logConsole.scrollHeight;
}

el("clear-log").addEventListener("click", () => {
  logConsole.textContent = "";
});

// ---------- Status ----------
function setStatus(status, detail) {
  statusBadge.className = "badge badge-" + status;
  statusBadge.textContent = status.charAt(0).toUpperCase() + status.slice(1);
  statusDetail.textContent = detail || "";
}

function setRunningUI(running) {
  stopBtn.disabled = !running;
  el("run-ingest").disabled = running;
  el("run-ask").disabled = running;
}

// ---------- Streaming ----------
function startStreaming() {
  if (currentEventSource) currentEventSource.close();
  currentEventSource = new EventSource("/api/stream");

  currentEventSource.onmessage = (event) => {
    const line = JSON.parse(event.data);
    appendLogLine(line);
  };

  currentEventSource.addEventListener("done", (event) => {
    const exitCode = event.data;
    currentEventSource.close();
    currentEventSource = null;
    setRunningUI(false);
    if (exitCode === "0") {
      setStatus("finished", "exit code 0");
    } else {
      setStatus("failed", `exit code ${exitCode} — see output below`);
    }
  });

  currentEventSource.onerror = () => {
    // Connection dropped unexpectedly (e.g. server restarted). Don't spin forever.
    if (currentEventSource) {
      currentEventSource.close();
      currentEventSource = null;
    }
  };
}

// ---------- Run: Ingest ----------
el("run-ingest").addEventListener("click", async () => {
  const useUpload = document.querySelector('input[name="source-mode"]:checked').value === "upload";
  const formData = new FormData();

  if (useUpload) {
    const fileInput = el("pdf-file");
    if (!fileInput.files.length) {
      alert("Choose a PDF file first.");
      return;
    }
    formData.append("pdf", fileInput.files[0]);
  } else {
    const path = el("pdf-path").value.trim();
    if (!path) {
      alert("Enter a path to an existing PDF first.");
      return;
    }
    formData.append("existing_path", path);
  }

  formData.append("graph_path", el("ingest-graph").value.trim() || "graph.json");
  formData.append("elements_path", el("ingest-elements").value.trim() || "elements.json");
  formData.append("verbose", el("ingest-verbose").checked ? "true" : "false");

  appendLogLine(`\n=== Starting: ingest ===\n`);
  setStatus("running");
  setRunningUI(true);

  const resp = await fetch("/api/run/ingest", { method: "POST", body: formData });
  const data = await resp.json();
  if (!resp.ok) {
    setRunningUI(false);
    setStatus("failed", data.error || "Could not start");
    appendLogLine(`[browser] ${data.error}\n`);
    return;
  }
  // On success, carry the output paths over to the Ask tab for convenience.
  el("ask-graph").value = data.graph;
  el("ask-elements").value = data.elements;
  startStreaming();
});

// ---------- Run: Ask ----------
el("run-ask").addEventListener("click", async () => {
  const question = el("ask-question").value.trim();
  if (!question) {
    alert("Type a question first.");
    return;
  }

  appendLogLine(`\n=== Starting: ask ===\n`);
  setStatus("running");
  setRunningUI(true);

  const resp = await fetch("/api/run/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      graph_path: el("ask-graph").value.trim() || "graph.json",
      elements_path: el("ask-elements").value.trim() || "elements.json",
      verbose: el("ask-verbose").checked,
    }),
  });
  const data = await resp.json();
  if (!resp.ok) {
    setRunningUI(false);
    setStatus("failed", data.error || "Could not start");
    appendLogLine(`[browser] ${data.error}\n`);
    return;
  }
  startStreaming();
});

// ---------- Stop ----------
stopBtn.addEventListener("click", async () => {
  await fetch("/api/stop", { method: "POST" });
});

// ---------- File/folder browse modal ----------
const modal = el("browse-modal");
let browseTargetInput = null;

document.querySelectorAll("[data-browse-target]").forEach((btn) => {
  btn.addEventListener("click", () => {
    browseTargetInput = el(btn.dataset.browseTarget);
    const startPath = browseTargetInput.value.trim();
    openBrowser(startPath || ".");
  });
});

el("browse-close").addEventListener("click", () => modal.classList.add("hidden"));
modal.addEventListener("click", (e) => { if (e.target === modal) modal.classList.add("hidden"); });

async function openBrowser(path) {
  const resp = await fetch("/api/browse?path=" + encodeURIComponent(path));
  const data = await resp.json();
  renderBrowser(data);
  modal.classList.remove("hidden");
}

function renderBrowser(data) {
  el("browse-path").textContent = data.path;
  const list = el("browse-list");
  list.innerHTML = "";

  const upItem = document.createElement("div");
  upItem.className = "browse-item is-dir";
  upItem.textContent = ".. (up one level)";
  upItem.addEventListener("click", () => openBrowser(data.parent));
  list.appendChild(upItem);

  data.entries.forEach((entry) => {
    const item = document.createElement("div");
    item.className = "browse-item" + (entry.is_dir ? " is-dir" : "");
    item.textContent = (entry.is_dir ? "📁 " : "📄 ") + entry.name;
    item.addEventListener("click", () => {
      if (entry.is_dir) {
        openBrowser(entry.path);
      } else {
        browseTargetInput.value = entry.path;
        modal.classList.add("hidden");
      }
    });
    list.appendChild(item);
  });
}

// ---------- Restore state on page load (e.g. after a refresh mid-run) ----------
(async function init() {
  const resp = await fetch("/api/status");
  const data = await resp.json();
  if (data.running) {
    setStatus("running");
    setRunningUI(true);
    startStreaming();
  } else if (data.status === "finished" || data.status === "failed") {
    setStatus(data.status, data.exit_code !== null ? `exit code ${data.exit_code}` : "");
  }
})();
