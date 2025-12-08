#!/usr/bin/env python3
"""
Local UI for browsing generated HTML outputs (graphs + reports).

Run:
    python ui/report_viewer.py --port 5050

The app lists recent HTML files under outputs/graphs and outputs/reports and
renders them in an iframe for quick viewing.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from flask import Flask, abort, jsonify, render_template_string, request, send_from_directory

# Ensure project root is on path when run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import Config

# Resolve output locations
OUTPUT_ROOT = Path(Config.OUTPUT_DIR).resolve()
GRAPH_DIR = Path(Config.GRAPHS_DIR).resolve()
REPORT_DIR = Path(Config.REPORTS_DIR).resolve()

app = Flask(__name__)
app.json_sort_keys = False


def _serialize_file(path: Path) -> Dict:
    """Return metadata for a single file."""
    stat = path.stat()
    rel_path = path.relative_to(OUTPUT_ROOT)
    return {
        "name": path.name,
        "path": str(rel_path).replace("\\", "/"),
        "pretty_name": path.stem.replace("_", " ").title(),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "size_kb": round(stat.st_size / 1024, 1),
    }


def _list_html_files(directory: Path) -> List[Dict]:
    """List HTML files in a directory sorted by modified time (newest first)."""
    if not directory.exists():
        return []
    files = sorted(directory.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [_serialize_file(path) for path in files]


@app.get("/api/files")
def list_files():
    """Return JSON listing of available HTML outputs."""
    graphs = _list_html_files(GRAPH_DIR)
    reports = _list_html_files(REPORT_DIR)
    recent = sorted(graphs + reports, key=lambda f: f["modified"], reverse=True)[:10]
    return jsonify(
        {
            "graphs": graphs,
            "reports": reports,
            "recent": recent,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "output_root": str(OUTPUT_ROOT),
        }
    )


@app.get("/view")
def view_file():
    """Serve an HTML file from the outputs directory."""
    rel_path = request.args.get("path")
    if not rel_path:
        abort(400, description="Missing path parameter")

    safe_path = (OUTPUT_ROOT / rel_path).resolve()
    if not str(safe_path).startswith(str(OUTPUT_ROOT)):
        abort(403, description="Invalid path")
    if not safe_path.exists():
        abort(404, description="File not found")

    # Split directory and filename for send_from_directory
    return send_from_directory(OUTPUT_ROOT, safe_path.relative_to(OUTPUT_ROOT))


# Inline HTML keeps the viewer self-contained
INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Botnet Detector Output Viewer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {
      --bg: #0c1222;
      --panel: #111b34;
      --panel-2: #0f1a2d;
      --accent: #6de0ff;
      --accent-2: #f76c9c;
      --text: #e8edf7;
      --muted: #9fb3d1;
      --border: #1f2c48;
      --shadow: 0 20px 80px rgba(5, 8, 20, 0.6);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: 'Inter', 'Manrope', 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: radial-gradient(circle at 15% 20%, rgba(109,224,255,0.1), transparent 25%),
                  radial-gradient(circle at 80% 0%, rgba(247,108,156,0.08), transparent 30%),
                  var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 0;
    }
    aside {
      background: linear-gradient(160deg, var(--panel), var(--panel-2));
      border-right: 1px solid var(--border);
      padding: 24px;
      box-shadow: var(--shadow);
      position: relative;
      overflow-y: auto;
    }
    main {
      padding: 28px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    h1 {
      margin: 0 0 12px 0;
      font-size: 24px;
      letter-spacing: -0.02em;
    }
    p.subtitle {
      margin: 0 0 18px 0;
      color: var(--muted);
      font-size: 14px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: rgba(109,224,255,0.12);
      border: 1px solid rgba(109,224,255,0.3);
      border-radius: 999px;
      padding: 8px 12px;
      color: var(--text);
      font-size: 12px;
      margin-right: 8px;
    }
    .section {
      margin-bottom: 18px;
    }
    .section h3 {
      margin: 0 0 10px 0;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
    }
    .file-list {
      list-style: none;
      padding: 0;
      margin: 0;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .file-list li button {
      width: 100%;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.02);
      color: var(--text);
      border-radius: 12px;
      padding: 10px 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      cursor: pointer;
      transition: border-color 0.15s ease, transform 0.12s ease, background 0.15s ease;
    }
    .file-list li button:hover {
      border-color: var(--accent);
      transform: translateY(-1px);
      background: rgba(109,224,255,0.06);
    }
    .file-meta {
      display: flex;
      gap: 10px;
      color: var(--muted);
      font-size: 12px;
    }
    .refresh-row {
      display: flex;
      gap: 10px;
      align-items: center;
      margin-bottom: 14px;
    }
    .refresh-row button {
      padding: 10px 14px;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      color: #0a0f1c;
      border: none;
      border-radius: 10px;
      font-weight: 700;
      cursor: pointer;
      transition: transform 0.12s ease, box-shadow 0.12s ease;
      box-shadow: 0 8px 24px rgba(109,224,255,0.25);
    }
    .refresh-row button:hover { transform: translateY(-1px); }
    .refresh-row span { color: var(--muted); font-size: 12px; }
    .viewer {
      flex: 1;
      background: #0d1629;
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: var(--shadow);
      position: relative;
    }
    .viewer header {
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(255,255,255,0.02);
    }
    .viewer header .title {
      font-size: 14px;
      color: var(--muted);
    }
    .viewer header .actions {
      display: flex;
      gap: 10px;
      align-items: center;
    }
    .viewer header a {
      color: var(--accent);
      text-decoration: none;
      font-size: 13px;
    }
    .viewer iframe {
      width: 100%;
      height: calc(100vh - 170px);
      border: none;
      background: #0d1629;
    }
    .empty {
      position: absolute;
      inset: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      color: var(--muted);
      gap: 8px;
      pointer-events: none;
    }
    .badge {
      padding: 4px 8px;
      border-radius: 8px;
      background: rgba(255,255,255,0.05);
      border: 1px solid var(--border);
      font-size: 12px;
      color: var(--muted);
    }
    @media (max-width: 960px) {
      body { grid-template-columns: 1fr; }
      aside { border-right: none; border-bottom: 1px solid var(--border); }
      .viewer iframe { height: 60vh; }
    }
  </style>
</head>
<body>
  <aside>
    <h1>Output Viewer</h1>
    <p class="subtitle">Browse recent HTML reports and graphs generated by the botnet detection runs.</p>
    <div class="pill">Outputs root: {{ output_root }}</div>
    <div class="section">
      <div class="refresh-row">
        <button id="refresh-btn">↻ Refresh list</button>
        <span id="last-updated">Loading…</span>
      </div>
    </div>
    <div class="section">
      <h3>Recent</h3>
      <ul id="recent-list" class="file-list"></ul>
    </div>
    <div class="section">
      <h3>Graphs (HTML)</h3>
      <ul id="graphs-list" class="file-list"></ul>
    </div>
    <div class="section">
      <h3>Reports (HTML)</h3>
      <ul id="reports-list" class="file-list"></ul>
    </div>
  </aside>
  <main>
    <div class="viewer">
      <header>
        <div>
          <div class="title">Currently viewing</div>
          <div id="current-file" style="font-weight: 700; font-size: 15px;">Select a file to preview</div>
        </div>
        <div class="actions">
          <span id="file-meta" class="badge">—</span>
          <a id="open-tab" href="#" target="_blank" rel="noopener" style="display:none;">Open in new tab ↗</a>
        </div>
      </header>
      <iframe id="preview" title="Output preview"></iframe>
      <div id="empty-state" class="empty">
        <div style="font-weight: 700;">Pick a report or graph on the left</div>
        <div>HTML previews will load here without leaving the page.</div>
      </div>
    </div>
  </main>

  <script>
    const preview = document.getElementById('preview');
    const currentFile = document.getElementById('current-file');
    const openTab = document.getElementById('open-tab');
    const fileMeta = document.getElementById('file-meta');
    const emptyState = document.getElementById('empty-state');
    const lastUpdated = document.getElementById('last-updated');

    const renderList = (containerId, items) => {
      const container = document.getElementById(containerId);
      container.innerHTML = '';
      if (!items || !items.length) {
        container.innerHTML = '<li style="color:var(--muted);font-size:13px;">No HTML files yet.</li>';
        return;
      }
      items.forEach((item) => {
        const li = document.createElement('li');
        const btn = document.createElement('button');
        btn.innerHTML = '<div><div style="font-weight:700;">' + item.pretty_name + '</div>' +
                        '<div class="file-meta"><span>' + item.name + '</span><span>' + item.modified +
                        '</span><span>' + item.size_kb + ' KB</span></div></div>' +
                        '<div style="color:var(--muted);font-size:12px;">View</div>';
        btn.addEventListener('click', () => selectFile(item));
        li.appendChild(btn);
        container.appendChild(li);
      });
    };

    const loadFiles = async () => {
      lastUpdated.textContent = 'Refreshing...';
      const res = await fetch('/api/files');
      const data = await res.json();
      renderList('recent-list', data.recent);
      renderList('graphs-list', data.graphs);
      renderList('reports-list', data.reports);
      lastUpdated.textContent = 'Updated ' + data.generated_at;
    };

    const selectFile = (item) => {
      const url = '/view?path=' + encodeURIComponent(item.path);
      preview.src = url;
      currentFile.textContent = item.pretty_name;
      fileMeta.textContent = item.modified + ' · ' + item.size_kb + ' KB';
      openTab.href = url;
      openTab.style.display = 'inline';
      emptyState.style.display = 'none';
    };

    document.getElementById('refresh-btn').addEventListener('click', loadFiles);
    loadFiles();
  </script>
</body>
</html>
"""


@app.get("/")
def index():
    """Serve the viewer UI."""
    return render_template_string(INDEX_HTML, output_root=str(OUTPUT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Launch local HTML output viewer")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5050, help="Port to bind (default: 5050)")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
