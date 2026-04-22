/* Document Summarizer UI.
 *
 * Flow:
 *   1. User picks a document file (.txt, .md, .pdf).
 *   2. Form submits multipart/form-data to /ui/summarize.
 *   3. Server saves the file, invokes the LangGraph summarizer, returns JSON
 *      containing both the extracted raw text and the final summary.
 *   4. We render each as markdown in its own tab.
 *   5. The "Download .md" button saves the raw markdown of the active tab.
 */

(() => {
  const form = document.getElementById("upload-form");
  const fileInput = document.getElementById("document");
  const fileName = document.getElementById("file-name");
  const submitBtn = document.getElementById("submit");
  const statusEl = document.getElementById("status");
  const resultEl = document.getElementById("result");
  const resultFilename = document.getElementById("result-filename");
  const resultMeta = document.getElementById("result-meta");
  const panelSummary = document.getElementById("panel-summary");
  const panelExtracted = document.getElementById("panel-extracted");
  const downloadBtn = document.getElementById("download-md");
  const tabs = document.querySelectorAll(".tab");

  // Raw markdown for each tab is cached here so the download button can grab
  // it without re-parsing the rendered HTML.
  const rawMarkdown = {
    "panel-summary": "",
    "panel-extracted": "",
  };
  let activePanel = "panel-summary";
  let sourceFilename = "";

  function show(el) { el.hidden = false; }
  function hide(el) { el.hidden = true; }

  function setStatus(kind, text) {
    statusEl.className = `status ${kind}`;
    statusEl.textContent = text;
    show(statusEl);
  }

  function renderMarkdown(target, text) {
    if (window.marked && typeof window.marked.parse === "function") {
      target.innerHTML = window.marked.parse(text || "");
    } else {
      target.textContent = text || "";
    }
  }

  function activateTab(tab) {
    tabs.forEach((t) => {
      const active = t === tab;
      t.classList.toggle("active", active);
      const panel = document.getElementById(t.dataset.panel);
      if (!panel) return;
      panel.classList.toggle("active", active);
      if (active) {
        show(panel);
        activePanel = t.dataset.panel;
      } else {
        hide(panel);
      }
    });
    updateDownloadLabel();
  }

  function updateDownloadLabel() {
    const which = activePanel === "panel-summary" ? "summary" : "extracted";
    downloadBtn.textContent = `\u2193 Download ${which}.md`;
    downloadBtn.disabled = !rawMarkdown[activePanel];
  }

  function downloadActive() {
    const md = rawMarkdown[activePanel];
    if (!md) return;

    const base = (sourceFilename || "document").replace(/\.[^.]+$/, "");
    const suffix = activePanel === "panel-summary" ? "summary" : "extracted";
    const outName = `${base}.${suffix}.md`;

    const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = outName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  tabs.forEach((t) => t.addEventListener("click", () => activateTab(t)));
  downloadBtn.addEventListener("click", downloadActive);

  fileInput.addEventListener("change", () => {
    const f = fileInput.files && fileInput.files[0];
    fileName.textContent = f ? f.name : "";
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = fileInput.files && fileInput.files[0];
    if (!file) return;

    hide(resultEl);
    setStatus("loading", `Summarizing ${file.name}…`);
    submitBtn.disabled = true;

    const body = new FormData();
    body.append("document", file);

    try {
      const resp = await fetch("/ui/summarize", {
        method: "POST",
        credentials: "same-origin",
        body,
      });
      const payload = await resp.json().catch(() => ({}));

      if (!resp.ok) {
        throw new Error(payload.error || `HTTP ${resp.status}`);
      }

      sourceFilename = payload.filename || file.name;
      resultFilename.textContent = sourceFilename;
      const meta = payload.metadata || {};
      const bits = [];
      if (meta.char_count) bits.push(`${meta.char_count.toLocaleString()} chars`);
      if (payload.chunk_count) bits.push(`${payload.chunk_count} chunks`);
      resultMeta.textContent = bits.join(" · ");

      rawMarkdown["panel-summary"] = payload.summary || "";
      rawMarkdown["panel-extracted"] = payload.extracted_text || "";

      renderMarkdown(panelSummary, rawMarkdown["panel-summary"] || "_(no summary returned)_");
      renderMarkdown(panelExtracted, rawMarkdown["panel-extracted"] || "_(no extracted text)_");

      activateTab(document.querySelector('.tab[data-panel="panel-summary"]'));
      show(resultEl);
      hide(statusEl);
    } catch (err) {
      console.error(err);
      setStatus("error", `Error: ${err.message || err}`);
    } finally {
      submitBtn.disabled = false;
    }
  });
})();
