const $ = (id) => document.getElementById(id);

// --- HTTP helper -----------------------------------------------------------

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!response.ok) {
    let detail;
    try {
      const body = await response.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      detail = await response.text();
    }
    throw new Error(`${response.status}: ${detail}`);
  }

  return response.json();
}

// --- Utilities -------------------------------------------------------------

function escapeHtml(value) {
  return String(value).replace(
    /[&<>"']/g,
    (ch) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      })[ch],
  );
}

// --- Repos -----------------------------------------------------------------

async function loadRepos() {
  const list = $("repos-list");
  const select = $("query-repo");
  const askBtn = $("ask-btn");

  try {
    const { repos } = await api("/repos");

    if (repos.length === 0) {
      list.innerHTML = "<li>No repositories indexed yet.</li>";
      select.innerHTML = '<option value="">No repos indexed</option>';
      select.disabled = true;
      askBtn.disabled = true;
      return;
    }

    list.innerHTML = repos.map((r) => `<li>${escapeHtml(r)}</li>`).join("");
    select.innerHTML = repos
      .map((r) => `<option value="${escapeHtml(r)}">${escapeHtml(r)}</option>`)
      .join("");
    select.disabled = false;
    askBtn.disabled = false;
  } catch (error) {
    list.innerHTML = `<li>Error: ${escapeHtml(error.message)}</li>`;
  }
}

// --- Ingest ----------------------------------------------------------------

async function ingestRepo() {
  const url = $("repo-url").value.trim();
  if (!url) return;

  const status = $("ingest-status");
  const button = $("ingest-btn");

  button.disabled = true;
  status.className = "status loading";
  status.textContent = "Cloning, chunking, embedding… (this can take a minute)";

  try {
    const result = await api("/ingest", {
      method: "POST",
      body: JSON.stringify({ repo_url: url }),
    });
    status.className = "status success";
    status.textContent =
      `Ingested ${result.repo_name}: ` +
      `${result.files_indexed} files, ${result.chunks_created} chunks.`;
    $("repo-url").value = "";
    await loadRepos();
  } catch (error) {
    status.className = "status error";
    status.textContent = `Error: ${error.message}`;
  } finally {
    button.disabled = false;
  }
}

// --- Query -----------------------------------------------------------------

async function askQuestion() {
  const question = $("question").value.trim();
  const repo = $("query-repo").value;
  if (!question || !repo) return;

  const button = $("ask-btn");
  const result = $("result");
  const answer = $("answer");
  const citations = $("citations");

  button.disabled = true;
  result.hidden = false;
  answer.textContent = "Thinking…";
  citations.innerHTML = "";

  try {
    const data = await api("/query", {
      method: "POST",
      body: JSON.stringify({ question, repo_name: repo }),
    });

    answer.textContent = data.answer;

    citations.innerHTML = data.citations.length
      ? data.citations
          .map((c) => {
            const name = c.name ? `<span class="citation-name">(${escapeHtml(c.name)})</span>` : "";
            return `<li><code>${escapeHtml(c.file_path)}:${c.start_line}-${c.end_line}</code>${name}</li>`;
          })
          .join("")
      : "<li>No citations returned.</li>";
  } catch (error) {
    answer.textContent = `Error: ${error.message}`;
  } finally {
    button.disabled = false;
  }
}

// --- Wiring ----------------------------------------------------------------

$("ingest-btn").addEventListener("click", ingestRepo);

$("ask-btn").addEventListener("click", askQuestion);

$("repo-url").addEventListener("keydown", (event) => {
  if (event.key === "Enter") ingestRepo();
});

$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    askQuestion();
  }
});

loadRepos();
