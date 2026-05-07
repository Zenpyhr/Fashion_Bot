const form = document.querySelector("#recommend-form");
const input = document.querySelector("#query");
const statusEl = document.querySelector("#status");
const resultsEl = document.querySelector("#results");
const promptButtons = document.querySelectorAll("[data-prompt]");
const wardrobeForm = document.querySelector("#wardrobe-upload-form");
const wardrobeUploadBtn = document.querySelector("#wardrobe-upload-btn");
const wardrobeUserIdInput = document.querySelector("#wardrobe-user-id");
const wardrobeImageInput = document.querySelector("#wardrobe-image");
const wardrobeStatusEl = document.querySelector("#wardrobe-status");
const wardrobeClearBtn = document.querySelector("#wardrobe-clear");
const pageTabs = document.querySelectorAll("[data-page-target]");
const pageViews = document.querySelectorAll(".page-view");

const qaForm = document.querySelector("#qa-form");
const qaQuestionInput = document.querySelector("#qa-question");
const qaStatusEl = document.querySelector("#qa-status");
const qaChatLogEl = document.querySelector("#qa-chat-log");

const fallbackImage = "/static/placeholder.svg";
let recommendAbortController = null;
let qaAbortController = null;

function setStatus(message) {
  statusEl.textContent = message;
}

function setQaStatus(message) {
  if (qaStatusEl) {
    qaStatusEl.textContent = message;
  }
}

function titleCase(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function itemMarkup(item) {
  const imageUrl = item.image_url || fallbackImage;
  const name = item.display_name || titleCase(item.normalized_category);
  const detail = `${titleCase(item.normalized_category)} · ${titleCase(item.normalized_color)}`;

  return `
    <article class="item">
      <img class="item-image" src="${imageUrl}" alt="${name}" loading="lazy" />
      <div class="item-meta">
        <p class="role">${titleCase(item.recommendation_role)}</p>
        <p class="name">${name}</p>
        <p class="detail">${detail}</p>
      </div>
    </article>
  `;
}

function outfitMarkup(outfit, index) {
  const items = outfit.items || [];
  return `
    <article class="outfit-card">
      <div class="outfit-header">
        <h2>Outfit ${index + 1}</h2>
        <span class="score">Score ${outfit.score}</span>
      </div>
      <div class="item-grid">
        ${items.map(itemMarkup).join("")}
      </div>
      <p class="explanation">${outfit.explanation || "A complete outfit suggestion from the catalog."}</p>
    </article>
  `;
}

function renderResults(outfits) {
  if (!outfits || outfits.length === 0) {
    resultsEl.innerHTML = '<div class="empty-state">No complete outfits came back for this query.</div>';
    return;
  }

  resultsEl.innerHTML = outfits.map(outfitMarkup).join("");
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatQaInline(text) {
  let content = escapeHtml(text || "");
  content = content.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  content = content.replace(
    /(https?:\/\/[^\s)]+)/g,
    '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>',
  );
  content = content.replace(
    /\[Source\s+(\d+)\]/gi,
    '<span class="qa-citation">[Source $1]</span>',
  );
  return content;
}

function formatQaParagraphs(text) {
  const blocks = String(text || "")
    .split(/\n{2,}/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (blocks.length === 0) {
    return "";
  }

  return blocks
    .map((block) => `<p>${formatQaInline(block.replace(/\n/g, " "))}</p>`)
    .join("");
}

function parseQaSections(answer) {
  const lines = String(answer || "").split("\n");
  const sections = {};
  let current = "answer";
  sections[current] = [];

  for (const line of lines) {
    const headingMatch = line.match(/^###\s+(.+?)\s*$/);
    if (headingMatch) {
      const key = headingMatch[1].trim().toLowerCase();
      current = key;
      if (!sections[current]) {
        sections[current] = [];
      }
      continue;
    }
    sections[current].push(line);
  }

  const getSection = (name) => (sections[name] || []).join("\n").trim();
  return {
    answer: getSection("answer"),
    keyTrends: getSection("key trends"),
    evidence: getSection("evidence"),
    sources: getSection("sources"),
  };
}

function parseKeyTrendBullets(sectionText) {
  return String(sectionText || "")
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.startsWith("- "))
    .map((line) => line.slice(2).trim())
    .filter(Boolean);
}

function sourceListMarkup(payload) {
  const sources = Array.isArray(payload?.sources) ? payload.sources : [];
  if (sources.length === 0) {
    return "";
  }

  const deduped = [];
  const seen = new Set();

  for (const source of sources) {
    const key = `${source?.title || ""}::${source?.url || ""}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    deduped.push(source);
  }

  return `
    <ol class="qa-source-list">
      ${deduped
        .map((source, index) => {
          const title = escapeHtml(source?.title || `Source ${index + 1}`);
          const url = source?.url ? String(source.url).trim() : "";
          if (!url) {
            return `<li>${title}</li>`;
          }
          return `<li><a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${title}</a></li>`;
        })
        .join("")}
    </ol>
  `;
}

function isInsufficientEvidenceResponse(answerText, sections) {
  const normalized = `${answerText}\n${sections.answer}\n${sections.keyTrends}\n${sections.evidence}`
    .toLowerCase()
    .replace(/\s+/g, " ");
  return (
    normalized.includes("i do not have enough reliable evidence in the retrieved sources") ||
    normalized.includes("no supported trend can be concluded from the retrieved sources") ||
    normalized.includes("does not contain enough direct, relevant evidence for the user question") ||
    normalized.includes("retrieved sources are insufficient to support a direct answer to this question")
  );
}

function buildQaAssistantMarkup(payload) {
  const answerText = String(payload?.answer || "").trim();
  const sections = parseQaSections(answerText);
  const trendBullets = parseKeyTrendBullets(sections.keyTrends);
  const citations = Array.isArray(payload?.citations) ? payload.citations : [];
  const hideSources = isInsufficientEvidenceResponse(answerText, sections);

  const answerMarkup = sections.answer
    ? `<section class="qa-block"><h3>Answer</h3>${formatQaParagraphs(sections.answer)}</section>`
    : "";

  const trendMarkup = trendBullets.length
    ? `
      <section class="qa-block">
        <h3>Key Trends</h3>
        <ul>
          ${trendBullets.map((trend) => `<li>${formatQaInline(trend)}</li>`).join("")}
        </ul>
      </section>
    `
    : sections.keyTrends
      ? `<section class="qa-block"><h3>Key Trends</h3>${formatQaParagraphs(sections.keyTrends)}</section>`
      : "";

  const evidenceMarkup = sections.evidence
    ? `<section class="qa-block"><h3>Evidence</h3>${formatQaParagraphs(sections.evidence)}</section>`
    : "";

  const sourcesFromPayload = sourceListMarkup(payload);
  const sourcesMarkup = hideSources
    ? ""
    : sourcesFromPayload
      ? `<section class="qa-block"><h3>Sources</h3>${sourcesFromPayload}</section>`
      : sections.sources
        ? `<section class="qa-block"><h3>Sources</h3>${formatQaParagraphs(sections.sources)}</section>`
        : "";

  const citationMarkup = !hideSources && citations.length
    ? `<p class="qa-citation-row">${citations.map((item) => `<span class="qa-citation">${escapeHtml(item)}</span>`).join(" ")}</p>`
    : "";

  return `
    <div class="qa-structured-answer">
      ${answerMarkup}
      ${trendMarkup}
      ${evidenceMarkup}
      ${sourcesMarkup}
      ${citationMarkup}
    </div>
  `;
}

function ensureChatReady() {
  if (!qaChatLogEl) {
    return;
  }
  const emptyState = qaChatLogEl.querySelector(".chat-empty");
  if (emptyState) {
    emptyState.remove();
  }
}

function appendChatMessage(role, innerMarkup) {
  if (!qaChatLogEl) {
    return;
  }
  ensureChatReady();
  const item = document.createElement("article");
  item.className = `chat-bubble ${role}`;
  item.innerHTML = innerMarkup;
  qaChatLogEl.appendChild(item);
  qaChatLogEl.scrollTop = qaChatLogEl.scrollHeight;
}

async function fetchRecommendations(query) {
  if (recommendAbortController) {
    recommendAbortController.abort();
  }
  recommendAbortController = new AbortController();

  const response = await fetch("/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal: recommendAbortController.signal,
    body: JSON.stringify({
      user_query: query,
      use_owned_only: false,
      user_id: wardrobeUserIdInput?.value?.trim() || null,
    }),
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with status ${response.status}`);
  }

  return response.json();
}

async function fetchQaAnswer(question) {
  if (qaAbortController) {
    qaAbortController.abort();
  }
  qaAbortController = new AbortController();

  const response = await fetch("/qa", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal: qaAbortController.signal,
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;
    try {
      const payload = await response.json();
      if (payload?.detail) {
        message = payload.detail;
      }
    } catch {
      const text = await response.text();
      if (text) {
        message = text;
      }
    }
    throw new Error(message);
  }

  return response.json();
}

async function uploadWardrobeImage(userId, file) {
  const body = new FormData();
  body.append("user_id", userId);
  body.append("image", file);

  const response = await fetch("/wardrobe/upload", {
    method: "POST",
    body,
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Upload failed with status ${response.status}`);
  }

  return response.json();
}

async function clearWardrobe(userId) {
  const body = new FormData();
  body.append("user_id", userId);

  const response = await fetch("/wardrobe/clear", { method: "POST", body });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Clear failed with status ${response.status}`);
  }
  return response.json();
}

async function submitQuery(query) {
  const trimmedQuery = query.trim();
  if (!trimmedQuery) {
    input.focus();
    return;
  }

  setStatus("Styling options…");
  resultsEl.innerHTML = '<div class="empty-state">Building outfit cards from the demo catalog.</div>';

  try {
    const data = await fetchRecommendations(trimmedQuery);
    renderResults(data.outfits);
    const wardrobeStatus = data?.parsed_constraints?.wardrobe_status;
    if (wardrobeStatus && wardrobeStatus !== "ok") {
      setStatus(`Wardrobe: ${wardrobeStatus}`);
    } else {
      setStatus(`${data.outfits.length} outfits returned`);
    }
  } catch (error) {
    if (error?.name === "AbortError") {
      // A newer request replaced this one; do not show an error.
      return;
    }
    resultsEl.innerHTML = '<div class="empty-state">The recommendation request failed.</div>';
    setStatus(error.message);
  }
}

async function submitQaQuestion(question) {
  const trimmedQuestion = String(question || "").trim();
  if (!trimmedQuestion) {
    qaQuestionInput?.focus();
    return;
  }

  appendChatMessage("user", `<p>${escapeHtml(trimmedQuestion)}</p>`);
  setQaStatus("Finding sources and drafting answer...");
  if (qaQuestionInput) {
    qaQuestionInput.value = "";
  }

  try {
    const payload = await fetchQaAnswer(trimmedQuestion);
    appendChatMessage("assistant", buildQaAssistantMarkup(payload));
    setQaStatus("Answer ready");
  } catch (error) {
    if (error?.name === "AbortError") {
      return;
    }
    appendChatMessage(
      "assistant",
      `<div class="qa-structured-answer"><section class="qa-block"><h3>Error</h3><p>${escapeHtml(error.message || "QA request failed.")}</p></section></div>`,
    );
    setQaStatus(error.message || "QA request failed.");
  }
}

function showPage(pageId) {
  pageViews.forEach((pageView) => {
    const isMatch = pageView.id === pageId;
    pageView.classList.toggle("is-hidden", !isMatch);
  });

  pageTabs.forEach((tab) => {
    const isActive = tab.dataset.pageTarget === pageId;
    tab.classList.toggle("is-active", isActive);
    tab.setAttribute("aria-selected", isActive ? "true" : "false");
  });
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitQuery(input.value);
});

promptButtons.forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.prompt;
    submitQuery(input.value);
  });
});

// Intentionally do not handle wardrobe form submit: browsers can submit a form
// when the user presses Enter in an input, causing duplicate uploads. We only
// upload when the explicit upload button is clicked.

wardrobeUploadBtn?.addEventListener("click", async (event) => {
  event.preventDefault();
  const userId = wardrobeUserIdInput?.value?.trim();
  const file = wardrobeImageInput?.files?.[0];
  if (!userId || !file) {
    wardrobeStatusEl.textContent = "Provide a user id and pick an image file.";
    setStatus(wardrobeStatusEl.textContent);
    return;
  }

  wardrobeStatusEl.textContent = "Uploading...";
  setStatus("Uploading wardrobe...");
  try {
    const data = await uploadWardrobeImage(userId, file);
    wardrobeStatusEl.textContent = `Uploaded: ${data.wardrobe_item_id}`;
    setStatus(`Wardrobe upload OK: ${data.wardrobe_item_id}`);
    submitQuery(input.value);
  } catch (err) {
    wardrobeStatusEl.textContent = err.message || "Upload failed.";
    setStatus(wardrobeStatusEl.textContent);
  }
});

wardrobeClearBtn?.addEventListener("click", async () => {
  const userId = wardrobeUserIdInput?.value?.trim();
  if (!userId) {
    wardrobeStatusEl.textContent = "Provide a user id.";
    setStatus(wardrobeStatusEl.textContent);
    return;
  }
  wardrobeStatusEl.textContent = "Clearing wardrobe...";
  setStatus("Clearing wardrobe...");
  try {
    const data = await clearWardrobe(userId);
    wardrobeStatusEl.textContent = `Cleared ${data.deleted} items.`;
    setStatus(`Wardrobe cleared: ${data.deleted}`);
    submitQuery(input.value);
  } catch (err) {
    wardrobeStatusEl.textContent = err.message || "Clear failed.";
    setStatus(wardrobeStatusEl.textContent);
  }
});

qaForm?.addEventListener("submit", (event) => {
  event.preventDefault();
  submitQaQuestion(qaQuestionInput?.value || "");
});

pageTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const targetPageId = tab.dataset.pageTarget;
    if (!targetPageId) {
      return;
    }
    showPage(targetPageId);
  });
});
