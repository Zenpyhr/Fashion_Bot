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

const fallbackImage = "/static/placeholder.svg";
let recommendAbortController = null;

function setStatus(message) {
  statusEl.textContent = message;
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
