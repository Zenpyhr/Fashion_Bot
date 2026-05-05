const form = document.querySelector("#recommend-form");
const input = document.querySelector("#query");
const statusEl = document.querySelector("#status");
const resultsEl = document.querySelector("#results");
const promptButtons = document.querySelectorAll("[data-prompt]");

const fallbackImage = "/static/placeholder.svg";

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
  const response = await fetch("/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_query: query, use_owned_only: false }),
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with status ${response.status}`);
  }

  return response.json();
}

async function submitQuery(query) {
  const trimmedQuery = query.trim();
  if (!trimmedQuery) {
    input.focus();
    return;
  }

  setStatus("Styling options...");
  resultsEl.innerHTML = '<div class="empty-state">Building outfit cards from the demo catalog.</div>';

  try {
    const data = await fetchRecommendations(trimmedQuery);
    renderResults(data.outfits);
    setStatus(`${data.outfits.length} outfits returned`);
  } catch (error) {
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

submitQuery(input.value);
