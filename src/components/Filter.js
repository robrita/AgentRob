import { renderData } from "./Grid.js";
import { containsText, normalizeText } from "../utils/formatters.js";

export const Filters = () => `
  <label class="transcript__search">
    <span class="visually-hidden">Search transcript</span>
    <input
      id="transcriptSearch"
      type="search"
      placeholder="Search transcript"
      aria-label="Search transcript"
    />
  </label>
`;

export const setupFilters = ({ data, onFiltered }) => {
  const searchInput = document.getElementById("transcriptSearch");
  const container = document.getElementById("transcriptBody");
  const apply = () => {
    const query = normalizeText(searchInput?.value);
    const filtered = data.filter((item) => {
      if (!query) return true;
      return (
        containsText(item.author, query) ||
        containsText(item.text, query)
      );
    });

    if (typeof onFiltered === "function") {
      onFiltered(filtered);
      return;
    }

    renderData(filtered, container);
  };

  if (!searchInput) {
    const fallback = () => {
      if (typeof onFiltered === "function") {
        onFiltered(data);
        return;
      }

      renderData(data, container);
    };

    return { applyFilters: fallback };
  }

  searchInput.addEventListener("input", apply);
  apply();

  return { applyFilters: apply };
};
