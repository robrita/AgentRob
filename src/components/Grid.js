import { createCard } from "./Card.js";

export const renderData = (data, container) => {
  if (!container) return [];
  container.innerHTML = "";
  const nodes = data.map((item) => createCard(item));
  nodes.forEach((node) => container.appendChild(node));
  container.scrollTop = container.scrollHeight;
  return nodes;
};
