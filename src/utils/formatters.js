export const normalizeText = (value) =>
  String(value || "").toLowerCase().trim();

export const containsText = (source, query) => {
  if (!query) return true;
  return normalizeText(source).includes(query);
};
