export const createCard = (item) => {
  const message = document.createElement("div");
  message.className = "message";

  const authorEl = document.createElement("div");
  authorEl.className = "message__author";
  authorEl.textContent = item.author;

  const textEl = document.createElement("p");
  textEl.textContent = item.text;

  message.appendChild(authorEl);
  message.appendChild(textEl);

  return message;
};
