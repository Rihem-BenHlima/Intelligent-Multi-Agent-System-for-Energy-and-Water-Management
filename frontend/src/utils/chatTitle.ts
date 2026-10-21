// Derives a conversation title from the user's first message.

const MAX_TITLE_LEN = 40;

export function deriveTitleFromMessage(text: string): string {
  const trimmed = text.trim();
  if (trimmed.length <= MAX_TITLE_LEN) return trimmed;
  // Cut at the last whole word within the limit rather than mid-word.
  const cut = trimmed.slice(0, MAX_TITLE_LEN);
  const lastSpace = cut.lastIndexOf(' ');
  return `${(lastSpace > 10 ? cut.slice(0, lastSpace) : cut).trim()}…`;
}
