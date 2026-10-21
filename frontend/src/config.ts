/**
 * Frontend runtime configuration.
 * MODIFY: set VITE_API_URL in .env to the backend URL.
 */
export const API_BASE: string =
  (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/+$/, '') ||
  'http://localhost:8000';
