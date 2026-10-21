import { useEffect, useState } from 'react';

export const SIDEBAR_PRESETS = [
  { name: 'Charcoal', color: '#181b23' },
  { name: 'Midnight', color: '#0f172a' },
  { name: 'Deep Royal', color: '#111827' },
  { name: 'Ocean', color: '#0c2233' },
  { name: 'Forest', color: '#12211a' },
  { name: 'Plum', color: '#1e1424' },
] as const;

const STORAGE_KEY = 'sfm-sidebar-color';
const DEFAULT = SIDEBAR_PRESETS[0].color;

function read(): string {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v && /^#[0-9a-fA-F]{6}$/.test(v)) return v;
  } catch { /* ignore */ }
  return DEFAULT;
}

/** Sidebar background colour — user-chosen, persisted. Lift once in App and
 *  pass the value to <Sidebar> and <SettingsModal>. */
export function useSidebarColor() {
  const [sidebarColor, setColor] = useState<string>(read);

  const setSidebarColor = (c: string) => {
    setColor(c);
    try { localStorage.setItem(STORAGE_KEY, c); } catch { /* ignore */ }
  };

  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, sidebarColor); } catch { /* ignore */ }
  }, [sidebarColor]);

  return { sidebarColor, setSidebarColor } as const;
}
