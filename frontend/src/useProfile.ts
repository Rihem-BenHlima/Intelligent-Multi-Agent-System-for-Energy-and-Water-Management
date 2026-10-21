import { useEffect, useState, useCallback } from 'react';

export interface Profile {
  name: string;
  email: string;
  photo: string | null; // data-URL, or null for an initials avatar
}

const STORAGE_KEY = 'sfm-profile';
const DEFAULT: Profile = {
  name: 'User',
  email: 'user@example.com',
  photo: null,
};

function read(): Profile {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const p = JSON.parse(raw);
      return {
        name: typeof p.name === 'string' && p.name.trim() ? p.name : DEFAULT.name,
        email: typeof p.email === 'string' ? p.email : DEFAULT.email,
        photo: typeof p.photo === 'string' && p.photo.startsWith('data:') ? p.photo : null,
      };
    }
  } catch { /* ignore */ }
  return { ...DEFAULT };
}

export function initialsOf(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return '?';
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

/** Initials-disc background — on-brand indigo/blue tints matching the app's
 *  #1f3bb3 primary. Slight per-name variation so two users aren't identical. */
export function avatarColor(name: string): string {
  const palette = ['#1f3bb3', '#183096', '#2f4fc9', '#3b3f8f', '#1e40af'];
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return palette[h % palette.length];
}

/** User profile — display name / email / photo, persisted per device.
 *  Lift once in App and pass to <Header> and <SettingsModal>. */
export function useProfile() {
  const [profile, setProfile] = useState<Profile>(read);

  const update = useCallback((patch: Partial<Profile>) => {
    setProfile((prev) => ({ ...prev, ...patch }));
  }, []);

  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(profile)); } catch { /* ignore */ }
  }, [profile]);

  return { profile, update } as const;
}
