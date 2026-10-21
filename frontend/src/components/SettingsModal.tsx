import React, { useRef, useState } from 'react';
import { Settings, X, Check, Upload, Trash2 } from 'lucide-react';
import { SIDEBAR_PRESETS } from '../useSidebarColor';
import type { Profile } from '../useProfile';
import { Avatar } from './Avatar';

interface SettingsModalProps {
  isOpen?: boolean;
  setIsOpen?: (open: boolean) => void;
  sidebarColor: string;
  setSidebarColor: (color: string) => void;
  profile: Profile;
  updateProfile: (patch: Partial<Profile>) => void;
}

const MAX_PHOTO_BYTES = 1_500_000; // ~1.5 MB before base64 inflation

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen: externalIsOpen,
  setIsOpen: externalSetIsOpen,
  sidebarColor,
  setSidebarColor,
  profile,
  updateProfile,
}) => {
  const [internalIsOpen, setInternalIsOpen] = useState(false);
  const isOpen = externalIsOpen !== undefined ? externalIsOpen : internalIsOpen;
  const setIsOpen = externalSetIsOpen || setInternalIsOpen;

  const fileRef = useRef<HTMLInputElement>(null);
  const [photoError, setPhotoError] = useState<string | null>(null);

  const onPickPhoto = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = ''; // allow re-picking the same file
    if (!file) return;
    if (!file.type.startsWith('image/')) { setPhotoError('Please choose an image file.'); return; }
    if (file.size > MAX_PHOTO_BYTES) { setPhotoError('Image is too large (max ~1.5 MB).'); return; }
    setPhotoError(null);
    const reader = new FileReader();
    reader.onload = () => updateProfile({ photo: String(reader.result) });
    reader.onerror = () => setPhotoError('Could not read that file.');
    reader.readAsDataURL(file);
  };

  return (
    <>
      {/* Slide-over Drawer / Modal */}
      {isOpen && (
        <div className="fixed inset-0 z-50 overflow-hidden">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/40 transition-opacity"
            onClick={() => setIsOpen(false)}
          />

          <div className="fixed inset-y-0 right-0 max-w-full flex pl-10">
            <div className="w-screen max-w-sm bg-white shadow-2xl border-l border-slate-200 p-6 flex flex-col justify-between overflow-y-auto">
              <div>
                <div className="flex items-center justify-between pb-4 border-b border-slate-200">
                  <div className="flex items-center gap-2">
                    <Settings className="w-5 h-5 text-[#1f3bb3]" />
                    <h3 className="text-base font-bold text-slate-800">
                      Settings &amp; Customization
                    </h3>
                  </div>
                  <button
                    onClick={() => setIsOpen(false)}
                    className="p-1 rounded text-slate-400 hover:text-slate-600 hover:bg-slate-100"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>

                <div className="py-6 space-y-7">
                  {/* ── Profile ─────────────────────────────────────────── */}
                  <div>
                    <label className="text-xs font-bold text-slate-500 uppercase tracking-wider block mb-3">
                      Profile
                    </label>
                    <div className="flex items-center gap-4">
                      <Avatar name={profile.name} photo={profile.photo} size={56} />
                      <div className="flex flex-col gap-2">
                        <input
                          ref={fileRef}
                          type="file"
                          accept="image/*"
                          className="hidden"
                          onChange={onPickPhoto}
                        />
                        <button
                          onClick={() => fileRef.current?.click()}
                          className="inline-flex items-center gap-1.5 text-xs font-semibold text-[#1f3bb3] hover:text-[#183096]"
                        >
                          <Upload className="w-3.5 h-3.5" />
                          {profile.photo ? 'Change photo' : 'Upload photo'}
                        </button>
                        {profile.photo && (
                          <button
                            onClick={() => updateProfile({ photo: null })}
                            className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-400 hover:text-rose-500"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                            Remove
                          </button>
                        )}
                      </div>
                    </div>
                    {photoError && (
                      <p className="text-[11px] text-rose-500 mt-2">{photoError}</p>
                    )}

                    <div className="mt-4 space-y-3">
                      <div>
                        <label className="text-[11px] font-medium text-slate-500 block mb-1">Display name</label>
                        <input
                          type="text"
                          value={profile.name}
                          onChange={(e) => updateProfile({ name: e.target.value })}
                          placeholder="Your name"
                          className="w-full text-sm px-3 py-2 rounded-lg border border-slate-200 focus:border-[#1f3bb3] focus:outline-none focus:ring-1 focus:ring-[#1f3bb3] text-slate-800"
                        />
                      </div>
                      <div>
                        <label className="text-[11px] font-medium text-slate-500 block mb-1">Email</label>
                        <input
                          type="email"
                          value={profile.email}
                          onChange={(e) => updateProfile({ email: e.target.value })}
                          placeholder="you@example.com"
                          className="w-full text-sm px-3 py-2 rounded-lg border border-slate-200 focus:border-[#1f3bb3] focus:outline-none focus:ring-1 focus:ring-[#1f3bb3] text-slate-800"
                        />
                      </div>
                    </div>
                  </div>

                  {/* ── Sidebar colour ──────────────────────────────────── */}
                  <div>
                    <label className="text-xs font-bold text-slate-500 uppercase tracking-wider block mb-1">
                      Sidebar Color
                    </label>
                    <p className="text-xs text-slate-400 mb-3">
                      Pick the navigation bar's background.
                    </p>
                    <div className="flex items-center flex-wrap gap-3">
                      {SIDEBAR_PRESETS.map((preset) => (
                        <button
                          key={preset.color}
                          onClick={() => setSidebarColor(preset.color)}
                          className={`w-9 h-9 rounded-full flex items-center justify-center shadow-xs transition-all ${
                            sidebarColor === preset.color
                              ? 'ring-2 ring-offset-2 ring-[#1f3bb3]'
                              : 'ring-1 ring-slate-200 hover:ring-slate-300'
                          }`}
                          style={{ backgroundColor: preset.color }}
                          title={preset.name}
                          aria-label={preset.name}
                          aria-pressed={sidebarColor === preset.color}
                        >
                          {sidebarColor === preset.color && (
                            <Check className="w-4 h-4 text-white" />
                          )}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              <p className="text-[11px] text-slate-400 pt-4 border-t border-slate-100">
                Preferences are saved on this device.
              </p>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
