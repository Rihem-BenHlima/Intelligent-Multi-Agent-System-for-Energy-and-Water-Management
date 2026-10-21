import React from 'react';
import { initialsOf, avatarColor } from '../useProfile';

interface AvatarProps {
  name: string;
  photo?: string | null;
  size?: number;      // px
  className?: string;
}

/** Circular user avatar — shows the uploaded photo if set, otherwise a
 *  coloured initials disc. No external image dependency. */
export const Avatar: React.FC<AvatarProps> = ({ name, photo, size = 32, className = '' }) => {
  const ring = 'ring-2 ring-slate-200';
  if (photo) {
    return (
      <img
        src={photo}
        alt={name}
        style={{ width: size, height: size }}
        className={`rounded-full object-cover ${ring} ${className}`}
      />
    );
  }
  return (
    <span
      aria-label={name}
      style={{
        width: size,
        height: size,
        backgroundColor: avatarColor(name),
        fontSize: Math.round(size * 0.4),
      }}
      className={`rounded-full ${ring} flex items-center justify-center font-semibold text-white select-none ${className}`}
    >
      {initialsOf(name)}
    </span>
  );
};
