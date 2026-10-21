import React from 'react';
import { Zap, Droplets } from 'lucide-react';

interface SubHeaderToolbarProps {
  utilityType?: 'energy' | 'water';
  setUtilityType?: (type: 'energy' | 'water') => void;
}

export const SubHeaderToolbar: React.FC<SubHeaderToolbarProps> = ({
  utilityType = 'energy',
  setUtilityType,
}) => {
  return (
    <div className="flex items-center justify-end w-full">
      {/* Right Switch Button for Energy and Water */}
      <div className="bg-slate-100 p-1 rounded-xl flex items-center border border-slate-200 shrink-0 shadow-2xs">
        <button
          onClick={() => setUtilityType?.('energy')}
          className={`flex items-center gap-2.5 px-5 py-2.5 text-sm font-bold rounded-lg transition-all cursor-pointer ${
            utilityType === 'energy'
              ? 'bg-white text-[#1f3bb3] shadow-xs'
              : 'text-slate-600 hover:text-slate-900 font-semibold'
          }`}
          id="switch-energy-btn"
        >
          <Zap className="w-4 h-4" />
          <span>Energy</span>
        </button>

        <button
          onClick={() => setUtilityType?.('water')}
          className={`flex items-center gap-2.5 px-5 py-2.5 text-sm font-bold rounded-lg transition-all cursor-pointer ${
            utilityType === 'water'
              ? 'bg-white text-[#1f3bb3] shadow-xs'
              : 'text-slate-600 hover:text-slate-900 font-semibold'
          }`}
          id="switch-water-btn"
        >
          <Droplets className="w-4 h-4" />
          <span>Water</span>
        </button>
      </div>
    </div>
  );
};
