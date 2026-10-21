import React from 'react';
import { Zap, Droplets } from 'lucide-react';

interface StatusSummaryCardProps {
  utilityType?: 'energy' | 'water';
  /** Real ambient weather, shown on both domains. null while loading. */
  weather?: { tempC: number; humidity: number; condition: string } | null;
  /** Real marginal tariff rate — the rate that applies to the NEXT unit,
   * given this billing period's cumulative usage so far. */
  rate?: number | null;
}

export const StatusSummaryCard: React.FC<StatusSummaryCardProps> = ({
  utilityType = 'energy',
  weather = null,
  rate = null,
}) => {
  const isWater = utilityType === 'water';

  return (
    <div className="flex flex-col gap-3.5 h-full justify-between">
      {/* Top Royal Blue Weather Card — real data, shown on both domains. */}
      <div className="bg-[#1f3bb3] text-white rounded-lg p-4 lg:p-4.5 shadow-2xs relative overflow-hidden flex flex-col justify-between flex-1 min-h-[145px] lg:min-h-[155px]">
          {/* Card Header & Values */}
          <div className="z-10">
            <div className="flex items-center justify-between mb-1.5">
              <h3 className="text-xs font-semibold text-white/80 tracking-wide uppercase">
                Weather
              </h3>
            </div>
            <div className="flex items-baseline gap-2">
              <div className="text-3xl lg:text-4xl font-extrabold tracking-tight text-white">
                {weather ? `${weather.tempC}°C` : '—'}
              </div>
            </div>
            <p className="text-xs text-blue-100 font-medium mt-1">
              {weather ? `${weather.condition} • Outdoor Temp` : 'Outdoor Temp'}
            </p>
          </div>

          {/* Weather Conditions Row */}
          <div className="z-10 flex items-center justify-between text-xs text-blue-100 mt-2.5">
            <div className="flex items-center gap-1.5">
              <Droplets className="w-4 h-4 text-blue-300" />
              <span>Humidity {weather ? `${Math.round(weather.humidity)}%` : '—'}</span>
            </div>
          </div>

          {/* Sparkline Wave Graphic */}
          <div className="absolute inset-0 pointer-events-none flex items-end justify-end overflow-hidden opacity-30">
            <svg
              className="w-full h-28 text-[#38bdf8]"
              viewBox="0 0 300 100"
              fill="none"
              preserveAspectRatio="none"
            >
              <path
                d="M 0 60 C 50 10, 100 90, 150 40 C 200 -10, 250 80, 300 30"
                stroke="currentColor"
                strokeWidth="3.5"
                strokeLinecap="round"
              />
            </svg>
          </div>
        </div>

      {/* Bottom Utility Rate Card */}
      <div className="bg-white border border-slate-100 rounded-lg p-4 lg:p-4.5 shadow-2xs flex items-center justify-between gap-4">
        {/* Icon Container */}
        <div
          className={`w-12 h-12 shrink-0 rounded-xl flex items-center justify-center border ${
            isWater
              ? 'bg-blue-50 border-blue-100'
              : 'bg-amber-50 border-amber-100'
          }`}
        >
          {isWater ? (
            <Droplets className="w-6 h-6 text-blue-500 fill-blue-500" />
          ) : (
            <Zap className="w-6 h-6 text-amber-500 fill-amber-500" />
          )}
        </div>

        <div className="flex-1 text-left">
          <span className="text-xs text-slate-400 block font-medium mb-0.5">
            {isWater ? 'Water Rate' : 'Electricity Rate'}
          </span>
          <span className="text-base lg:text-lg font-bold text-slate-800">
            {rate !== null && rate !== undefined
              ? isWater ? rate.toFixed(5) : rate.toFixed(3)
              : '—'} DT{' '}
            <span className="text-xs font-normal text-slate-500">
              / {isWater ? 'L' : 'kWh'}
            </span>
          </span>
        </div>
      </div>
    </div>
  );
};
