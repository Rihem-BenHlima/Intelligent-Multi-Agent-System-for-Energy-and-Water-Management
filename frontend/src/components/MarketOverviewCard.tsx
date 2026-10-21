import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';

export interface TimeOfDayBlock { block: string; timeRange: string; main: number | null; compare: number; }

interface MarketOverviewCardProps {
  utilityType?: 'energy' | 'water';
  total: number | null;
  changePct: number | null;
  data: TimeOfDayBlock[];
}

export const MarketOverviewCard: React.FC<MarketOverviewCardProps> = ({
  utilityType = 'energy',
  total,
  changePct,
  data,
}) => {
  const isWater = utilityType === 'water';
  const unit = isWater ? 'L' : 'kWh';
  const isDecrease = changePct !== null && changePct <= 0;
  const maxVal = data.reduce((m, d) => Math.max(m, d.main ?? 0, d.compare), 0);
  // Set Y-axis maximum with baseline floor
  const yMax = Math.max(isWater ? 4 : 2, Math.ceil(maxVal * 1.15));

  return (
    <div className="bg-white border border-slate-100 rounded-lg p-3.5 lg:p-4 shadow-2xs flex flex-col justify-between h-full">
      {/* Top Header & Metrics Row */}
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
        <div>
          <h3 className="text-xs lg:text-sm font-bold text-slate-800">
            Time-of-Day Consumption Blocks
          </h3>
          <p className="text-[11px] text-slate-400 font-normal mt-0.5">
            {isWater
              ? 'Water consumed across 12 daily time blocks'
              : 'Energy consumed across 12 daily time blocks'}
          </p>
        </div>

        {/* Metric Value & Legend */}
        <div className="flex items-center gap-4">
          <div className="flex items-baseline gap-1.5">
            <span className="text-lg lg:text-xl font-bold text-slate-800 tracking-tight">
              {total !== null ? `${total.toLocaleString('en', { maximumFractionDigits: 1 })} ${unit}` : '—'}
            </span>
            {changePct !== null && (
              <span
                className={`text-[11px] font-semibold ${
                  isDecrease ? 'text-emerald-500' : 'text-amber-500'
                }`}
              >
                ({changePct >= 0 ? '+' : ''}{changePct.toFixed(1)}% vs yesterday)
              </span>
            )}
          </div>

          <div className="hidden sm:flex items-center gap-3 text-xs font-medium">
            <div className="flex items-center gap-1.5 text-slate-600">
              <span className="w-2.5 h-2.5 rounded-full bg-[#1f3bb3]" />
              <span>Today ({unit})</span>
            </div>
            <div className="flex items-center gap-1.5 text-slate-600">
              <span className="w-2.5 h-2.5 rounded-full bg-[#cbd5e1]" />
              <span>Yesterday ({unit})</span>
            </div>
          </div>
        </div>
      </div>

      {/* Bar Chart for the 12 Time Blocks */}
      <div className="relative w-full h-28 lg:h-32">
        <div className="absolute top-0 left-0 w-9 text-center z-10 text-[10px] font-semibold text-slate-400">
          {unit}
        </div>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            margin={{ top: 20, right: 10, left: -10, bottom: 0 }}
            barGap={3}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              vertical={false}
              stroke="#f1f5f9"
            />

            <XAxis
              dataKey="block"
              axisLine={false}
              tickLine={false}
              tick={{ fill: '#64748b', fontSize: 11, fontWeight: 600 }}
              dy={4}
            />

            <YAxis
              width={32}
              axisLine={false}
              tickLine={false}
              tick={{ fill: '#94a3b8', fontSize: 10 }}
              domain={[0, yMax]}
            />

            <Tooltip
              contentStyle={{
                backgroundColor: '#ffffff',
                borderColor: '#e2e8f0',
                borderRadius: '8px',
                fontSize: '11px',
                boxShadow: '0 4px 12px rgba(0,0,0,0.05)',
              }}
              formatter={(value: any) => [`${value} ${unit}`]}
              labelFormatter={(label: any) => {
                const item = data.find((d) => d.block === label);
                return `${label} (${item?.timeRange || ''})`;
              }}
            />

            <Bar
              dataKey="main"
              name={`Today (${unit})`}
              fill="#1f3bb3"
              radius={[3, 3, 0, 0]}
              maxBarSize={24}
              minPointSize={4}
            />

            <Bar
              dataKey="compare"
              name={`Yesterday (${unit})`}
              fill="#cbd5e1"
              radius={[3, 3, 0, 0]}
              maxBarSize={24}
              minPointSize={4}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
