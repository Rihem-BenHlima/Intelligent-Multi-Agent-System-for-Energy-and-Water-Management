import React from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';

export interface ChartDataPoint { day: string; thisWeek: number; }

interface PerformanceChartCardProps {
  utilityType?: 'energy' | 'water';
  data: ChartDataPoint[];
}

export const PerformanceChartCard: React.FC<PerformanceChartCardProps> = ({
  utilityType = 'energy',
  data,
}) => {
  const isWater = utilityType === 'water';
  const unit = isWater ? 'L' : 'kWh';
  const maxVal = data.reduce((m, d) => Math.max(m, d.thisWeek), 0);
  const yMax = Math.max(10, Math.ceil(maxVal * 1.15));

  const renderCustomYAxisTick = (props: any) => {
    const { x, y, payload } = props;

    if (payload.value === yMax) {
      return (
        <g transform={`translate(${x},${y})`}>
          <text
            x={-4}
            y={-14}
            textAnchor="end"
            fill="#94a3b8"
            fontSize={12}
            fontWeight={600}
          >
            {unit}
          </text>
          <text
            x={-4}
            y={4}
            textAnchor="end"
            fill="#94a3b8"
            fontSize={11}
          >
            {yMax}
          </text>
        </g>
      );
    }

    return (
      <g transform={`translate(${x},${y})`}>
        <text
          x={-4}
          y={4}
          textAnchor="end"
          fill="#94a3b8"
          fontSize={11}
        >
          {payload.value}
        </text>
      </g>
    );
  };

  return (
    <div className="bg-white border border-slate-100 rounded-lg p-3.5 lg:p-4 shadow-2xs flex flex-col justify-between h-full">
      {/* Card Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2">
        <div>
          <h3 className="text-xs lg:text-sm font-bold text-slate-800">
            {isWater ? 'Water Usage' : 'Energy Usage'}
          </h3>
          <p className="text-[11px] text-slate-400 font-normal mt-0.5">
            Last 7 days
          </p>
        </div>
      </div>

      {/* Chart Canvas */}
      <div className="w-full flex-1 min-h-[220px] lg:min-h-[260px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={data}
            margin={{ top: 20, right: 10, left: -25, bottom: 0 }}
          >
            <defs>
              <linearGradient id="thisWeekGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#1f3bb3" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#1f3bb3" stopOpacity={0.0} />
              </linearGradient>
            </defs>

            <CartesianGrid
              strokeDasharray="3 3"
              vertical={false}
              stroke="#f1f5f9"
            />

            <XAxis
              dataKey="day"
              axisLine={false}
              tickLine={false}
              tick={{ fill: '#94a3b8', fontSize: 11, fontWeight: 500 }}
              dy={8}
            />

            <YAxis
              width={32}
              axisLine={false}
              tickLine={false}
              tick={renderCustomYAxisTick}
              domain={[0, yMax]}
              ticks={[0, Math.round(yMax * 0.33), Math.round(yMax * 0.66), yMax]}
            />

            <Tooltip
              contentStyle={{
                backgroundColor: '#ffffff',
                borderColor: '#e2e8f0',
                borderRadius: '8px',
                fontSize: '12px',
                boxShadow: '0 4px 12px rgba(0,0,0,0.05)',
              }}
              formatter={(value: any) => [`${value} ${unit}`, 'Last 7 Days']}
            />

            {/* Last 7 days area */}
            <Area
              type="monotone"
              dataKey="thisWeek"
              name="Last 7 days"
              stroke="#1f3bb3"
              strokeWidth={2.5}
              fill="url(#thisWeekGrad)"
              dot={{ r: 4, fill: '#1f3bb3', strokeWidth: 2, stroke: '#ffffff' }}
              activeDot={{ r: 6, fill: '#1f3bb3' }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
