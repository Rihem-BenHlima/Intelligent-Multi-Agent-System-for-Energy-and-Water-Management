import React, { useEffect, useState } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';
import { AiInsightsCard } from './AiInsightsCard';
import { SubHeaderToolbar } from './SubHeaderToolbar';
import { API_BASE } from '../config';

interface DemandForecastPageProps {
  utilityType?: 'energy' | 'water';
  setUtilityType?: (type: 'energy' | 'water') => void;
}

interface ChartPoint { idx: number; time: string; actual: number | null; forecast: number | null; confidence: [number, number] | null; }
interface HourlyPoint { hour: string; forecast: number; }

export const DemandForecastPage: React.FC<DemandForecastPageProps> = ({
  utilityType = 'energy',
  setUtilityType,
}) => {
  const isEnergy = utilityType === 'energy';
  const yAxisUnit = isEnergy ? 'kWh' : 'L';
  const conv = (v: number) => v;

  const [chartData, setChartData] = useState<ChartPoint[]>([]);
  const [hourlyForecast, setHourlyForecast] = useState<HourlyPoint[]>([]);
  const [recommendations, setRecommendations] = useState<string[]>([]);
  const [recoLoading, setRecoLoading] = useState(false);

  useEffect(() => {
    const fetchForecast = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/forecast/${utilityType}`);
        const json = await res.json();

        const rawActuals = [...(json.actuals || [])].sort((a: any, b: any) => a.date - b.date);
        const displayActuals = rawActuals.slice(-48);
        const lastActual = displayActuals.length ? conv(displayActuals[displayActuals.length - 1].total_consumption ?? 0) : 0;

        let lastDay = -1;
        const smartLabel = (ts: number) => {
          const d = new Date(ts);
          const day = d.getUTCDate();
          if (day !== lastDay) { lastDay = day; return d.toLocaleString('default', { month: 'short', day: 'numeric', timeZone: 'UTC' }); }
          return String(d.getUTCHours()).padStart(2, '0') + ':00';
        };

        const actualPoints: Omit<ChartPoint, 'idx'>[] = displayActuals.map((a: any) => ({
          time: smartLabel(a.date), actual: +conv(a.total_consumption ?? 0).toFixed(3), forecast: null, confidence: null,
        }));

        const forecasts = (json.forecasts || []).sort((a: any, b: any) => a.date - b.date);
        setHourlyForecast(forecasts.map((f: any) => ({
          hour: `${String(new Date(f.date).getUTCHours()).padStart(2, '0')}:00`,
          forecast: +conv(f.predicted_value ?? 0).toFixed(1),
        })));

        const forecastPoints: Omit<ChartPoint, 'idx'>[] = [
          { time: 'Now', actual: null, forecast: +lastActual.toFixed(3), confidence: [+(lastActual * 0.92).toFixed(3), +(lastActual * 1.08).toFixed(3)] },
          ...forecasts.map((f: any) => {
            const val = conv(f.predicted_value ?? 0);
            const margin = val * 0.08;
            return {
              time: smartLabel(f.date), actual: null,
              forecast: +val.toFixed(3),
              confidence: [+Math.max(0, val - margin).toFixed(3), +(val + margin).toFixed(3)] as [number, number],
            };
          }),
        ];

        setChartData(
          [...actualPoints, ...forecastPoints].map((p, idx) => ({ ...p, idx }))
        );
      } catch (err) {
        console.error('Forecast fetch failed:', err);
        setChartData([]);
        setHourlyForecast([]);
      }
    };
    fetchForecast();
    const interval = setInterval(fetchForecast, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [utilityType]);

  useEffect(() => {
    let cancelled = false;
    const fetchRecos = async () => {
      setRecoLoading(true);
      try {
        const res = await fetch(`${API_BASE}/api/recommendations/${utilityType}`);
        if (!res.ok) return;
        const json = await res.json();
        if (cancelled) return;
        setRecommendations(Array.isArray(json.recommendations) ? json.recommendations : []);
      } catch {}
      finally { if (!cancelled) setRecoLoading(false); }
    };
    fetchRecos();
    return () => { cancelled = true; };
  }, [utilityType]);

  const total24hForecast = hourlyForecast.reduce((sum, h) => sum + h.forecast, 0);
  const avgHourlyForecast = hourlyForecast.length ? total24hForecast / hourlyForecast.length : 0;
  const peakHourItem = hourlyForecast.length ? [...hourlyForecast].sort((a, b) => b.forecast - a.forecast)[0] : null;
  const minHourItem = hourlyForecast.length ? [...hourlyForecast].sort((a, b) => a.forecast - b.forecast)[0] : null;

  const chartMax = chartData.reduce((m, d) => {
    const vals = [d.actual, d.forecast, ...(d.confidence || [])].filter(
      (v): v is number => v !== null && v !== undefined
    );
    return vals.length ? Math.max(m, ...vals) : m;
  }, 0);

  const rawMax = isEnergy ? Math.max(chartMax, 0.1) : Math.max(chartMax * 1.15, 0.1);
  const niceStep = isEnergy
    ? (rawMax <= 2 ? 0.5 : rawMax <= 4 ? 1 : rawMax <= 10 ? 2 : rawMax <= 50 ? 10 :
       rawMax <= 200 ? 50 : rawMax <= 1000 ? 100 : 500)
    : (rawMax <= 2 ? 0.5 : rawMax <= 10 ? 2 : rawMax <= 50 ? 10 :
       rawMax <= 200 ? 50 : rawMax <= 1000 ? 100 : 500);
  let yMax = Math.ceil(rawMax / niceStep) * niceStep;
  if (isEnergy && yMax - rawMax < niceStep * 0.15) yMax += niceStep;
  const yTicks = Array.from(
    { length: Math.round(yMax / niceStep) + 1 },
    (_, i) => Math.round(i * niceStep * 100) / 100
  );

  const hourLabelPattern = /^\d{2}:00$/;
  const xTicks = Array.from(new Set([
    ...chartData.filter(p => !hourLabelPattern.test(p.time)).map(p => p.idx),
    ...chartData.filter((_, i) => i % 4 === 0).map(p => p.idx),
  ])).sort((a, b) => a - b);

  return (
    <div className="space-y-2">
      {/* SubHeader Toolbar with Energy/Water Switch */}
      <SubHeaderToolbar
        utilityType={utilityType}
        setUtilityType={setUtilityType}
      />

      {/* Main Grid Layout: Left 8 cols (KPI Cards + Chart) & Right 4 cols (AI Insights Card) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 lg:gap-3.5 items-stretch">
        {/* Left Column (8 cols): 4 Summary KPI Cards + Chart Canvas */}
        <div className="lg:col-span-8 flex flex-col gap-3 lg:gap-3.5">
          {/* Summary Highlight KPI Cards — real, no fabricated deltas */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 lg:gap-3.5">
            <div className="bg-white border border-slate-100 rounded-lg p-3.5 lg:p-4 shadow-2xs hover:shadow-xs transition-shadow flex flex-col justify-between">
              <p className="text-[11px] lg:text-xs text-slate-400 font-medium truncate mb-1">
                Total Forecast
              </p>
              <h3 className="text-xl lg:text-2xl font-bold text-slate-800 tracking-tight my-1 truncate">
                {total24hForecast.toLocaleString('en', { maximumFractionDigits: 1 })} <span className="text-xs font-normal text-slate-500">{yAxisUnit}</span>
              </h3>
              <div className="flex items-center gap-1 text-[11px] font-normal text-slate-400 text-left">
                next 24 hours
              </div>
            </div>

            <div className="bg-white border border-slate-100 rounded-lg p-3.5 lg:p-4 shadow-2xs hover:shadow-xs transition-shadow flex flex-col justify-between">
              <p className="text-[11px] lg:text-xs text-slate-400 font-medium truncate mb-1">
                Avg Hourly Forecast
              </p>
              <h3 className="text-xl lg:text-2xl font-bold text-slate-800 tracking-tight my-1 truncate">
                {avgHourlyForecast.toFixed(1)} <span className="text-xs font-normal text-slate-500">{yAxisUnit}</span>
              </h3>
              <div className="flex items-center gap-1 text-[11px] text-slate-400 font-normal text-left">
                Avg across next 24h
              </div>
            </div>

            <div className="bg-white border border-slate-100 rounded-lg p-3.5 lg:p-4 shadow-2xs hover:shadow-xs transition-shadow flex flex-col justify-between">
              <p className="text-[11px] lg:text-xs text-slate-400 font-medium truncate mb-1">
                Expected Peak Load
              </p>
              <h3 className="text-xl lg:text-2xl font-bold text-slate-800 tracking-tight my-1 truncate">
                {peakHourItem ? peakHourItem.forecast.toLocaleString() : '—'} <span className="text-xs font-normal text-slate-500">{yAxisUnit}</span>
              </h3>
              <div className="flex items-center gap-1 text-[11px] text-slate-400 font-normal truncate text-left">
                {peakHourItem ? `at ${peakHourItem.hour}` : ''}
              </div>
            </div>

            <div className="bg-white border border-slate-100 rounded-lg p-3.5 lg:p-4 shadow-2xs hover:shadow-xs transition-shadow flex flex-col justify-between">
              <p className="text-[11px] lg:text-xs text-slate-400 font-medium truncate mb-1">
                Minimum Base Load
              </p>
              <h3 className="text-xl lg:text-2xl font-bold text-slate-800 tracking-tight my-1 truncate">
                {minHourItem ? minHourItem.forecast.toLocaleString() : '—'} <span className="text-xs font-normal text-slate-500">{yAxisUnit}</span>
              </h3>
              <div className="flex items-center gap-1 text-[11px] text-slate-400 font-normal truncate text-left">
                {minHourItem ? `at ${minHourItem.hour}` : ''}
              </div>
            </div>
          </div>

          {/* Main Forecast Chart Canvas */}
          <div className="w-full bg-white border border-slate-100 rounded-lg p-4 lg:p-5 shadow-2xs flex-1 flex flex-col justify-between">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
              <div>
                <h3 className="text-base lg:text-lg font-bold text-slate-800 tracking-tight">
                  Continuous {isEnergy ? 'Energy' : 'Water'} Consumption Timeline
                </h3>
                <p className="text-xs lg:text-sm text-slate-500 font-normal mt-0.5">
                  Past usage and the next 24 hours
                </p>
              </div>

              {/* Chart Legends */}
              <div className="flex items-center gap-4 text-xs font-medium self-end sm:self-auto">
                <div className="flex items-center gap-1.5 text-slate-600">
                  <span className="w-2.5 h-2.5 rounded-full bg-[#1f3bb3]" />
                  <span>Actual Data</span>
                </div>
                <div className="flex items-center gap-1.5 text-slate-600">
                  <span className="w-2.5 h-2.5 rounded-full bg-slate-400" />
                  <span>24h Forecast</span>
                </div>
                <div className="flex items-center gap-1.5 text-slate-600">
                  <span className="w-3 h-2.5 rounded-xs bg-indigo-200/80 border border-indigo-300" />
                  <span>Confidence</span>
                </div>
              </div>
            </div>

            {/* Recharts Canvas */}
            <div className="relative w-full min-h-[220px] lg:min-h-[260px]">
              <div className="absolute top-0 left-0 w-9 text-center z-10 text-[10px] font-semibold text-slate-400">
                {yAxisUnit}
              </div>
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart
                  data={chartData}
                  margin={{ top: 20, right: 10, left: -10, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                  <XAxis
                    dataKey="idx"
                    tickFormatter={(idx: number) => chartData[idx]?.time ?? ''}
                    stroke="#94a3b8"
                    fontSize={11}
                    tickLine={false}
                    axisLine={{ stroke: '#e2e8f0' }}
                    ticks={xTicks}
                  />
                  <YAxis
                    width={40}
                    stroke="#94a3b8"
                    tickLine={false}
                    axisLine={false}
                    tick={{ fill: '#94a3b8', fontSize: 10 }}
                    domain={[0, yMax]}
                    ticks={yTicks}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#ffffff',
                      borderColor: '#e2e8f0',
                      borderRadius: '8px',
                      fontSize: '12px',
                      boxShadow: '0 4px 12px rgba(0, 0, 0, 0.05)',
                    }}
                    labelFormatter={(idx: any) => chartData[idx]?.time ?? ''}
                    formatter={(value: any, name: any) => {
                      if (name === 'confidence') {
                        if (Array.isArray(value) && value[0] !== null && value[1] !== null && value[0] !== value[1]) {
                          return [`${value[0]} - ${value[1]} ${yAxisUnit}`, 'Confidence'];
                        }
                        return [null, null];
                      }
                      if (value === null) return ['--', name];
                      return [
                        `${value} ${yAxisUnit}`,
                        name === 'actual' ? 'Actual Load' : name === 'forecast' ? '24h Forecast' : name,
                      ];
                    }}
                  />

                  {/* Confidence band — standard forecast-chart convention:
                      an evenly-tinted region between two visible dashed
                      edges, not a shadow fading to nothing. A gradient
                      fading to 0% opacity at the bottom doesn't read as "a
                      range between two bounds" — it just looks like a haze
                      trailing off, which is what "not well represented"
                      was pointing at. */}
                  <Area
                    type="monotone"
                    dataKey="confidence"
                    stroke="none"
                    fill="#6366f1"
                    fillOpacity={0.12}
                    connectNulls={false}
                  />

                  {/* Actual Load Line */}
                  <Line
                    type="monotone"
                    dataKey="actual"
                    stroke="#1f3bb3"
                    strokeWidth={2.5}
                    dot={false}
                    activeDot={{ r: 4, fill: '#1f3bb3' }}
                    connectNulls={false}
                  />

                  {/* Forecast Line */}
                  <Line
                    type="monotone"
                    dataKey="forecast"
                    stroke="#64748b"
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    dot={false}
                    activeDot={{ r: 4, fill: '#64748b' }}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Right Column (4 cols): AI Insights Card — real recommendations */}
        <div className="lg:col-span-4 flex flex-col">
          <AiInsightsCard tips={recommendations} loading={recoLoading} />
        </div>
      </div>

      {/* 24-Hour Forecast Detailed Breakdown Small Blocks (Full Width at Bottom) */}
      <div className="w-full bg-white border border-slate-100 rounded-lg p-4 lg:p-4.5 shadow-2xs space-y-3">
        <div>
          <h3 className="text-sm font-bold text-slate-800">
            24-Hour Forecasted Values
          </h3>
        </div>

        {/* Single Row 24-Block Layout Spanning Page Width */}
        <div className="grid grid-cols-6 sm:grid-cols-12 lg:grid-cols-[repeat(24,minmax(0,1fr))] gap-1.5 overflow-x-auto">
          {hourlyForecast.map((row, idx) => (
            <div
              key={idx}
              className="bg-slate-50/90 hover:bg-white hover:shadow-2xs border border-slate-200/80 hover:border-indigo-300 rounded-md py-2.5 px-1 transition-all flex flex-col justify-between items-center text-center space-y-1 group min-w-[36px]"
            >
              <div className="text-[10px] lg:text-[11px] font-semibold text-slate-500 group-hover:text-[#1f3bb3] transition-colors truncate w-full">
                {row.hour}
              </div>
              <div className="text-xs lg:text-[13px] font-extrabold text-slate-900 tracking-tight">
                {row.forecast.toLocaleString()}
              </div>
              <div className="text-[9px] lg:text-[10px] font-medium text-slate-400">
                {yAxisUnit}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
