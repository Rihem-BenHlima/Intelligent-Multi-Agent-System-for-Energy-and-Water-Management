import React, { useEffect, useState } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Scatter,
} from 'recharts';
import { Shield } from 'lucide-react';
import { SubHeaderToolbar } from './SubHeaderToolbar';
import { MetricCards } from './MetricCards';
import { MetricItem } from '../types';
import { API_BASE } from '../config';

interface AnomalyDetectionPageProps {
  utilityType?: 'energy' | 'water';
  setUtilityType?: (type: 'energy' | 'water') => void;
}

interface AnomalyEntry { id: number; anomaly_name: string; start_date: number; end_date: number; description: string; }
interface TimelinePoint { timestamp: string; point_error: number; point_threshold: number; flagged: boolean; }

function timeAgo(ms: number): string {
  const hours = (Date.now() - ms) / 3600000;
  if (hours < 1) return 'Just now';
  if (hours < 24) return `${Math.round(hours)}h ago`;
  const days = hours / 24;
  if (days < 30) return `${Math.round(days)}d ago`;
  return `${Math.round(days / 30)}mo ago`;
}

// Format date as DD/MM/YYYY
function formatDMY(d: Date): string {
  const dd = String(d.getDate()).padStart(2, '0');
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  return `${dd}/${mm}/${d.getFullYear()}`;
}

export const AnomalyDetectionPage: React.FC<AnomalyDetectionPageProps> = ({
  utilityType = 'energy',
  setUtilityType,
}) => {
  const isEnergy = utilityType === 'energy';
  const [anomalies, setAnomalies] = useState<AnomalyEntry[]>([]);
  const [rawTimeline, setRawTimeline] = useState<TimelinePoint[]>([]);
  const [totalCount, setTotalCount] = useState(0);

  useEffect(() => {
    const fetchAnomalies = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/anomalies/${utilityType}`);
        const json = await res.json();
        setAnomalies(json.anomalies || []);
        setRawTimeline(json.timeline || []);
        setTotalCount(json.total_count ?? (json.anomalies || []).length);
      } catch (err) {
        console.error('Anomaly fetch failed:', err);
        setAnomalies([]);
        setRawTimeline([]);
        setTotalCount(0);
      }
    };
    fetchAnomalies();
    const interval = setInterval(fetchAnomalies, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [utilityType]);

  // Anchor to latest timestamp present in the timeline
  const latestTimelineTs = rawTimeline.length
    ? new Date(rawTimeline[rawTimeline.length - 1].timestamp).getTime()
    : Date.now();

  const timeline = rawTimeline.map((t, idx) => {
    return {
      idx,
      time: new Date(t.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      fullTime: new Date(t.timestamp).toLocaleString(),
      actual: +t.point_error.toFixed(4),
      baseline: +t.point_threshold.toFixed(4),
      flaggedError: t.flagged ? +t.point_error.toFixed(4) : null,
    };
  });
  const currentThreshold = rawTimeline.length ? rawTimeline[rawTimeline.length - 1].point_threshold : 0;

  // Evenly spaced x-axis ticks ensuring the final point is included
  const xTickCount = 8;
  const xStep = Math.max(1, Math.ceil((timeline.length - 1) / (xTickCount - 1)));
  const xTicks: number[] = [];
  for (let i = 0; i < timeline.length; i += xStep) xTicks.push(i);
  if (timeline.length && xTicks[xTicks.length - 1] !== timeline.length - 1) {
    xTicks.push(timeline.length - 1);
  }

  const last24hCutoff = latestTimelineTs - 24 * 3600 * 1000;
  const activeAnomalies = anomalies.filter((a) => (a.end_date ?? a.start_date) >= last24hCutoff);
  const lastDetected = anomalies.length
    ? timeAgo(Math.max(...anomalies.map((a) => a.end_date ?? a.start_date)))
    : '—';
  const flaggedHours = rawTimeline.filter((t) => t.flagged).length;
  const anomalyRate = rawTimeline.length ? (flaggedHours / rawTimeline.length) * 100 : 0;

  const avgDurationHours = anomalies.length
    ? anomalies.reduce((sum, a) => sum + (a.end_date - a.start_date), 0) / anomalies.length / 3600000
    : 0;
  const avgDurationLabel = anomalies.length ? `${avgDurationHours.toFixed(1)}h` : '—';

  const anomalyMetrics: MetricItem[] = [
    { id: 'active-anomalies', title: 'Active Anomalies', value: activeAnomalies.length.toString(), change: 'last 24h', isPositive: false, hideArrow: true },
    { id: 'avg-detection', title: 'Avg Duration', value: avgDurationLabel, change: 'of all detected anomalies', isPositive: true, hideArrow: true },
    { id: 'anomaly-rate', title: 'Anomaly Rate', value: `${anomalyRate.toFixed(1)}%`, change: 'last 30 days', isPositive: true, hideArrow: true },
    { id: 'last-detected', title: 'Last Detected', value: lastDetected, change: '', isPositive: true, hideArrow: true },
    { id: 'total-events', title: 'Total Events', value: totalCount.toString(), change: '', isPositive: true, hideArrow: true },
  ];

  return (
    <div className="space-y-2">
      {/* SubHeader Toolbar with Energy/Water Switch */}
      <SubHeaderToolbar
        utilityType={utilityType}
        setUtilityType={setUtilityType}
      />

      {/* 5 Top Metric Cards matching Dashboard */}
      <MetricCards metrics={anomalyMetrics} />

      {/* Middle Section: Reconstruction Error Chart */}
      <div className="bg-white border border-slate-100 rounded-lg p-3.5 lg:p-4 shadow-2xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3 pb-2 border-b border-slate-100">
          <div>
            <h3 className="text-xs lg:text-sm font-bold text-slate-800">Reconstruction Error Timeline</h3>
            <p className="text-[11px] text-slate-400">
              Last 30 days. Values above threshold indicate anomalies
            </p>
          </div>
          <div className="flex items-center gap-3 text-xs">
            <div className="flex items-center gap-1.5">
              <span className="w-5 h-0.5 bg-red-500 rounded-full" />
              <span className="text-red-600 font-semibold text-[11px]">Threshold ({currentThreshold.toFixed(4)})</span>
            </div>
          </div>
        </div>

        {timeline.length === 0 ? (
          <div className="py-10 px-4 text-center text-slate-400 text-sm">
            Not enough recent data to compute a reconstruction-error timeline yet.
          </div>
        ) : (
          <div className="h-60 lg:h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={timeline} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis
                  dataKey="idx"
                  tickFormatter={(idx: number) => timeline[idx]?.time ?? ''}
                  tickLine={false}
                  axisLine={false}
                  tick={{ fontSize: 11, fill: '#94a3b8' }}
                  ticks={xTicks}
                />
                {/* Sqrt scale for water to compress extreme spike error variance */}
                {isEnergy ? (
                  <YAxis width={40} tickLine={false} axisLine={false} tick={{ fontSize: 11, fill: '#94a3b8' }} />
                ) : (
                  <YAxis width={40} scale="sqrt" domain={[0, 'auto']} tickLine={false} axisLine={false} tick={{ fontSize: 11, fill: '#94a3b8' }} />
                )}
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    borderRadius: '8px',
                    border: 'none',
                    color: '#fff',
                    fontSize: '12px',
                  }}
                  labelFormatter={(idx: any) => timeline[idx]?.fullTime ?? ''}
                />
                {/* Adaptive threshold line */}
                <Line type="monotone" dataKey="baseline" name="Threshold" stroke="#ef4444" strokeDasharray="4 4" strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="actual" stroke="#1f3bb3" strokeWidth={2.5} dot={false} />
                <Scatter dataKey="flaggedError" fill="#ef4444" stroke="#ef4444" isAnimationActive={false} shape="circle" legendType="none" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Bottom Section: Detected Events Table Card (Full Width) */}
      <div className="w-full">
        <div className="bg-white border border-slate-100 rounded-lg p-3.5 lg:p-4 shadow-2xs flex flex-col justify-between h-full">
          <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
            <div>
              <h3 className="text-xs lg:text-sm font-bold text-slate-800">
                Detected Anomaly Events
              </h3>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500 font-medium">
                {totalCount} total
              </span>
            </div>
          </div>

          {/* Table of detected anomalies */}
          <div className="overflow-x-auto">
            <table className="w-full table-fixed text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-100 text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                  <th className="pb-2.5 pt-1 px-3 w-[25%]">Name</th>
                  <th className="pb-2.5 pt-1 px-3 w-[25%]">Date</th>
                  <th className="pb-2.5 pt-1 px-3 w-[50%]">Description</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-xs">
                {anomalies.length === 0 ? (
                  <tr>
                    <td colSpan={3} className="py-10 text-center text-slate-400">
                      <Shield className="w-8 h-8 mx-auto mb-2 opacity-30" />
                      No anomalies detected in the current period.
                    </td>
                  </tr>
                ) : anomalies.map((item) => {
                  const start = new Date(item.start_date);
                  const end = new Date(item.end_date ?? item.start_date);
                  const sameDay = start.toDateString() === end.toDateString();
                  const dateLabel = sameDay
                    ? `${formatDMY(start)} ${start.toLocaleTimeString()}`
                    : `${formatDMY(start)} – ${formatDMY(end)}`;
                  return (
                    <tr key={item.id} className="hover:bg-slate-50/60 transition-colors">
                      <td className="py-2.5 px-3 font-semibold text-slate-800 truncate">{item.anomaly_name}</td>
                      <td className="py-2.5 px-3 text-slate-500 whitespace-nowrap">{dateLabel}</td>
                      <td className="py-2.5 px-3 text-slate-500 leading-normal">{item.description}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
