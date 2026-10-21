import { useEffect, useState } from 'react';
import { getMarginalRate as getEnergyRate } from '../utils/stegTariff';
import { getMarginalRate as getWaterRate } from '../utils/sonedeTariff';
import { API_BASE } from '../config';

export interface AppNotification {
  id: string;
  icon: string;
  text: string;
  timestamp: number;
}

// Returns 'YYYY-Q#' calendar trimester for SONEDE billing.
function quarterPrefix(dateStr: string): string {
  const year = dateStr.slice(0, 4);
  const month = parseInt(dateStr.slice(5, 7), 10);
  const quarter = Math.floor((month - 1) / 3) + 1;
  return `${year}-Q${quarter}`;
}

function dailyTotals(records: any[]): Map<string, number> {
  const m = new Map<string, number>();
  records.forEach((r) => {
    const key = r.local_date || new Date(r.date).toISOString().slice(0, 10);
    m.set(key, (m.get(key) ?? 0) + (r.total_consumption ?? 0));
  });
  return m;
}

const DOMAINS: Array<'energy' | 'water'> = ['energy', 'water'];
const DOMAIN_LABEL: Record<'energy' | 'water', string> = { energy: 'Energy', water: 'Water' };
const DOMAIN_UNIT: Record<'energy' | 'water', string> = { energy: 'kWh', water: 'L' };

// Spike ratio relative to recent 90th percentile daily consumption
const SPIKE_RATIO = 1.15;

function shortDate(ms: number): string {
  return new Date(ms).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function percentile(nums: number[], p: number): number {
  if (!nums.length) return 0;
  const s = [...nums].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(p * (s.length - 1)))];
}

async function fetchAnomalyNotifications(domain: 'energy' | 'water'): Promise<AppNotification[]> {
  try {
    const res = await fetch(`${API_BASE}/api/anomalies/${domain}`);
    const json = await res.json();
    const anomalies: any[] = json.anomalies || [];
    if (!anomalies.length) return [];
    const mostRecent = anomalies.reduce((a, b) =>
      (b.end_date ?? b.start_date) > (a.end_date ?? a.start_date) ? b : a
    );
    const endTs = mostRecent.end_date ?? mostRecent.start_date;
    return [{
      id: `anomaly-${domain}-${mostRecent.id}`,
      icon: '⚠️',
      text: `${DOMAIN_LABEL[domain]} anomaly (${shortDate(endTs)}) — ${mostRecent.description}`,
      timestamp: endTs,
    }];
  } catch {
    return [];
  }
}

async function fetchReportNotification(domain: 'energy' | 'water'): Promise<AppNotification[]> {
  try {
    const res = await fetch(`${API_BASE}/api/reports/${domain}`);
    const json = await res.json();
    const reports: any[] = (json.reports || []).filter((r: any) => r.source === 'weekly_cron');
    if (!reports.length) return [];
    const latest = reports[0];
    const isoTs = latest.generated_at.endsWith('Z') ? latest.generated_at : latest.generated_at + 'Z';
    const ts = new Date(isoTs).getTime();
    if (Date.now() - ts > 7 * 24 * 3600 * 1000) return [];
    return [{
      id: `report-${domain}-${latest.id}`,
      icon: '📄',
      text: `Weekly ${DOMAIN_LABEL[domain].toLowerCase()} report is ready`,
      timestamp: ts,
    }];
  } catch {
    return [];
  }
}

async function fetchTariffTierNotifications(): Promise<AppNotification[]> {
  const out: AppNotification[] = [];
  try {
    const [eJson, wJson] = await Promise.all([
      fetch(`${API_BASE}/api/dashboard/energy?days=30`).then((r) => r.json()),
      fetch(`${API_BASE}/api/dashboard/water?days=95`).then((r) => r.json()),
    ]);

    const eTotals = dailyTotals(eJson.records || []);
    const eToday = [...eTotals.keys()].sort().pop();
    if (eToday) {
      const monthPrefix = eToday.slice(0, 7);
      const before = [...eTotals.entries()]
        .filter(([d]) => d.startsWith(monthPrefix) && d < eToday)
        .reduce((s, [, v]) => s + v, 0);
      const todayVal = eTotals.get(eToday) ?? 0;
      if (getEnergyRate(before + todayVal) > getEnergyRate(before)) {
        out.push({
          id: `tier-energy-${eToday}`,
          icon: '💸',
          text: 'Energy usage crossed into a higher pricing tier this month',
          timestamp: Date.now(),
        });
      }
    }

    const wTotals = dailyTotals(wJson.records || []);
    const wToday = [...wTotals.keys()].sort().pop();
    if (wToday) {
      const qk = quarterPrefix(wToday);
      const before = [...wTotals.entries()]
        .filter(([d]) => quarterPrefix(d) === qk && d < wToday)
        .reduce((s, [, v]) => s + v, 0);
      const todayVal = wTotals.get(wToday) ?? 0;
      if (getWaterRate(before + todayVal) > getWaterRate(before)) {
        out.push({
          id: `tier-water-${wToday}`,
          icon: '💸',
          text: 'Water usage crossed into a higher pricing tier this quarter',
          timestamp: Date.now(),
        });
      }
    }
  } catch {
    // skip
  }
  return out;
}

async function fetchForecastSpikeNotification(domain: 'energy' | 'water'): Promise<AppNotification[]> {
  try {
    const res = await fetch(`${API_BASE}/api/forecast/${domain}`);
    const json = await res.json();
    const forecasts: any[] = json.forecasts || [];
    const actuals: any[] = json.actuals || [];
    if (forecasts.length < 24 || !actuals.length) return [];

    const forecastTotal = forecasts.reduce((s: number, f: any) => s + (f.predicted_value ?? 0), 0);
    const dayTotals = [...dailyTotals(actuals).values()].filter((v) => v > 0);
    const busyDay = percentile(dayTotals, 0.9); // ~the busiest recent days
    if (busyDay <= 0 || forecastTotal < busyDay * SPIKE_RATIO) return [];

    const fmt = (n: number) => domain === 'water' ? Math.round(n).toLocaleString() : n.toFixed(1);
    return [{
      id: `spike-${domain}`,
      icon: '📈',
      text: `${DOMAIN_LABEL[domain]} demand forecast to run high in the next 24h — ~${fmt(forecastTotal)} ${DOMAIN_UNIT[domain]}, above the busiest recent days (~${fmt(busyDay)} ${DOMAIN_UNIT[domain]})`,
      timestamp: Date.now(),
    }];
  } catch {
    return [];
  }
}

// Data-driven notifications for anomalies, reports, tariff tiers, and demand spikes.
export function useNotifications(): AppNotification[] {
  const [notifications, setNotifications] = useState<AppNotification[]>([]);

  useEffect(() => {
    let cancelled = false;

    const fetchAll = async () => {
      const perDomain = await Promise.all(
        DOMAINS.flatMap((domain) => [
          fetchAnomalyNotifications(domain),
          fetchReportNotification(domain),
          fetchForecastSpikeNotification(domain),
        ])
      );
      const tierNotifications = await fetchTariffTierNotifications();
      const all = [...perDomain.flat(), ...tierNotifications];
      all.sort((a, b) => b.timestamp - a.timestamp);
      if (!cancelled) setNotifications(all);
    };

    fetchAll();
    const interval = setInterval(fetchAll, 5 * 60 * 1000);
    return () => { cancelled = true; clearInterval(interval); };
  }, []);

  return notifications;
}
