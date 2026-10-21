import { useCallback, useEffect, useRef, useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { SubHeaderToolbar } from './components/SubHeaderToolbar';
import { MetricCards } from './components/MetricCards';
import { StatusSummaryCard } from './components/StatusSummaryCard';
import { PerformanceChartCard, type ChartDataPoint } from './components/PerformanceChartCard';
import { MarketOverviewCard, type TimeOfDayBlock } from './components/MarketOverviewCard';
import { ChatbotWidget } from './components/ChatbotWidget';
import { ChatbotPage } from './components/ChatbotPage';
import { DemandForecastPage } from './components/DemandForecastPage';
import { AnomalyDetectionPage } from './components/AnomalyDetectionPage';
import { ReportsPage } from './components/ReportsPage';
import { SettingsModal } from './components/SettingsModal';
import { useSidebarColor } from './useSidebarColor';
import { useProfile } from './useProfile';
import { INITIAL_CHAT_SESSIONS } from './data/mockChatData';
import { ChatSession, MetricItem } from './types';
import { computeTieredCost, getMarginalRate } from './utils/stegTariff';
import { getMarginalRate as getMarginalWaterRate } from './utils/sonedeTariff';
import { API_BASE } from './config';

// 'YYYY-MM-DD' -> 'YYYY-Q#' — SONEDE's billing period (calendar trimester),
// mirrors the backend's water_tariff_service._quarter_prefix.
function quarterPrefix(dateStr: string): string {
  const year = dateStr.slice(0, 4);
  const month = parseInt(dateStr.slice(5, 7), 10);
  const quarter = Math.floor((month - 1) / 3) + 1;
  return `${year}-Q${quarter}`;
}

export default function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [activeNavItem, setActiveNavItem] = useState('dashboard');
  const [utilityType, setUtilityType] = useState<'energy' | 'water'>('energy');
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const { sidebarColor, setSidebarColor } = useSidebarColor();
  const { profile, update: updateProfile } = useProfile();

  // Shared Chat State across floating widget & dedicated chatbot page.
  // Persisted to localStorage — without this, a page refresh wiped the
  // visible conversation back to the default greeting every time, even
  // though the backend's Redis-backed checkpointer was correctly
  // remembering the same thread_id's history the whole time. This closes
  // that gap on the display side (backend memory was never the problem).
  const CHAT_STORAGE_KEY = 'sfmconnect_chat_sessions';
  const ACTIVE_SESSION_STORAGE_KEY = 'sfmconnect_active_session_id';

  const [chatSessions, setChatSessions] = useState<ChatSession[]>(() => {
    try {
      const saved = localStorage.getItem(CHAT_STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) return parsed;
      }
    } catch (err) {
      console.error('[App] Failed to restore chat sessions from localStorage:', err);
    }
    return INITIAL_CHAT_SESSIONS;
  });
  const [activeSessionId, setActiveSessionId] = useState<string>(() => {
    try {
      const saved = localStorage.getItem(ACTIVE_SESSION_STORAGE_KEY);
      if (saved) return saved;
    } catch (err) {
      console.error('[App] Failed to restore active session id from localStorage:', err);
    }
    return INITIAL_CHAT_SESSIONS[0].id;
  });

  useEffect(() => {
    try {
      localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(chatSessions));
    } catch (err) {
      console.error('[App] Failed to persist chat sessions to localStorage:', err);
    }
  }, [chatSessions]);

  useEffect(() => {
    try {
      localStorage.setItem(ACTIVE_SESSION_STORAGE_KEY, activeSessionId);
    } catch (err) {
      console.error('[App] Failed to persist active session id to localStorage:', err);
    }
  }, [activeSessionId]);

  // Leaving the conversation (closing the floating widget, or navigating
  // away from the dedicated Chatbot page) finalizes the current session —
  // it stays in `sessions`/History exactly as-is, already persisted — and
  // starts a fresh one, so re-opening the assistant ALWAYS lands on a new
  // conversation, never resumes the last one. Keeping the widget open (not
  // clicking X) never triggers this, so an in-progress chat is untouched.
  // Refs (not the state values directly) so this stays a stable callback:
  // it's invoked from a useEffect cleanup in ChatbotPage, which must NOT
  // re-fire on every message, only on actual unmount.
  const chatSessionsRef = useRef(chatSessions);
  const activeSessionIdRef = useRef(activeSessionId);
  useEffect(() => { chatSessionsRef.current = chatSessions; }, [chatSessions]);
  useEffect(() => { activeSessionIdRef.current = activeSessionId; }, [activeSessionId]);

  const CHAT_GREETING = 'Hello! I’m your AI assistant — ask me about consumption, anomalies, or forecasts using your real data.';

  const startNewChatSession = useCallback((title: string) => {
    const newId = `session-${Date.now()}`;
    const newSession: ChatSession = {
      id: newId,
      title,
      date: 'Just now',
      lastMessage: CHAT_GREETING,
      messages: [{
        id: `msg-${Date.now()}`,
        sender: 'ai',
        text: CHAT_GREETING,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      }],
    };
    // Drop any earlier session that never got past the opening greeting — a
    // conversation the user opened but never actually wrote in is not
    // history and must not be kept.
    setChatSessions((prev) => [newSession, ...prev.filter((s) => s.messages.length > 1)]);
    setActiveSessionId(newId);
  }, []);

  const handleLeaveChat = useCallback(() => {
    const current = chatSessionsRef.current.find((s) => s.id === activeSessionIdRef.current);
    // Archive + replace only when the active session actually has content
    // (more than the opening greeting). A pristine greeting-only session IS
    // already the fresh one to re-open into, so reuse it — otherwise every
    // close/navigate would stack up empty "New Energy Query" sessions in
    // History.
    if (current && current.messages.length > 1) {
      startNewChatSession('New Energy Query');
    }
  }, [startNewChatSession]);

  // ── Real dashboard data ──────────────────────────────────────────────────
  const [dashLoading, setDashLoading] = useState(true);
  const [todayConsumption, setTodayConsumption] = useState<number | null>(null);
  const [todayDeltaPct, setTodayDeltaPct] = useState<number | null>(null);
  const [peakUsage, setPeakUsage] = useState<number | null>(null);
  const [peakUsageTime, setPeakUsageTime] = useState<string | null>(null);
  const [todaysCost, setTodaysCost] = useState<number | null>(null);
  const [costDeltaPct, setCostDeltaPct] = useState<number | null>(null);
  const [avgDailyKwh, setAvgDailyKwh] = useState<number | null>(null);
  const [activeAnomaliesCount, setActiveAnomaliesCount] = useState(0);
  const [weekComparisonData, setWeekComparisonData] = useState<ChartDataPoint[]>([]);
  const [timeOfDayData, setTimeOfDayData] = useState<TimeOfDayBlock[]>([]);
  const [weather, setWeather] = useState<{ tempC: number; humidity: number; condition: string } | null>(null);
  const [rate, setRate] = useState<number | null>(null);

  const isEnergy = utilityType === 'energy';
  const unit = isEnergy ? 'kWh' : 'L';
  // Water is stored and served in LITRES now (same as energy in kWh) — no
  // conversion. Kept as a pass-through so the call sites below don't churn.
  const conv = (v: number) => v;

  useEffect(() => {
    const domain = utilityType;
    const fetchDashboard = async () => {
      setDashLoading(true);
      try {
        const [dashRes30, anomRes] = await Promise.all([
          fetch(`${API_BASE}/api/dashboard/${domain}?days=30`),
          fetch(`${API_BASE}/api/anomalies/${domain}`),
        ]);
        const dashData30 = await dashRes30.json();
        const anomData = await anomRes.json();
        const allRecords: any[] = dashData30.records || [];

        const dayMap = new Map<string, number>();
        allRecords.forEach((r: any) => {
          const key = r.local_date || new Date(r.date).toISOString().slice(0, 10);
          dayMap.set(key, (dayMap.get(key) ?? 0) + conv(r.total_consumption ?? 0));
        });
        const sortedEntries = Array.from(dayMap.entries()).sort(([a], [b]) => a.localeCompare(b));

        // "Today" is anchored on the LATEST date actually present in the
        const todayKey = sortedEntries.length
          ? sortedEntries[sortedEntries.length - 1][0]
          : new Date().toISOString().slice(0, 10);
        const todayRecordsRaw = allRecords
          .filter((r: any) => (r.local_date || new Date(r.date).toISOString().slice(0, 10)) === todayKey)
          .sort((a: any, b: any) => a.date - b.date);

        const todayIdx = sortedEntries.findIndex(([d]) => d === todayKey);
        let yesterdayRecordsRaw: any[] = [];
        if (todayIdx > 0) {
          const yesterdayKey = sortedEntries[todayIdx - 1][0];
          yesterdayRecordsRaw = allRecords.filter((r: any) => (r.local_date || new Date(r.date).toISOString().slice(0, 10)) === yesterdayKey);
        }

        const suppressDelta = domain === 'water';
        const yesterdaySoFar = suppressDelta ? null : dashData30.yesterday_cost_so_far;
        const yesterdayConsumptionSoFar = yesterdaySoFar
          ? conv(domain === 'water' ? (yesterdaySoFar.consumption_l ?? 0) : (yesterdaySoFar.consumption_kwh ?? 0))
          : null;

        if (todayIdx !== -1) {
          const todayVal = sortedEntries[todayIdx][1];
          setTodayConsumption(todayVal);
          if (todayIdx > 0 && yesterdayConsumptionSoFar !== null && yesterdayConsumptionSoFar > 0) {
            setTodayDeltaPct(((todayVal - yesterdayConsumptionSoFar) / yesterdayConsumptionSoFar) * 100);
          } else {
            setTodayDeltaPct(null);
          }
        } else {
          setTodayConsumption(null);
          setTodayDeltaPct(null);
        }

        if (domain === 'energy' && todayIdx !== -1) {
          const monthPrefix = todayKey.slice(0, 7);
          const monthBeforeToday = sortedEntries
            .filter(([d]) => d.startsWith(monthPrefix) && d < todayKey)
            .reduce((sum, [, v]) => sum + v, 0);
          const cost = computeTieredCost(monthBeforeToday, sortedEntries[todayIdx][1]);
          setTodaysCost(cost);
          setCostDeltaPct(yesterdaySoFar && yesterdaySoFar.cost_dt > 0 ? ((cost - yesterdaySoFar.cost_dt) / yesterdaySoFar.cost_dt) * 100 : null);
        } else if (domain === 'water') {
          const tcSoFar = dashData30.today_cost_so_far;
          setTodaysCost(tcSoFar ? tcSoFar.cost_dt : null);
          setCostDeltaPct(tcSoFar && yesterdaySoFar && yesterdaySoFar.cost_dt > 0 ? ((tcSoFar.cost_dt - yesterdaySoFar.cost_dt) / yesterdaySoFar.cost_dt) * 100 : null);
        } else {
          setTodaysCost(null);
          setCostDeltaPct(null);
        }

        let peakVal: number | null = null;
        let peakTime: string | null = null;
        if (domain === 'energy') {
          todayRecordsRaw.forEach((r: any) => {
            const v = r.total_consumption ?? 0;
            if (peakVal === null || v > peakVal) {
              peakVal = v;
              peakTime = new Date(r.date).toLocaleTimeString('default', { hour: '2-digit', minute: '2-digit' });
            }
          });
        } else {
          const latest = [...todayRecordsRaw].reverse().find((r: any) => (r.total_consumption ?? 0) > 0)
            ?? todayRecordsRaw[todayRecordsRaw.length - 1];
          if (latest) {
            peakVal = conv(latest.total_consumption ?? 0);
            peakTime = new Date(latest.date).toLocaleTimeString('default', { hour: '2-digit', minute: '2-digit' });
          }
        }
        setPeakUsage(peakVal);
        setPeakUsageTime(peakTime);
        setAvgDailyKwh(dashData30.summary?.avg_daily ?? null);

        const latestRecordTs = allRecords.length
          ? Math.max(...allRecords.map((r: any) => r.date))
          : Date.now();
        const last24hCutoff = latestRecordTs - 24 * 3600 * 1000;
        const active = (anomData.anomalies || []).filter((a: any) => (a.end_date ?? a.start_date) >= last24hCutoff);
        setActiveAnomaliesCount(active.length);

        const thisWeekArr = sortedEntries.slice(-7);
        setWeekComparisonData(thisWeekArr.map(([dateStr, val]) => {
          const d = new Date(dateStr + 'T00:00:00');
          return {
            day: d.toLocaleString('default', { weekday: 'short' }).toUpperCase(),
            thisWeek: +val.toFixed(2),
          };
        }));

        const nowHour = 24;
        setTimeOfDayData(Array.from({ length: 12 }, (_, i) => {
          const startH = i * 2;
          const sumFor = (rows: any[]) => rows
            .filter((r: any) => { const h = new Date(r.date).getHours(); return h >= startH && h < startH + 2; })
            .reduce((s: number, r: any) => s + conv(r.total_consumption ?? 0), 0);
          const reached = startH <= nowHour;
          const mainSum = +sumFor(todayRecordsRaw).toFixed(2);
          const compareSum = +sumFor(yesterdayRecordsRaw).toFixed(2);
          return {
            block: `${String(startH).padStart(2, '0')}:00`,
            timeRange: `${String(startH).padStart(2, '0')}:00–${String(startH + 2).padStart(2, '0')}:00`,
            main: reached && mainSum > 0 ? mainSum : null,
            compare: compareSum > 0 ? compareSum : null,
          };
        }));
      } catch (err) {
        console.error('Dashboard fetch failed:', err);
      } finally {
        setDashLoading(false);
      }
    };
    fetchDashboard();
    const interval = setInterval(fetchDashboard, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [utilityType]);

  useEffect(() => {
    let cancelled = false;
    const fetchWeather = () => {
      fetch(`${API_BASE}/api/weather`)
        .then((res) => res.json())
        .then((wData) => {
          if (cancelled) return;
          const latest = wData.current || null;
          if (!latest) { setWeather(null); return; }
          const condition = latest.cloud_cover > 60 ? 'Cloudy' : latest.cloud_cover > 30 ? 'Partly Cloudy' : 'Sunny';
          setWeather({ tempC: latest.temperature, humidity: latest.humidity, condition });
        })
        .catch((err) => console.error('[App] weather fetch failed:', err));
    };
    fetchWeather();
    const interval = setInterval(fetchWeather, 5 * 60 * 1000);
    return () => { cancelled = true; clearInterval(interval); };
  }, []);

  // Real marginal tariff rate — STEG (energy) / SONEDE (water).
  useEffect(() => {
    let cancelled = false;
    if (utilityType === 'water') {
      fetch(`${API_BASE}/api/dashboard/water?days=95`)
        .then((res) => res.json())
        .then((data) => {
          if (cancelled) return;
          const todayStr = new Date().toISOString().slice(0, 10);
          const quarterKey = quarterPrefix(todayStr);
          const dayTotals = new Map<string, number>();
          (data.records || []).forEach((r: any) => {
            const key = r.local_date || new Date(r.date).toISOString().slice(0, 10);
            dayTotals.set(key, (dayTotals.get(key) ?? 0) + (r.total_consumption ?? 0));
          });
          let quarterTotal = 0;
          dayTotals.forEach((v, d) => { if (quarterPrefix(d) === quarterKey) quarterTotal += v; });
          setRate(getMarginalWaterRate(quarterTotal));
        })
        .catch((err) => console.error('[App] water tariff calc failed:', err));
    } else {
      fetch(`${API_BASE}/api/dashboard/energy?days=30`)
        .then((res) => res.json())
        .then((data) => {
          if (cancelled) return;
          const monthPrefix = new Date().toISOString().slice(0, 7);
          const monthTotal = (data.records || [])
            .filter((r: any) => (r.local_date || new Date(r.date).toISOString().slice(0, 10)).startsWith(monthPrefix))
            .reduce((sum: number, r: any) => sum + (r.total_consumption ?? 0), 0);
          setRate(getMarginalRate(monthTotal));
        })
        .catch((err) => console.error('[App] tariff calc failed:', err));
    }
    return () => { cancelled = true; };
  }, [utilityType]);

  const metrics: MetricItem[] = [
    {
      id: '1', title: 'Current Consumption',
      value: todayConsumption !== null ? `${todayConsumption.toLocaleString('en', { maximumFractionDigits: 1 })} ${unit}` : '—',
      // Water never has a delta (suppressed unconditionally above), so the
      // 'vs yesterday' placeholder text would show with nothing to anchor
      // it to — leave it blank there instead.
      change: todayDeltaPct !== null ? `${Math.abs(todayDeltaPct).toFixed(1)}%` : (isEnergy ? 'vs yesterday' : ''),
      isPositive: todayDeltaPct !== null ? todayDeltaPct >= 0 : true,
      hideArrow: todayDeltaPct === null,
      subtext: todayDeltaPct !== null ? 'vs yesterday' : undefined,
    },
    {
      id: '2', title: isEnergy ? 'Peak Usage' : 'Flow Rate',
      value: peakUsage !== null ? `${peakUsage.toFixed(1)} ${isEnergy ? unit : `${unit}/h`}` : '—',
      change: isEnergy && peakUsageTime ? `at ${peakUsageTime}` : '',
      isPositive: true, hideArrow: true,
    },
    {
      id: '3', title: "Today's Cost",
      value: todaysCost !== null ? `${todaysCost.toFixed(3)} DT` : '—',
      change: costDeltaPct !== null ? `${Math.abs(costDeltaPct).toFixed(1)}%` : (isEnergy ? 'vs yesterday' : ''),
      isPositive: costDeltaPct !== null ? costDeltaPct >= 0 : true,
      hideArrow: costDeltaPct === null,
      subtext: costDeltaPct !== null ? 'vs yesterday' : undefined,
    },
    {
      id: '4', title: 'Daily Average',
      value: avgDailyKwh !== null ? `${conv(avgDailyKwh).toFixed(1)} ${unit}` : '—',
      change: '30-day average', isPositive: true, hideArrow: true,
    },
    {
      id: '5', title: 'Active Anomalies',
      value: activeAnomaliesCount.toString(),
      change: 'last 24h', isPositive: true, hideArrow: true,
    },
  ];

  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => {
      setToastMessage(null);
    }, 3000);
  };

  return (
    <div className="min-h-screen bg-[#f4f5f7] text-slate-800 font-sans antialiased flex flex-col">
      {/* Toast Notification Alert */}
      {toastMessage && (
        <div className="fixed top-5 right-5 z-50 bg-slate-900 text-white text-xs font-semibold px-4 py-3 rounded-xl shadow-2xl flex items-center gap-2 border border-slate-800 animate-fade-in">
          <span className="w-2 h-2 rounded-full bg-indigo-500" />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* Main Layout Container */}
      <div className="flex flex-1 relative">
        {/* Navigation Sidebar */}
        <Sidebar
          collapsed={sidebarCollapsed}
          setCollapsed={setSidebarCollapsed}
          activeItem={activeNavItem}
          setActiveItem={setActiveNavItem}
          mobileOpen={mobileOpen}
          setMobileOpen={setMobileOpen}
          onOpenSettings={() => setIsSettingsOpen(true)}
          sidebarColor={sidebarColor}
        />

        {/* Main Content Area */}
        <div
          className={`flex-1 flex flex-col min-w-0 transition-all duration-300 ${
            sidebarCollapsed ? 'lg:ml-20' : 'lg:ml-64'
          }`}
        >
          {/* Header */}
          <Header
            onMobileMenuToggle={() => setMobileOpen(!mobileOpen)}
            activeNavItem={activeNavItem}
            utilityType={utilityType}
            userName={profile.name}
            userEmail={profile.email}
            userPhoto={profile.photo}
            onOpenSettings={() => setIsSettingsOpen(true)}
          />

          {/* Body Canvas */}
          <main className="flex-1 p-3 lg:p-4 max-w-[1600px] w-full mx-auto space-y-2">
            {activeNavItem === 'chatbot' ? (
              /* Dedicated Full Chatbot Page */
              <ChatbotPage
                sessions={chatSessions}
                setSessions={setChatSessions}
                activeSessionId={activeSessionId}
                setActiveSessionId={setActiveSessionId}
                onLeave={handleLeaveChat}
              />
            ) : activeNavItem === 'demand-forecast' ? (
              /* Dedicated Demand Forecast Page */
              <DemandForecastPage
                utilityType={utilityType}
                setUtilityType={(type) => {
                  setUtilityType(type);
                  showToast(`Switched view to ${type === 'energy' ? 'Energy' : 'Water'} metrics`);
                }}
              />
            ) : activeNavItem === 'anomaly-detection' ? (
              /* Dedicated Anomaly Detection Page */
              <AnomalyDetectionPage
                utilityType={utilityType}
                setUtilityType={(type) => {
                  setUtilityType(type);
                  showToast(`Switched view to ${type === 'energy' ? 'Energy' : 'Water'} metrics`);
                }}
              />
            ) : activeNavItem === 'reports' ? (
              /* Dedicated Reports Page */
              <ReportsPage
                utilityType={utilityType}
                setUtilityType={(type) => {
                  setUtilityType(type);
                  showToast(`Switched view to ${type === 'energy' ? 'Energy' : 'Water'} metrics`);
                }}
              />
            ) : (
              /* Main Dashboard View */
              <>
                {/* SubHeader Toolbar (Utility Switch) */}
                <SubHeaderToolbar
                  utilityType={utilityType}
                  setUtilityType={(type) => {
                    setUtilityType(type);
                    showToast(`Switched view to ${type === 'energy' ? 'Energy' : 'Water'} metrics`);
                  }}
                />

                {dashLoading ? (
                  <div className="flex items-center justify-center py-24 text-slate-400 text-sm">Loading real data…</div>
                ) : (
                  <>
                    {/* 5 Top Metric Cards */}
                    <MetricCards metrics={metrics} utilityType={utilityType} />

                    {/* Middle Section: Status Summary + Performance Line Chart */}
                    <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 lg:gap-3.5 mb-3 items-stretch">
                      {/* Left: Status Summary Card (3 cols) */}
                      <div className="lg:col-span-3 flex flex-col">
                        <StatusSummaryCard utilityType={utilityType} weather={weather} rate={rate} />
                      </div>

                      {/* Right: Performance Line Chart Card (9 cols) */}
                      <div className="lg:col-span-9 flex flex-col">
                        <PerformanceChartCard utilityType={utilityType} data={weekComparisonData} />
                      </div>
                    </div>

                    {/* Bottom Section: Market Overview */}
                    <div className="w-full">
                      <MarketOverviewCard
                        utilityType={utilityType}
                        total={todayConsumption}
                        changePct={todayDeltaPct}
                        data={timeOfDayData}
                      />
                    </div>
                  </>
                )}
              </>
            )}
          </main>
        </div>
      </div>

      {/* Small Floating Bottom-Right Chatbot Circle & Popover Drawer */}
      {activeNavItem !== 'chatbot' && (
        <ChatbotWidget
          sessions={chatSessions}
          setSessions={setChatSessions}
          activeSessionId={activeSessionId}
          setActiveSessionId={setActiveSessionId}
          onOpenFullPage={() => setActiveNavItem('chatbot')}
          onLeave={handleLeaveChat}
        />
      )}

      {/* Settings Modal Drawer (accessible via Sidebar / Settings) */}
      <SettingsModal
        isOpen={isSettingsOpen}
        setIsOpen={setIsSettingsOpen}
        sidebarColor={sidebarColor}
        setSidebarColor={setSidebarColor}
        profile={profile}
        updateProfile={updateProfile}
      />
    </div>
  );
}
