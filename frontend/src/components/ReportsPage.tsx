import React, { useEffect, useRef, useState } from 'react';
import {
  FileText,
  Calendar,
  Download,
  FileCheck,
  Eye,
  Trash2,
} from 'lucide-react';
import { SubHeaderToolbar } from './SubHeaderToolbar';
import { API_BASE } from '../config';

interface ReportsPageProps {
  utilityType?: 'energy' | 'water';
  setUtilityType?: (type: 'energy' | 'water') => void;
}

interface ReportEntry {
  id: number;
  domain: string;
  source: 'manual' | 'weekly_cron';
  generated_at: string;
  report_text: string;
}

function formatTimestamp(iso: string) {
  const d = new Date(iso.endsWith('Z') ? iso : iso + 'Z');
  return d.toLocaleString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit', timeZone: 'UTC' });
}

export const ReportsPage: React.FC<ReportsPageProps> = ({
  utilityType = 'energy',
  setUtilityType,
}) => {
  const isEnergy = utilityType === 'energy';

  const [reports, setReports] = useState<ReportEntry[]>([]);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [downloadingId, setDownloadingId] = useState<number | null>(null);
  const [openingId, setOpeningId] = useState<number | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [isDeleting, setIsDeleting] = useState(false);

  const fetchReports = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/reports/${utilityType}`);
      const json = await res.json();
      return (json.reports || []) as ReportEntry[];
    } catch {
      return [];
    }
  };

  useEffect(() => {
    fetchReports().then(setReports);
    setSelectedIds(new Set()); // switching domain shows a different list — stale selections don't carry over
  }, [utilityType]);

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3000);
  };

  const toggleSelected = (id: number) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handleDeleteSelected = async () => {
    if (selectedIds.size === 0) return;
    setIsDeleting(true);
    try {
      const ids = Array.from(selectedIds).join(',');
      const res = await fetch(`${API_BASE}/api/reports?ids=${ids}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const fresh = await fetchReports();
      setReports(fresh);
      showToast(`Deleted ${selectedIds.size} report${selectedIds.size === 1 ? '' : 's'}.`);
      setSelectedIds(new Set());
    } catch (e: any) {
      showToast(e.message || 'Failed to delete report(s)');
    } finally {
      setIsDeleting(false);
    }
  };

  const generatingRef = useRef(false);

  const handleGenerateReport = async () => {
    if (generatingRef.current) return;
    generatingRef.current = true;
    setIsGenerating(true);
    try {
      const res = await fetch(`${API_BASE}/api/report/${utilityType}`);
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      await res.json(); // triggers + persists generation server-side
      const fresh = await fetchReports();
      setReports(fresh);
      showToast(`New ${isEnergy ? 'Energy' : 'Water'} report generated successfully!`);
    } catch (e: any) {
      showToast(e.message || 'Failed to generate report');
    } finally {
      setIsGenerating(false);
      generatingRef.current = false;
    }
  };

  const handleDownloadPDF = async (e: React.MouseEvent, report: ReportEntry) => {
    e.stopPropagation();
    setDownloadingId(report.id);
    try {
      const res = await fetch(`${API_BASE}/api/reports/${report.id}/pdf`);
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `sfmconnect-${utilityType}-report-${report.id}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
      showToast(`Downloaded PDF for report (${formatTimestamp(report.generated_at)})`);
    } catch (e: any) {
      showToast(e.message || 'Failed to download PDF');
    } finally {
      setDownloadingId(null);
    }
  };

  const handleOpenPDF = async (e: React.MouseEvent, report: ReportEntry) => {
    e.stopPropagation();
    setOpeningId(report.id);
    try {
      const res = await fetch(`${API_BASE}/api/reports/${report.id}/pdf`);
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const blob = await res.blob();
      // Open PDF blob in a new browser tab
      const url = URL.createObjectURL(blob);
      window.open(url, '_blank', 'noopener,noreferrer');
      setTimeout(() => URL.revokeObjectURL(url), 30000);
    } catch (e: any) {
      showToast(e.message || 'Failed to open PDF');
    } finally {
      setOpeningId(null);
    }
  };

  const latestDate = reports[0] ? formatTimestamp(reports[0].generated_at) : '—';

  return (
    <div className="space-y-2">
      {/* Toast Notification */}
      {toast && (
        <div className="fixed bottom-6 right-6 z-50 bg-slate-900 text-white text-xs font-semibold px-4 py-3 rounded-lg shadow-2xl flex items-center gap-2 border border-slate-800 animate-fade-in">
          <FileCheck className="w-4 h-4 text-emerald-400" />
          <span>{toast}</span>
        </div>
      )}

      {/* Top Standard SubHeader Toolbar */}
      <SubHeaderToolbar
        utilityType={utilityType}
        setUtilityType={setUtilityType}
      />

      {/* Card 1: Report Generator Card */}
      <div className="bg-white border border-slate-100 rounded-lg p-3.5 lg:p-4 shadow-2xs flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3.5">
          <div className="w-10 h-10 rounded-lg bg-indigo-50 border border-indigo-100 text-[#1f3bb3] flex items-center justify-center shrink-0">
            <FileText className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-800 tracking-tight">
              AI Report — {isEnergy ? 'Electricity (kWh)' : 'Water Usage (Liters)'}
            </h3>
            <p className="text-xs text-slate-400 font-normal mt-0.5">
              {reports.length} report{reports.length === 1 ? '' : 's'} stored · latest {latestDate}
            </p>
          </div>
        </div>

        <button
          onClick={handleGenerateReport}
          disabled={isGenerating}
          className="bg-slate-800 hover:bg-slate-900 active:scale-98 text-white font-semibold text-xs px-4 py-2 rounded-lg flex items-center justify-center gap-2 shadow-2xs transition-all cursor-pointer shrink-0 disabled:opacity-70"
          id="generate-report-btn"
        >
          {isGenerating ? (
            <>
              <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              <span>Generating...</span>
            </>
          ) : (
            <>
              <FileText className="w-4 h-4" />
              <span>Generate Report</span>
            </>
          )}
        </button>
      </div>

      {/* Card 2: Past Reports List */}
      <div className="bg-white border border-slate-100 rounded-lg p-3.5 lg:p-4 shadow-2xs space-y-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-bold text-slate-800">
              Past Reports
            </h3>
            <p className="text-xs text-slate-400 font-normal mt-0.5">
              {reports.length} report{reports.length === 1 ? '' : 's'} stored
            </p>
          </div>

          {/* Only appears once at least one report is selected */}
          {selectedIds.size > 0 && (
            <button
              onClick={handleDeleteSelected}
              disabled={isDeleting}
              className="px-3 py-1.5 rounded-lg bg-rose-50 hover:bg-rose-100 border border-rose-200 text-rose-600 text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer disabled:opacity-60 shrink-0"
            >
              {isDeleting ? (
                <div className="w-3.5 h-3.5 border-2 border-rose-300 border-t-transparent rounded-full animate-spin" />
              ) : (
                <Trash2 className="w-3.5 h-3.5" />
              )}
              <span>{isDeleting ? 'Deleting…' : `Delete Selected (${selectedIds.size})`}</span>
            </button>
          )}
        </div>

        {reports.length === 0 ? (
          <div className="py-8 text-center text-xs text-slate-400">
            No reports yet — click Generate Report above.
          </div>
        ) : (
          <div className="divide-y divide-slate-100 border-t border-slate-100">
            {reports.map((report, i) => (
              <div
                key={report.id}
                onClick={() => toggleSelected(report.id)}
                className={`py-2.5 first:pt-2.5 px-2 -mx-2 rounded-lg flex items-center justify-between gap-3 cursor-pointer transition-colors ${
                  selectedIds.has(report.id) ? 'bg-indigo-50/60' : 'hover:bg-slate-50/60'
                }`}
              >
                <div className="flex items-center gap-2.5 min-w-0">
                  <input
                    type="checkbox"
                    checked={selectedIds.has(report.id)}
                    onChange={() => toggleSelected(report.id)}
                    onClick={(e) => e.stopPropagation()}
                    className="w-3.5 h-3.5 rounded border-slate-300 text-[#1f3bb3] focus:ring-indigo-500 cursor-pointer shrink-0"
                  />
                  <Calendar className="w-4 h-4 text-slate-400 shrink-0" />
                  <span className="text-xs font-semibold text-slate-800 truncate">
                    {formatTimestamp(report.generated_at)}
                  </span>
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  {i === 0 && (
                    <span className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-indigo-50 text-[#1f3bb3] border border-indigo-200">
                      Latest
                    </span>
                  )}

                  <span className="px-2.5 py-0.5 rounded-full text-[10px] font-medium bg-slate-100 text-slate-600">
                    {report.source === 'weekly_cron' ? 'Automated' : 'Manual'}
                  </span>

                  <button
                    onClick={(e) => handleDownloadPDF(e, report)}
                    disabled={downloadingId === report.id}
                    className="p-1.5 rounded-md border border-slate-200 bg-white hover:bg-slate-50 text-slate-400 hover:text-slate-600 shadow-2xs transition-all cursor-pointer disabled:opacity-60"
                    title="Download PDF"
                  >
                    {downloadingId === report.id ? (
                      <div className="w-4 h-4 border-2 border-slate-300 border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <Download className="w-4 h-4" />
                    )}
                  </button>

                  <button
                    onClick={(e) => handleOpenPDF(e, report)}
                    disabled={openingId === report.id}
                    className="p-1.5 rounded-md border border-slate-200 bg-white hover:bg-slate-50 text-slate-400 hover:text-slate-600 shadow-2xs transition-all cursor-pointer disabled:opacity-60"
                    title="View PDF"
                  >
                    {openingId === report.id ? (
                      <div className="w-4 h-4 border-2 border-slate-300 border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <Eye className="w-4 h-4" />
                    )}
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
