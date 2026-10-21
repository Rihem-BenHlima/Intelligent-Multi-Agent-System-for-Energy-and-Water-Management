import React from 'react';

interface AiInsightsCardProps {
  tips: string[];
  loading?: boolean;
}

export const AiInsightsCard: React.FC<AiInsightsCardProps> = ({ tips, loading = false }) => {
  return (
    <div className="bg-white border border-slate-100 rounded-lg p-3.5 lg:p-4 shadow-2xs flex flex-col justify-between h-full">
      <div>
        <div className="flex items-center gap-2 mb-3 pb-2.5 border-b border-slate-100">
          <h3 className="text-base lg:text-lg font-bold text-slate-800 tracking-tight">AI Insights</h3>
        </div>

        {loading && (
          <div className="text-xs text-slate-400 text-center py-6">Loading…</div>
        )}

        {!loading && tips.length === 0 && (
          <div className="text-xs text-slate-400 text-center py-6">No recommendations yet</div>
        )}

        {!loading && tips.length > 0 && (
          <div className="space-y-3">
            {tips.slice(0, 3).map((tip, i) => (
              <div
                key={i}
                className="bg-slate-50/80 hover:bg-slate-50 rounded-lg p-3 border border-slate-100 transition-all text-xs"
              >
                <p className="text-slate-500 text-sm leading-relaxed">
                  {tip}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
