import React from 'react';
import { MetricItem } from '../types';
import { Triangle } from 'lucide-react';

interface MetricCardsProps {
  metrics: MetricItem[];
  utilityType?: 'energy' | 'water';
}

const formatMetricValue = (value: string, isWater: boolean): string => {
  let res = value.replace(/\b(\d+)\s+alerts?\b/gi, (_, countStr) => {
    const count = parseInt(countStr, 10);
    const label = count === 0 || count === 1 ? 'Alert' : 'Alerts';
    return `${count} ${label}`;
  });

  if (isWater) {
    res = res.replace(/kWh/g, 'L');
  }

  return res;
};

export const MetricCards: React.FC<MetricCardsProps> = ({
  metrics,
  utilityType = 'energy',
}) => {
  const isWater = utilityType === 'water';

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3.5 lg:gap-4 mb-3">
      {metrics.map((metric) => (
        <div
          key={metric.id}
          className="bg-white border border-slate-100 rounded-lg p-3.5 lg:p-4 shadow-2xs hover:shadow-xs transition-shadow flex flex-col justify-between"
          id={`metric-card-${metric.id}`}
        >
          <p className="text-[11px] lg:text-xs text-slate-400 font-medium truncate mb-1">
            {metric.title}
          </p>

          <h3 className="text-xl lg:text-2xl font-bold text-slate-800 tracking-tight my-1 truncate">
            {formatMetricValue(metric.value, isWater)}
          </h3>

          <div className="flex items-center gap-1 text-[11px] font-semibold flex-wrap min-h-[16px]">
            {metric.hideArrow ? (
              <span className="text-slate-400 font-normal text-[11px] truncate">
                {metric.change || ' '}
              </span>
            ) : (
              <>
                <div className="flex items-center gap-0.5">
                  <Triangle
                    className={`w-2 h-2 fill-current ${
                      metric.isPositive
                        ? 'text-emerald-500 rotate-0'
                        : 'text-rose-500 rotate-180'
                    }`}
                  />
                  <span
                    className={metric.isPositive ? 'text-emerald-600 font-semibold' : 'text-rose-500 font-semibold'}
                  >
                    {metric.change}
                  </span>
                </div>
                {metric.subtext && (
                  <span className="text-slate-400 font-normal text-[11px] truncate">
                    {metric.subtext}
                  </span>
                )}
              </>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};
