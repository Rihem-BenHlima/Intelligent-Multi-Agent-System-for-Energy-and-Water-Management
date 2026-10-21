// SONEDE (Tunisia) residential water tariff — progressive TRIMESTER
// (calendar-quarter) brackets. Same numbers as
// backend/services/water_tariff_service.py; keep both in sync.
//
// Everything in this app is LITRES. SONEDE publishes per m³ (= per 1000 L);
// converted here to litre boundaries and per-litre rates:
//   0–20 000 L: 0.200 TND/1000L · 20 000–40 000: 0.740 · 40 000–70 000: 1.040 ·
//   70 000–100 000: 1.490 · 100 000–150 000: 1.770 · >150 000: 2.310.
// Per JORT n°033, effective 2024-03-01.
export const SONEDE_TIERS: { upTo: number; rate: number }[] = [
  { upTo: 20_000, rate: 0.000200 },
  { upTo: 40_000, rate: 0.000740 },
  { upTo: 70_000, rate: 0.001040 },
  { upTo: 100_000, rate: 0.001490 },
  { upTo: 150_000, rate: 0.001770 },
  { upTo: Infinity, rate: 0.002310 },
];

// Splits litresConsumed across the brackets it spans, starting from
// litresBefore (cumulative litres already used this TRIMESTER before this chunk).
export function computeTieredCost(litresBefore: number, litresConsumed: number): number {
  if (litresConsumed <= 0) return 0;
  let cursor = litresBefore;
  let remaining = litresConsumed;
  let cost = 0;
  for (const { upTo, rate } of SONEDE_TIERS) {
    if (remaining <= 0) break;
    if (cursor >= upTo) continue;
    const roomInTier = upTo - cursor;
    const amount = Math.min(remaining, roomInTier);
    cost += amount * rate;
    cursor += amount;
    remaining -= amount;
  }
  return cost;
}

// The TND-per-litre rate that applies to the NEXT litre, given cumulative
// trimester usage (litres) so far.
export function getMarginalRate(cumulativeLitres: number): number {
  for (const { upTo, rate } of SONEDE_TIERS) {
    if (cumulativeLitres <= upTo) return rate;
  }
  return SONEDE_TIERS[SONEDE_TIERS.length - 1].rate;
}
