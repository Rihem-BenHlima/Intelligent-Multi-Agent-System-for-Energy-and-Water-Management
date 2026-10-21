// STEG (Tunisia) residential electricity tariff — progressive monthly
// brackets, hors taxes. Same numbers as backend/services/tariff_service.py;
// keep both in sync if the tariff ever changes.
// 0-50 kWh: 62 millimes · 51-100: 96 · 101-200: 176 · 201-300: 218 ·
// 301-500: 341 · >500: 414.
export const STEG_TIERS: { upTo: number; rate: number }[] = [
  { upTo: 50, rate: 0.062 },
  { upTo: 100, rate: 0.096 },
  { upTo: 200, rate: 0.176 },
  { upTo: 300, rate: 0.218 },
  { upTo: 500, rate: 0.341 },
  { upTo: Infinity, rate: 0.414 },
];

// Splits kwhConsumed across the brackets it spans, starting from kwhBefore
// (cumulative kWh already used this MONTH before this chunk).
export function computeTieredCost(kwhBefore: number, kwhConsumed: number): number {
  if (kwhConsumed <= 0) return 0;
  let cursor = kwhBefore;
  let remaining = kwhConsumed;
  let cost = 0;
  for (const { upTo, rate } of STEG_TIERS) {
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

// The rate that applies to the NEXT kWh, given cumulative monthly usage so far.
export function getMarginalRate(cumulativeKwh: number): number {
  for (const { upTo, rate } of STEG_TIERS) {
    if (cumulativeKwh <= upTo) return rate;
  }
  return STEG_TIERS[STEG_TIERS.length - 1].rate;
}
