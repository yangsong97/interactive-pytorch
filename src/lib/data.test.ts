import { describe, it, expect } from 'vitest';
import { mulberry32, makeMoons } from './data';

describe('data', () => {
  it('mulberry32 is deterministic for a seed', () => {
    const a = mulberry32(42), b = mulberry32(42);
    expect(a()).toBeCloseTo(b());
  });
  it('mulberry32 returns values in [0,1)', () => {
    const r = mulberry32(1);
    for (let i = 0; i < 100; i++) {
      const v = r();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });
  it('makeMoons returns n points with binary labels in range', () => {
    const { X, y } = makeMoons(50, 0.1, 7);
    expect(X.length).toBe(50);
    expect(y.length).toBe(50);
    expect(y.every((v) => v === 0 || v === 1)).toBe(true);
    expect(X.every(([a, b]) => Number.isFinite(a) && Number.isFinite(b))).toBe(true);
  });
});
