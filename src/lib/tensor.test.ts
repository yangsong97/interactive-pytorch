import { describe, it, expect } from 'vitest';
import { broadcastShape, canReshape, numel } from './tensor';

describe('tensor shape utils', () => {
  it('numel multiplies dims', () => {
    expect(numel([2, 3, 4])).toBe(24);
    expect(numel([])).toBe(1);
  });
  it('broadcasts trailing-aligned shapes', () => {
    expect(broadcastShape([3, 1], [1, 4])).toEqual([3, 4]);
    expect(broadcastShape([5], [3, 5])).toEqual([3, 5]);
    expect(broadcastShape([1], [2, 3, 4])).toEqual([2, 3, 4]);
  });
  it('returns null for incompatible shapes', () => {
    expect(broadcastShape([2, 3], [4, 3, 2])).toBeNull(); // 2 vs 3 mismatch
  });
  it('canReshape only when element counts match', () => {
    expect(canReshape([2, 6], [3, 4])).toBe(true);
    expect(canReshape([2, 6], [5, 2])).toBe(false);
  });
});
