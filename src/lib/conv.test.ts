import { describe, it, expect } from 'vitest';
import { convolve2d, KERNELS } from './conv';

describe('convolve2d', () => {
  it('valid conv shrinks output by (kernel - 1)', () => {
    const img = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ];
    const identity = [
      [0, 0, 0],
      [0, 1, 0],
      [0, 0, 0],
    ];
    const out = convolve2d(img, identity);
    expect(out.length).toBe(1);
    expect(out[0].length).toBe(1);
    expect(out[0][0]).toBe(5); // center pixel passes through
  });
  it('blur kernel averages the neighborhood', () => {
    const img = [
      [3, 3, 3],
      [3, 3, 3],
      [3, 3, 3],
    ];
    const out = convolve2d(img, KERNELS.blur);
    expect(out[0][0]).toBeCloseTo(3);
  });
  it('ships named 3x3 kernels', () => {
    expect(KERNELS.edge.length).toBe(3);
    expect(KERNELS.sharpen.length).toBe(3);
  });
});
