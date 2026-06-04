import { describe, it, expect } from 'vitest';
import { sgdStep, type Optimizer } from './gradientDescent';

describe('sgdStep', () => {
  it('moves params downhill: p -= lr*grad (momentum=0 → plain SGD)', () => {
    const s: Optimizer = { params: [1, -2], velocity: [0, 0] };
    sgdStep(s, [2, 4], 0.1, 0);
    expect(s.params[0]).toBeCloseTo(0.8); // 1 - 0.1*2
    expect(s.params[1]).toBeCloseTo(-2.4); // -2 - 0.1*4
  });
  it('with momentum, velocity accumulates across steps', () => {
    const s: Optimizer = { params: [0], velocity: [0] };
    sgdStep(s, [1], 0.1, 0.9); // v = 0.9*0 + 1 = 1 ;   p = 0   - 0.1*1   = -0.1
    sgdStep(s, [1], 0.1, 0.9); // v = 0.9*1 + 1 = 1.9 ; p = -0.1 - 0.1*1.9 = -0.29
    expect(s.params[0]).toBeCloseTo(-0.29);
  });
});
