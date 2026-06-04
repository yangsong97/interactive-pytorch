import { describe, it, expect } from 'vitest';
import { V } from './autograd';

describe('scalar autograd', () => {
  it('computes d/dx of x^2 at x=3 is 6', () => {
    const x = V(3);
    const y = x.mul(x);
    y.backward();
    expect(y.data).toBe(9);
    expect(x.grad).toBeCloseTo(6);
  });
  it('chains add/mul: f = (x*w+b), df/dw = x', () => {
    const x = V(2), w = V(-1), b = V(0.5);
    const f = x.mul(w).add(b);
    f.backward();
    expect(f.data).toBeCloseTo(-1.5);
    expect(w.grad).toBeCloseTo(2); // df/dw = x
    expect(x.grad).toBeCloseTo(-1); // df/dx = w
    expect(b.grad).toBeCloseTo(1);
  });
  it('relu blocks gradient when input < 0', () => {
    const a = V(-2);
    const r = a.relu();
    r.backward();
    expect(r.data).toBe(0);
    expect(a.grad).toBe(0);
  });
  it('tanh derivative is 1 - tanh^2', () => {
    const a = V(0.5);
    const t = a.tanh();
    t.backward();
    expect(a.grad).toBeCloseTo(1 - Math.tanh(0.5) ** 2);
  });
});
