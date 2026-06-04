import { describe, it, expect } from 'vitest';
import { MLP, mseLoss } from './mlp';
import { makeMoons } from './data';

function epochLoss(net: MLP, X: [number, number][], y: number[]): number {
  let s = 0;
  for (let i = 0; i < X.length; i++) s += mseLoss(net.forward(X[i]), y[i]).data;
  return s / X.length;
}

describe('MLP', () => {
  it('forward returns one scalar output Value per input', () => {
    const net = new MLP(2, 4, 1234);
    const out = net.forward([0.5, -0.3]);
    expect(typeof out.data).toBe('number');
    expect(Number.isFinite(out.data)).toBe(true);
  });

  it('training on moons reduces loss substantially', () => {
    const { X, y } = makeMoons(80, 0.1, 3);
    const net = new MLP(2, 8, 7);
    const before = epochLoss(net, X, y);
    for (let e = 0; e < 60; e++) net.trainEpoch(X, y, 0.1);
    const after = epochLoss(net, X, y);
    expect(after).toBeLessThan(before * 0.6); // at least 40% lower
  });
});
