import { describe, it, expect } from 'vitest';
import { relu, sigmoid, tanh, leakyRelu, dRelu, dSigmoid } from './activations';

describe('activations', () => {
  it('relu', () => {
    expect(relu(-1)).toBe(0);
    expect(relu(2)).toBe(2);
  });
  it('sigmoid(0)=0.5', () => {
    expect(sigmoid(0)).toBeCloseTo(0.5);
  });
  it('tanh(0)=0', () => {
    expect(tanh(0)).toBeCloseTo(0);
  });
  it('leakyRelu negative slope', () => {
    expect(leakyRelu(-10, 0.1)).toBeCloseTo(-1);
  });
  it('derivatives', () => {
    expect(dRelu(3)).toBe(1);
    expect(dRelu(-3)).toBe(0);
    expect(dSigmoid(0)).toBeCloseTo(0.25);
  });
});
