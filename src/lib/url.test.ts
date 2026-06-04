import { describe, it, expect } from 'vitest';
import { withBase } from './url';

describe('withBase', () => {
  it('prefixes a root-relative path with the base', () => {
    expect(withBase('/lessons/intro', '/interactive-pytorch')).toBe('/interactive-pytorch/lessons/intro');
  });
  it('avoids double slashes', () => {
    expect(withBase('/x', '/interactive-pytorch/')).toBe('/interactive-pytorch/x');
  });
  it('handles empty base', () => {
    expect(withBase('/x', '')).toBe('/x');
  });
  it('adds a leading slash to a bare path', () => {
    expect(withBase('lessons/a', '/base')).toBe('/base/lessons/a');
  });
});
