export function numel(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/** NumPy/PyTorch broadcasting: align from the right; dims must be equal or one is 1. */
export function broadcastShape(a: number[], b: number[]): number[] | null {
  const out: number[] = [];
  const n = Math.max(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const da = a[a.length - 1 - i] ?? 1;
    const db = b[b.length - 1 - i] ?? 1;
    if (da !== db && da !== 1 && db !== 1) return null;
    out.unshift(Math.max(da, db));
  }
  return out;
}

export function canReshape(from: number[], to: number[]): boolean {
  return numel(from) === numel(to);
}
