import { Value } from './autograd';
import { mulberry32 } from './data';

export function mseLoss(pred: Value, target: number): Value {
  return pred.sub(target).pow(2);
}

/** A 2-layer MLP: input → hidden (tanh) → 1 linear output. Scalar-valued,
 *  built on the from-scratch autograd engine so it actually backpropagates. */
export class MLP {
  W1: Value[][];
  b1: Value[];
  W2: Value[];
  b2: Value;
  hidden: number;
  rng: () => number;

  constructor(inDim: number, hidden: number, seed = 1) {
    const r = mulberry32(seed);
    const rn = () => r() - 0.5; // small init in [-0.5, 0.5] for stable training
    this.hidden = hidden;
    this.W1 = Array.from({ length: hidden }, () =>
      Array.from({ length: inDim }, () => new Value(rn())),
    );
    this.b1 = Array.from({ length: hidden }, () => new Value(0));
    this.W2 = Array.from({ length: hidden }, () => new Value(rn()));
    this.b2 = new Value(0);
    this.rng = r; // reused for per-epoch shuffling; deterministic given seed
  }

  /**
   * Forward pass for a single input `x`.
   *   h_j = tanh( sum_k W1[j][k]*x[k] + b1[j] )
   *   out = sum_j W2[j]*h_j + b2
   * (CONTRIBUTION POINT B — implemented here; rewrite using the Value API if
   *  you'd like the exercise.)
   */
  forward(x: number[]): Value {
    const h = this.W1.map((row, j) => {
      let s: Value = this.b1[j];
      for (let k = 0; k < x.length; k++) s = s.add(row[k].mul(x[k]));
      return s.tanh();
    });
    let out: Value = this.b2;
    for (let j = 0; j < this.hidden; j++) out = out.add(this.W2[j].mul(h[j]));
    return out;
  }

  parameters(): Value[] {
    return [...this.W1.flat(), ...this.b1, ...this.W2, this.b2];
  }

  /** One pass of online (per-sample) SGD over the dataset; returns mean loss.
   *  Shuffles the sample order each epoch — like DataLoader(shuffle=True) — so
   *  the optimizer doesn't see all of one class before the other (that ordering
   *  makes online SGD oscillate instead of converge). */
  trainEpoch(X: number[][], y: number[], lr: number): number {
    const params = this.parameters();
    const idx = X.map((_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(this.rng() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    let total = 0;
    for (const i of idx) {
      const loss = mseLoss(this.forward(X[i]), y[i]);
      loss.backward();
      for (const p of params) p.data -= lr * p.grad;
      total += loss.data;
    }
    return total / X.length;
  }
}
