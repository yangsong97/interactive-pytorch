export interface Optimizer {
  params: number[];
  velocity: number[]; // per-param momentum buffer (zeros when momentum unused)
}

/**
 * One in-place gradient-descent update over all params.
 *
 *   - momentum === 0 → plain SGD:      p_i -= lr * grad_i
 *   - momentum  >  0 → SGD + momentum:  v_i = momentum * v_i + grad_i ;  p_i -= lr * v_i
 *
 * The momentum form also covers plain SGD: when momentum === 0 the velocity
 * reduces to grad_i, so a single unified loop handles both cases — no branch.
 *
 * Design note: momentum lets the "ball" build speed across consistent gradient
 * directions and roll through small bumps (faster, but can overshoot); plain
 * SGD is steadier but slower in long valleys. The Gradient Descent widget lets
 * learners feel this trade-off by moving the momentum slider.
 */
export function sgdStep(opt: Optimizer, grads: number[], lr: number, momentum = 0): void {
  for (let i = 0; i < opt.params.length; i++) {
    opt.velocity[i] = momentum * opt.velocity[i] + grads[i];
    opt.params[i] -= lr * opt.velocity[i];
  }
}
