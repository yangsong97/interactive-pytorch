/**
 * A tiny scalar-valued reverse-mode autodiff engine (micrograd-style).
 * Each Value tracks its data, its accumulated gradient, and a closure that
 * knows how to push gradient to its inputs. This is the same chain-rule
 * machinery PyTorch's autograd uses — just scalar instead of tensor valued.
 */
export class Value {
  data: number;
  grad = 0;
  _backward: () => void = () => {};
  _prev: Value[];

  constructor(data: number, _children: Value[] = []) {
    this.data = data;
    this._prev = _children;
  }

  add(o: Value | number): Value {
    const ot = o instanceof Value ? o : new Value(o);
    const out = new Value(this.data + ot.data, [this, ot]);
    out._backward = () => {
      this.grad += out.grad;
      ot.grad += out.grad;
    };
    return out;
  }

  mul(o: Value | number): Value {
    const ot = o instanceof Value ? o : new Value(o);
    const out = new Value(this.data * ot.data, [this, ot]);
    out._backward = () => {
      this.grad += ot.data * out.grad;
      ot.grad += this.data * out.grad;
    };
    return out;
  }

  relu(): Value {
    const out = new Value(this.data < 0 ? 0 : this.data, [this]);
    out._backward = () => {
      this.grad += (out.data > 0 ? 1 : 0) * out.grad;
    };
    return out;
  }

  tanh(): Value {
    const t = Math.tanh(this.data);
    const out = new Value(t, [this]);
    out._backward = () => {
      this.grad += (1 - t * t) * out.grad;
    };
    return out;
  }

  sub(o: Value | number): Value {
    const ot = o instanceof Value ? o : new Value(o);
    return this.add(ot.mul(-1));
  }

  pow(p: number): Value {
    const out = new Value(Math.pow(this.data, p), [this]);
    out._backward = () => {
      this.grad += p * Math.pow(this.data, p - 1) * out.grad;
    };
    return out;
  }

  /** Topologically order the graph, then push gradients from this node backward. */
  backward(): void {
    const topo: Value[] = [];
    const seen = new Set<Value>();
    const build = (v: Value) => {
      if (!seen.has(v)) {
        seen.add(v);
        for (const c of v._prev) build(c);
        topo.push(v);
      }
    };
    build(this);
    for (const v of topo) v.grad = 0;
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
  }
}

export const V = (x: number) => new Value(x);
