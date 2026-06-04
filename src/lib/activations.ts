export const relu = (x: number) => (x < 0 ? 0 : x);
export const dRelu = (x: number) => (x < 0 ? 0 : 1);

export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
export const dSigmoid = (x: number) => {
  const s = sigmoid(x);
  return s * (1 - s);
};

export const tanh = (x: number) => Math.tanh(x);
export const dTanh = (x: number) => 1 - Math.tanh(x) ** 2;

export const leakyRelu = (x: number, slope = 0.01) => (x < 0 ? slope * x : x);
export const dLeakyRelu = (x: number, slope = 0.01) => (x < 0 ? slope : 1);
