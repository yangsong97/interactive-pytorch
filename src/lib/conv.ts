/** 2D "valid" cross-correlation (what nn.Conv2d actually computes): slide the
 *  kernel over the image with no padding, so output is (H-kh+1) x (W-kw+1). */
export function convolve2d(img: number[][], kernel: number[][]): number[][] {
  const kh = kernel.length;
  const kw = kernel[0].length;
  const oh = img.length - kh + 1;
  const ow = img[0].length - kw + 1;
  const out: number[][] = [];
  for (let i = 0; i < oh; i++) {
    const row: number[] = [];
    for (let j = 0; j < ow; j++) {
      let s = 0;
      for (let a = 0; a < kh; a++) {
        for (let b = 0; b < kw; b++) s += img[i + a][j + b] * kernel[a][b];
      }
      row.push(s);
    }
    out.push(row);
  }
  return out;
}

export const KERNELS: Record<string, number[][]> = {
  identity: [
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
  ],
  edge: [
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1],
  ],
  blur: [
    [1 / 9, 1 / 9, 1 / 9],
    [1 / 9, 1 / 9, 1 / 9],
    [1 / 9, 1 / 9, 1 / 9],
  ],
  sharpen: [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
  ],
};
