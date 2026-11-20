/**
 * A correctly rounded natural logarithm.
 *
 * The solver's entropies go through log(), and JavaScript's Math.log (V8 uses fdlibm) is not
 * correctly rounded: on the solver's inputs count / numHypotheses (1 <= count <= numHypotheses <=
 * 498, 124,251 values) it differs in the last bit from the correctly rounded value 8,418 times.
 * CPython's math.log is the C library's log, which is correctly rounded for almost every input
 * (Apple's libm, which wrote the test fixtures, misses 86 of those values by one ulp; glibc's error
 * bound is 0.519 ulp). So this module computes log in double-double arithmetic (about 100 bits) and
 * rounds once. The solver's choices did not change on 22,000 seeds when log was replaced by
 * Math.log, by Apple's log, or by a log that rounds the other way on all 4,751 inputs within 0.02
 * ulp of a rounding midpoint, so the remaining last-bit differences between C libraries do not
 * matter in practice.
 */

/** Return [s, e] with s = fl(a + b) and s + e = a + b exactly. */
function twoSum(a: number, b: number): [number, number] {
  const s = a + b;
  const bb = s - a;
  const e = a - (s - bb) + (b - bb);
  return [s, e];
}

/** Return [s, e] with s = fl(a + b) and s + e = a + b exactly, for |a| >= |b|. */
function quickTwoSum(a: number, b: number): [number, number] {
  const s = a + b;
  return [s, b - (s - a)];
}

const SPLITTER = 134217729; // 2^27 + 1

/** Return [p, e] with p = fl(a * b) and p + e = a * b exactly (Dekker). */
function twoProd(a: number, b: number): [number, number] {
  const p = a * b;
  let t = SPLITTER * a;
  const ahi = t - (t - a);
  const alo = a - ahi;
  t = SPLITTER * b;
  const bhi = t - (t - b);
  const blo = b - bhi;
  const e = ahi * bhi - p + ahi * blo + alo * bhi + alo * blo;
  return [p, e];
}

type DD = [number, number];

function ddAdd(a: DD, b: DD): DD {
  const [s, e1] = twoSum(a[0], b[0]);
  const [t, e2] = twoSum(a[1], b[1]);
  let [hi, lo] = quickTwoSum(s, e1 + t);
  [hi, lo] = quickTwoSum(hi, lo + e2);
  return [hi, lo];
}

function ddMul(a: DD, b: DD): DD {
  const [p, e] = twoProd(a[0], b[0]);
  return quickTwoSum(p, e + (a[0] * b[1] + a[1] * b[0]));
}

function ddMulNumber(a: DD, b: number): DD {
  const [p, e] = twoProd(a[0], b);
  return quickTwoSum(p, e + a[1] * b);
}

function ddDiv(a: DD, b: DD): DD {
  // Long division: two correction steps give a quotient accurate to double-double precision.
  const q1 = a[0] / b[0];
  let r = ddAdd(a, ddMulNumber(b, -q1));
  const q2 = r[0] / b[0];
  r = ddAdd(r, ddMulNumber(b, -q2));
  const q3 = r[0] / b[0];
  const [hi, lo] = quickTwoSum(q1, q2);
  return ddAdd([hi, lo], [q3, 0]);
}

// ln 2 to 106 bits.
const LN2: DD = [0.6931471805599453, 2.3190468138462996e-17];

const view = new DataView(new ArrayBuffer(8));

/** Split a positive finite x into [m, k] with x = m * 2^k and 1 <= m < 2. */
function frexp1(x: number): [number, number] {
  let scale = 0;
  if (x < 2.2250738585072014e-308) {
    x *= 2 ** 64; // subnormal: scale into the normal range
    scale = -64;
  }
  view.setFloat64(0, x);
  const high = view.getUint32(0);
  const exponent = ((high >>> 20) & 0x7ff) - 1023;
  view.setUint32(0, (high & 0x800fffff) | (1023 << 20));
  return [view.getFloat64(0), exponent + scale];
}

/** Return log(x) of a positive finite x as a double-double. */
function ddLog(x: number): DD {
  let [m, k] = frexp1(x);
  if (m > Math.SQRT2) {
    m /= 2; // exact
    k += 1;
  }
  // log(m) = 2 atanh(s) = 2 (s + s^3/3 + s^5/5 + ...), s = (m - 1) / (m + 1), |s| < 0.172.
  const num: DD = [m - 1, 0]; // exact (Sterbenz)
  const den: DD = twoSum(m, 1);
  const s = ddDiv(num, den);
  const s2 = ddMul(s, s);
  let power = s;
  let sum: DD = s;
  for (let n = 3; n <= 81; n += 2) {
    power = ddMul(power, s2);
    if (Math.abs(power[0]) < 1e-40 * Math.abs(sum[0]) || power[0] === 0) break;
    sum = ddAdd(sum, ddDiv(power, [n, 0]));
  }
  const logM: DD = [sum[0] * 2, sum[1] * 2];
  return ddAdd(ddMulNumber(LN2, k), logM);
}

const cache = new Map<number, number>();

/** Return the natural logarithm of x, correctly rounded to nearest (like a correctly rounded libm). */
export function crLog(x: number): number {
  if (!(x > 0) || !Number.isFinite(x)) return Math.log(x); // 0, negatives, NaN, Infinity
  if (x === 1) return 0;
  let result = cache.get(x);
  if (result === undefined) {
    const [hi, lo] = ddLog(x);
    result = hi + lo;
    if (cache.size >= 1_000_000) cache.clear();
    cache.set(x, result);
  }
  return result;
}
