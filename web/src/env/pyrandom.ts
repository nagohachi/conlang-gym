/**
 * A port of CPython 3.12's `random.Random` (Modules/_randommodule.c and Lib/random.py) that
 * produces the same streams for the same integer seed.
 *
 * Only the methods that conlang-gym uses are ported, each with CPython's algorithm, because the
 * number of generator calls matters as much as their results: `getrandbits`, `_randbelow`
 * (`_randbelow_with_getrandbits`), `choice`, `shuffle`, `sample` and `random`.
 */

const N = 624;
const M = 397;
const MATRIX_A = 0x9908b0df;
const UPPER_MASK = 0x80000000;
const LOWER_MASK = 0x7fffffff;

export type PySeed = number | bigint;

/** Split abs(seed) into 32-bit words, least significant first, as random_seed() does. */
function seedToKey(seed: PySeed): number[] {
  let value: bigint;
  if (typeof seed === "bigint") {
    value = seed;
  } else {
    if (!Number.isInteger(seed)) {
      throw new TypeError(`PyRandom seed must be an integer, got ${seed}`);
    }
    value = BigInt(seed);
  }
  if (value < 0n) value = -value;
  // keyused = 1 for 0, otherwise ceil(bit_length / 32).
  const key: number[] = [];
  do {
    key.push(Number(value & 0xffffffffn));
    value >>= 32n;
  } while (value > 0n);
  return key;
}

export class PyRandom {
  private readonly mt = new Uint32Array(N);
  private index = N;

  /** Seed like `random.Random(seed)` with an int seed (the sign is ignored, as in CPython). */
  constructor(seed: PySeed) {
    this.initByArray(seedToKey(seed));
  }

  private initGenrand(s: number): void {
    const mt = this.mt;
    mt[0] = s >>> 0;
    for (let i = 1; i < N; i++) {
      const prev = mt[i - 1]!;
      mt[i] = (Math.imul(1812433253, (prev ^ (prev >>> 30)) >>> 0) + i) >>> 0;
    }
    this.index = N;
  }

  private initByArray(key: readonly number[]): void {
    const mt = this.mt;
    this.initGenrand(19650218);
    let i = 1;
    let j = 0;
    for (let k = Math.max(N, key.length); k > 0; k--) {
      const prev = mt[i - 1]!;
      mt[i] = ((mt[i]! ^ Math.imul((prev ^ (prev >>> 30)) >>> 0, 1664525)) + key[j]! + j) >>> 0;
      i++;
      j++;
      if (i >= N) {
        mt[0] = mt[N - 1]!;
        i = 1;
      }
      if (j >= key.length) j = 0;
    }
    for (let k = N - 1; k > 0; k--) {
      const prev = mt[i - 1]!;
      mt[i] = ((mt[i]! ^ Math.imul((prev ^ (prev >>> 30)) >>> 0, 1566083941)) - i) >>> 0;
      i++;
      if (i >= N) {
        mt[0] = mt[N - 1]!;
        i = 1;
      }
    }
    mt[0] = 0x80000000; // MSB is 1, assuring a non-zero initial array
  }

  /** Return the next 32-bit output of MT19937 (genrand_uint32). */
  private genrandUint32(): number {
    const mt = this.mt;
    if (this.index >= N) {
      let kk = 0;
      let y: number;
      for (; kk < N - M; kk++) {
        y = (mt[kk]! & UPPER_MASK) | (mt[kk + 1]! & LOWER_MASK);
        mt[kk] = mt[kk + M]! ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0);
      }
      for (; kk < N - 1; kk++) {
        y = (mt[kk]! & UPPER_MASK) | (mt[kk + 1]! & LOWER_MASK);
        mt[kk] = mt[kk + (M - N)]! ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0);
      }
      y = (mt[N - 1]! & UPPER_MASK) | (mt[0]! & LOWER_MASK);
      mt[N - 1] = mt[M - 1]! ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0);
      this.index = 0;
    }
    let y = mt[this.index++]!;
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  /** Return a float in [0, 1) with 53 random bits, like `random.random()`. */
  random(): number {
    const a = this.genrandUint32() >>> 5;
    const b = this.genrandUint32() >>> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
  }

  /** Return an int with k random bits, like `random.getrandbits(k)`, for 0 <= k <= 53. */
  getrandbits(k: number): number {
    if (!Number.isInteger(k) || k < 0) throw new RangeError("number of bits must be non-negative");
    if (k === 0) return 0;
    if (k <= 32) return this.genrandUint32() >>> (32 - k);
    if (k <= 53) {
      // Two words, least significant first; the last one keeps its top k - 32 bits.
      const low = this.genrandUint32();
      const high = this.genrandUint32() >>> (64 - k);
      return high * 4294967296 + low;
    }
    throw new RangeError("getrandbits(k) with k > 53 does not fit a number; use getrandbitsBig");
  }

  /** Return an int with k random bits as a bigint, like `random.getrandbits(k)` for any k. */
  getrandbitsBig(k: number): bigint {
    if (!Number.isInteger(k) || k < 0) throw new RangeError("number of bits must be non-negative");
    if (k <= 32) return BigInt(this.getrandbits(k));
    let result = 0n;
    let shift = 0n;
    for (let remaining = k; remaining > 0; remaining -= 32) {
      let word = this.genrandUint32();
      if (remaining < 32) word >>>= 32 - remaining;
      result |= BigInt(word) << shift;
      shift += 32n;
    }
    return result;
  }

  /** Return a random int in [0, n), n > 0 (`_randbelow_with_getrandbits`). */
  randbelow(n: number): number {
    if (!Number.isInteger(n) || n <= 0 || n > Number.MAX_SAFE_INTEGER) {
      throw new RangeError(`randbelow(${n}): n must be a positive safe integer`);
    }
    const k = bitLength(n);
    let r = this.getrandbits(k);
    while (r >= n) r = this.getrandbits(k);
    return r;
  }

  /** Return a random element of a non-empty sequence, like `random.choice`. */
  choice<T>(seq: ArrayLike<T>): T {
    if (!seq.length) throw new RangeError("Cannot choose from an empty sequence");
    return seq[this.randbelow(seq.length)]!;
  }

  /** Shuffle the array in place, like `random.shuffle`. */
  shuffle<T>(x: T[]): void {
    for (let i = x.length - 1; i >= 1; i--) {
      const j = this.randbelow(i + 1);
      const tmp = x[i]!;
      x[i] = x[j]!;
      x[j] = tmp;
    }
  }

  /** Return k distinct elements of population, like `random.sample(population, k)`. */
  sample<T>(population: ArrayLike<T>, k: number): T[] {
    const n = population.length;
    if (!Number.isInteger(k) || !(0 <= k && k <= n)) {
      throw new RangeError("Sample larger than population or is negative");
    }
    const result: T[] = new Array<T>(k);
    let setsize = 21; // size of a small set minus size of an empty list
    if (k > 5) setsize += 4 ** ceilLog4(k * 3); // table size for big sets
    if (n <= setsize) {
      // An n-length list is smaller than a k-length set.
      const pool = Array.from(population);
      for (let i = 0; i < k; i++) {
        const j = this.randbelow(n - i);
        result[i] = pool[j]!;
        pool[j] = pool[n - i - 1]!; // move non-selected item into vacancy
      }
    } else {
      const selected = new Set<number>();
      for (let i = 0; i < k; i++) {
        let j = this.randbelow(n);
        while (selected.has(j)) j = this.randbelow(n);
        selected.add(j);
        result[i] = population[j]!;
      }
    }
    return result;
  }
}

/** Return `n.bit_length()` for a positive safe integer. */
function bitLength(n: number): number {
  if (n < 2 ** 32) return 32 - Math.clz32(n);
  return 32 + bitLength(Math.floor(n / 2 ** 32));
}

/**
 * Return `math.ceil(math.log(x, 4))` for an integer x >= 2, computed exactly: the smallest m with
 * 4 ** m >= x. sample() only calls it with x = 3k, which is never a power of 4, so the float
 * formula never lands near an integer and the two agree.
 */
function ceilLog4(x: number): number {
  let m = 0;
  let power = 1;
  while (power < x) {
    power *= 4;
    m++;
  }
  return m;
}
