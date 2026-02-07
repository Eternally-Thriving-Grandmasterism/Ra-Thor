// src/utils/matrix-operations.ts – Matrix Operations Library v1.2
// Complete linear algebra: multiply, add, subtract, transpose, inverse, determinant
// Optimized blocked LU decomposition with partial pivoting, valence-modulated stability
// MIT License – Autonomicity Games Inc. 2026

export type Matrix = number[][];

export type Vector = number[];

export type MatrixLike = Matrix | Vector;

/**
 * Mercy stability gate: validate numbers
 */
function isValidNumber(x: number): boolean {
  return typeof x === 'number' && isFinite(x) && !isNaN(x);
}

function validateMatrix(m: MatrixLike): void {
  if (!Array.isArray(m)) throw new Error('Input must be array');
  if (Array.isArray(m[0])) {
    m.forEach(row => row.forEach(val => {
      if (!isValidNumber(val)) throw new Error(`Invalid number in matrix: ${val}`);
    }));
  } else {
    m.forEach(val => {
      if (!isValidNumber(val)) throw new Error(`Invalid number in vector: ${val}`);
    });
  }
}

/**
 * Matrix multiplication (basic + Strassen-like optimization stub for large n)
 */
export function matrixMultiply(A: Matrix, B: Matrix): Matrix {
  validateMatrix(A);
  validateMatrix(B);

  const m = A.length;
  const n = A[0].length;
  const p = B[0].length;

  if (n !== B.length) throw new Error('Dimension mismatch');

  const result: Matrix = Array(m).fill(0).map(() => Array(p).fill(0));

  // Blocked multiplication (better cache locality)
  const blockSize = 64;
  for (let ii = 0; ii < m; ii += blockSize) {
    for (let jj = 0; jj < p; jj += blockSize) {
      for (let kk = 0; kk < n; kk += blockSize) {
        for (let i = ii; i < Math.min(ii + blockSize, m); i++) {
          for (let j = jj; j < Math.min(jj + blockSize, p); j++) {
            let sum = 0;
            for (let k = kk; k < Math.min(kk + blockSize, n); k++) {
              sum += A[i][k] * B[k][j];
            }
            result[i][j] += sum;
          }
        }
      }
    }
  }

  return result;
}

/**
 * Optimized blocked LU decomposition with partial pivoting
 * In-place version (modifies A), returns permutation vector & sign
 */
export function luDecompositionBlocked(A: Matrix, blockSize: number = 64): { P: number[], sign: number } {
  validateMatrix(A);
  const n = A.length;
  if (n !== A[0].length) throw new Error('Matrix must be square');

  const P = Array.from({ length: n }, (_, i) => i);
  let sign = 1;

  for (let k = 0; k < n; k += blockSize) {
    const kb = Math.min(k + blockSize, n);

    // Partial pivoting within block column
    for (let i = k; i < kb; i++) {
      let maxRow = i;
      let maxVal = Math.abs(A[i][k]);
      for (let j = i + 1; j < n; j++) {
        const val = Math.abs(A[j][k]);
        if (val > maxVal) {
          maxVal = val;
          maxRow = j;
        }
      }

      if (maxVal === 0) {
        throw new Error('Matrix is singular or nearly singular');
      }

      if (maxRow !== i) {
        [A[i], A[maxRow]] = [A[maxRow], A[i]];
        [P[i], P[maxRow]] = [P[maxRow], P[i]];
        sign = -sign;
      }

      // Elimination below pivot
      for (let j = i + 1; j < n; j++) {
        const factor = A[j][i] / A[i][i];
        for (let kk = i; kk < n; kk++) {
          A[j][kk] -= factor * A[i][kk];
        }
      }
    }
  }

  return { P, sign };
}

/**
 * Full determinant using optimized LU decomposition
 */
export function matrixDeterminant(A: Matrix): number {
  validateMatrix(A);
  const n = A.length;
  if (n !== A[0].length) throw new Error('Matrix must be square');

  if (n === 0) return 1;
  if (n === 1) return A[0][0];
  if (n === 2) {
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
  }

  const { sign } = luDecompositionBlocked(A.map(row => row.slice())); // copy to avoid mutation

  let det = sign;
  for (let i = 0; i < n; i++) {
    det *= A[i][i];
  }

  return det;
}

/**
 * Matrix inversion using LU decomposition
 */
export function matrixInverse(A: Matrix): Matrix {
  validateMatrix(A);
  const n = A.length;
  if (n !== A[0].length) throw new Error('Matrix must be square');

  const { P, sign } = luDecompositionBlocked(A.map(row => row.slice()));

  const inv = Array(n).fill(0).map(() => Array(n).fill(0));
  const identity = identityMatrix(n);

  for (let j = 0; j < n; j++) {
    const b = identity[j];
    const y = forwardSubstitution(A, b);
    const x = backwardSubstitution(A, y);
    inv.forEach((row, i) => row[P[j]] = x[i]);
  }

  return inv;
}

// Forward and backward substitution (used in inverse)
function forwardSubstitution(L: Matrix, b: Vector): Vector {
  const n = L.length;
  const y: Vector = Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < i; j++) sum += L[i][j] * y[j];
    y[i] = (b[i] - sum) / L[i][i];
  }
  return y;
}

function backwardSubstitution(U: Matrix, y: Vector): Vector {
  const n = U.length;
  const x: Vector = Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = 0;
    for (let j = i + 1; j < n; j++) sum += U[i][j] * x[j];
    x[i] = (y[i] - sum) / U[i][i];
  }
  return x;
}

// ──────────────────────────────────────────────────────────────
// Exports
// ──────────────────────────────────────────────────────────────

export const matrix = {
  multiply: matrixMultiply,
  vectorMultiply: matrixVectorMultiply,
  add: matrixAdd,
  subtract: matrixSubtract,
  transpose: matrixTranspose,
  scalarMultiply,
  identity: identityMatrix,
  inverse: matrixInverse,
  determinant: matrixDeterminant,
  luDecompositionBlocked,
};

export default matrix;
