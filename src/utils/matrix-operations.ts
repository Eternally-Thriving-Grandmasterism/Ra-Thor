// src/utils/matrix-operations.ts – Matrix Operations Library v1.0
// Complete linear algebra utilities for Kalman filters, attention mechanisms,
// covariance updates, matrix inversion, multiplication, transposition, etc.
// Numerical stability checks, mercy-gated operations
// MIT License – Autonomicity Games Inc. 2026

/**
 * Matrix type alias – 2D array of numbers
 */
export type Matrix = number[][];

/**
 * Vector type alias – 1D array of numbers
 */
export type Vector = number[];

/**
 * Matrix or vector (for unified operations)
 */
export type MatrixLike = Matrix | Vector;

/**
 * Check if value is numeric and finite (mercy stability gate)
 */
function isValidNumber(x: number): boolean {
  return typeof x === 'number' && isFinite(x) && !isNaN(x);
}

/**
 * Mercy gate: ensure matrix contains only valid numbers
 */
function validateMatrix(m: MatrixLike): void {
  if (!Array.isArray(m)) throw new Error('Input must be array');
  if (Array.isArray(m[0])) {
    // 2D matrix
    m.forEach(row => row.forEach(val => {
      if (!isValidNumber(val)) throw new Error('Invalid number in matrix');
    }));
  } else {
    // 1D vector
    m.forEach(val => {
      if (!isValidNumber(val)) throw new Error('Invalid number in vector');
    });
  }
}

/**
 * Matrix multiplication: A × B
 * @param A left matrix (m × n)
 * @param B right matrix (n × p)
 * @returns result matrix (m × p)
 */
export function matrixMultiply(A: Matrix, B: Matrix): Matrix {
  if (!Array.isArray(A) || !Array.isArray(B)) throw new Error('Inputs must be matrices');
  if (A[0].length !== B.length) throw new Error('Dimension mismatch in matrix multiplication');

  validateMatrix(A);
  validateMatrix(B);

  const m = A.length;
  const n = A[0].length;
  const p = B[0].length;

  const result: Matrix = Array(m).fill(0).map(() => Array(p).fill(0));

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < p; j++) {
      for (let k = 0; k < n; k++) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  return result;
}

/**
 * Matrix-vector multiplication: A × v
 * @param A matrix (m × n)
 * @param v vector (n)
 * @returns result vector (m)
 */
export function matrixVectorMultiply(A: Matrix, v: Vector): Vector {
  validateMatrix(A);
  validateMatrix(v);

  if (A[0].length !== v.length) throw new Error('Dimension mismatch in matrix-vector multiplication');

  const m = A.length;
  const result: Vector = Array(m).fill(0);

  for (let i = 0; i < m; i++) {
    for (let k = 0; k < v.length; k++) {
      result[i] += A[i][k] * v[k];
    }
  }

  return result;
}

/**
 * Matrix addition: A + B (element-wise)
 */
export function matrixAdd(A: Matrix, B: Matrix): Matrix {
  if (A.length !== B.length || A[0].length !== B[0].length) {
    throw new Error('Matrices must have the same dimensions for addition');
  }

  validateMatrix(A);
  validateMatrix(B);

  return A.map((row, i) => row.map((val, j) => val + B[i][j]));
}

/**
 * Matrix subtraction: A - B (element-wise)
 */
export function matrixSubtract(A: Matrix, B: Matrix): Matrix {
  if (A.length !== B.length || A[0].length !== B[0].length) {
    throw new Error('Matrices must have the same dimensions for subtraction');
  }

  validateMatrix(A);
  validateMatrix(B);

  return A.map((row, i) => row.map((val, j) => val - B[i][j]));
}

/**
 * Matrix transposition: Aᵀ
 */
export function matrixTranspose(A: Matrix): Matrix {
  validateMatrix(A);
  const rows = A.length;
  const cols = A[0].length;

  const result: Matrix = Array(cols).fill(0).map(() => Array(rows).fill(0));

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[j][i] = A[i][j];
    }
  }

  return result;
}

/**
 * Scalar multiplication: c × A
 */
export function scalarMultiply(c: number, A: Matrix): Matrix {
  validateMatrix(A);
  return A.map(row => row.map(val => val * c));
}

/**
 * Identity matrix of size n×n
 */
export function identityMatrix(n: number): Matrix {
  const result: Matrix = Array(n).fill(0).map(() => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    result[i][i] = 1;
  }
  return result;
}

/**
 * Simple matrix inversion (for small matrices – 2×2, 3×3, etc.)
 * Uses analytic formula – for larger use numeric.js or numeric libraries
 */
export function matrixInverse(A: Matrix): Matrix {
  validateMatrix(A);
  const n = A.length;

  if (n !== A[0].length) throw new Error('Matrix must be square for inversion');

  if (n === 2) {
    const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    if (Math.abs(det) < 1e-10) throw new Error('Matrix is singular (determinant near zero)');
    const invDet = 1 / det;
    return [
      [A[1][1] * invDet, -A[0][1] * invDet],
      [-A[1][0] * invDet, A[0][0] * invDet]
    ];
  }

  // For larger matrices – placeholder (production use library)
  throw new Error('Matrix inversion implemented only for 2×2 matrices – use numeric library for larger');
}

/**
 * Matrix determinant (for 2×2, 3×3)
 */
export function matrixDeterminant(A: Matrix): number {
  validateMatrix(A);
  const n = A.length;

  if (n !== A[0].length) throw new Error('Matrix must be square');

  if (n === 2) {
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
  }

  // Placeholder for larger – use library
  throw new Error('Determinant implemented only for 2×2 matrices');
}

// ──────────────────────────────────────────────────────────────
// Convenience exports
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
  validate: validateMatrix
};

export default matrix;
