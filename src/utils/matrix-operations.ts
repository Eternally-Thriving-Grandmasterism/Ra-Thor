// src/utils/matrix-operations.ts – Matrix Operations Library v1.1
// Complete linear algebra: multiply, add, subtract, transpose, inverse (2×2), determinant (any size)
// Laplace cofactor + LU decomposition, mercy-gated numerical stability, valence precision checks
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
 * Mercy stability gate: ensure value is finite & valid
 */
function isValidNumber(x: number): boolean {
  return typeof x === 'number' && isFinite(x) && !isNaN(x);
}

/**
 * Validate matrix/vector – throw on invalid numbers
 */
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
 * Matrix multiplication: A × B
 */
export function matrixMultiply(A: Matrix, B: Matrix): Matrix {
  validateMatrix(A);
  validateMatrix(B);

  const m = A.length;
  const n = A[0].length;
  const p = B[0].length;

  if (n !== B.length) throw new Error('Dimension mismatch in matrix multiplication');

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
 */
export function matrixVectorMultiply(A: Matrix, v: Vector): Vector {
  validateMatrix(A);
  validateMatrix(v);

  if (A[0].length !== v.length) throw new Error('Dimension mismatch');

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
 * Matrix addition: A + B
 */
export function matrixAdd(A: Matrix, B: Matrix): Matrix {
  if (A.length !== B.length || A[0].length !== B[0].length) {
    throw new Error('Matrices must have same dimensions for addition');
  }
  validateMatrix(A);
  validateMatrix(B);
  return A.map((row, i) => row.map((val, j) => val + B[i][j]));
}

/**
 * Matrix subtraction: A - B
 */
export function matrixSubtract(A: Matrix, B: Matrix): Matrix {
  if (A.length !== B.length || A[0].length !== B[0].length) {
    throw new Error('Matrices must have same dimensions for subtraction');
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
  for (let i = 0; i < n; i++) result[i][i] = 1;
  return result;
}

/**
 * Full determinant computation (Laplace expansion for small, LU for larger)
 * @param A square matrix
 * @returns determinant (number)
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

  // Laplace expansion (recursive) for small matrices (n ≤ 4)
  if (n <= 4) {
    let det = 0;
    for (let j = 0; j < n; j++) {
      const cofactor = matrixDeterminant(minor(A, 0, j)) * (j % 2 === 0 ? 1 : -1) * A[0][j];
      det += cofactor;
    }
    return det;
  }

  // LU decomposition for larger matrices (numerical)
  try {
    const { L, U, P } = luDecomposition(A);
    let det = 1;
    for (let i = 0; i < n; i++) det *= U[i][i];
    // Account for row swaps in permutation matrix P
    let sign = 1;
    for (let i = 0; i < n; i++) {
      if (P[i] !== i) sign = -sign;
    }
    return sign * det;
  } catch (e) {
    console.warn('[matrixDeterminant] LU failed, falling back to Laplace (slow for large n)');
    let det = 0;
    for (let j = 0; j < n; j++) {
      det += (j % 2 === 0 ? 1 : -1) * A[0][j] * matrixDeterminant(minor(A, 0, j));
    }
    return det;
  }
}

/**
 * Minor matrix (remove row i, column j)
 */
function minor(A: Matrix, row: number, col: number): Matrix {
  return A.filter((_, r) => r !== row).map(row => row.filter((_, c) => c !== col));
}

/**
 * LU decomposition with partial pivoting (for determinant & inversion)
 */
function luDecomposition(A: Matrix): { L: Matrix; U: Matrix; P: number[] } {
  const n = A.length;
  const L: Matrix = Array(n).fill(0).map(() => Array(n).fill(0));
  const U: Matrix = A.map(row => row.slice());
  const P: number[] = Array(n).fill(0).map((_, i) => i);

  for (let i = 0; i < n; i++) L[i][i] = 1;

  for (let i = 0; i < n; i++) {
    // Partial pivoting
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(U[k][i]) > Math.abs(U[maxRow][i])) maxRow = k;
    }

    // Swap rows
    if (maxRow !== i) {
      [U[i], U[maxRow]] = [U[maxRow], U[i]];
      [P[i], P[maxRow]] = [P[maxRow], P[i]];
      [L[i], L[maxRow]] = [L[maxRow], L[i]];
    }

    for (let k = i + 1; k < n; k++) {
      const factor = U[k][i] / U[i][i];
      L[k][i] = factor;
      for (let j = i; j < n; j++) {
        U[k][j] -= factor * U[i][j];
      }
    }
  }

  return { L, U, P };
}

/**
 * Matrix inversion using LU decomposition (for any size)
 */
export function matrixInverse(A: Matrix): Matrix {
  validateMatrix(A);
  const n = A.length;
  if (n !== A[0].length) throw new Error('Matrix must be square');

  const { L, U, P } = luDecomposition(A);

  // Solve for identity columns
  const inv = Array(n).fill(0).map(() => Array(n).fill(0));
  const identity = identityMatrix(n);

  for (let j = 0; j < n; j++) {
    const b = identity[j];
    const y = forwardSubstitution(L, b);
    const x = backwardSubstitution(U, y);
    inv.forEach((row, i) => row[j] = x[i]);
  }

  // Apply permutation
  const permutedInv: Matrix = Array(n).fill(0).map(() => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    permutedInv[P[i]] = inv[i];
  }

  return permutedInv;
}

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
  validate: validateMatrix,
  luDecomposition,
};

export default matrix;
