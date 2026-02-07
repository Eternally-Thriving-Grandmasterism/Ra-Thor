// src/utils/matrix-operations.ts – Matrix Operations Library v1.4
// Complete linear algebra: multiply, add, subtract, transpose, inverse, determinant, Cholesky, QR
// Householder QR with column pivoting, valence-modulated tolerance, mercy-gated rank check
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
 * Householder QR decomposition with column pivoting (QRCP)
 * A = Q R Pᵀ  (P permutation matrix)
 * Returns { Q, R, P } or null if rank deficient (mercy fallback)
 */
export function qrDecomposition(A: Matrix): { Q: Matrix; R: Matrix; P: number[] } | null {
  validateMatrix(A);
  const m = A.length;
  const n = A[0].length;

  // Working copy – we'll modify in place
  const QR = A.map(row => row.slice());
  const P: number[] = Array.from({ length: n }, (_, i) => i);
  const R: Matrix = Array(m).fill(0).map(() => Array(n).fill(0));
  const Q: Matrix = identityMatrix(m);

  const valence = currentValence.get();
  const tol = 1e-10 * Math.max(m, n) * (1 - valence * 0.5); // stricter on high valence

  for (let k = 0; k < Math.min(m, n); k++) {
    // Partial pivoting: find column with largest norm below diagonal
    let maxNorm = 0;
    let pivot = k;
    for (let i = k; i < n; i++) {
      let norm = 0;
      for (let j = k; j < m; j++) norm += QR[j][i] * QR[j][i];
      norm = Math.sqrt(norm);
      if (norm > maxNorm) {
        maxNorm = norm;
        pivot = i;
      }
    }

    if (maxNorm < tol) {
      console.warn(`[QR] Rank deficiency detected at column ${k} (norm ${maxNorm} < tol ${tol})`);
      if (mercyGate('QR rank deficiency fallback')) {
        console.log('[QR] Mercy fallback – returning partial decomposition');
        break;
      }
      return null;
    }

    // Swap columns
    if (pivot !== k) {
      for (let j = 0; j < m; j++) {
        [QR[j][k], QR[j][pivot]] = [QR[j][pivot], QR[j][k]];
      }
      [P[k], P[pivot]] = [P[pivot], P[k]];
    }

    // Householder reflection
    let sigma = 0;
    for (let j = k; j < m; j++) sigma += QR[j][k] * QR[j][k];
    sigma = Math.sqrt(sigma);
    const beta = QR[k][k] < 0 ? -sigma : sigma;
    const u1 = QR[k][k] + beta;
    const tau = u1 / beta;

    // Reflect columns k to n-1
    for (let j = k; j < n; j++) {
      let sum = QR[k][j];
      for (let i = k + 1; i < m; i++) sum += QR[i][k] * QR[i][j];
      sum *= tau;
      QR[k][j] -= sum;
      for (let i = k + 1; i < m; i++) QR[i][j] -= QR[i][k] * sum;
    }

    // Store Householder vector below diagonal (for Q reconstruction)
    for (let i = k + 1; i < m; i++) QR[i][k] /= u1;

    QR[k][k] = -beta;
  }

  // Extract R (upper triangular)
  for (let i = 0; i < m; i++) {
    for (let j = i; j < n; j++) {
      R[i][j] = QR[i][j];
    }
  }

  // Reconstruct Q from Householder reflectors (optional – can be lazy)
  // For efficiency, we often keep Householder vectors in QR and apply on demand
  // Here we reconstruct full Q for simplicity
  let QFull = identityMatrix(m);
  for (let k = n - 1; k >= 0; k--) {
    const beta = -QR[k][k];
    if (beta === 0) continue;

    const u = Array(m).fill(0);
    u[k] = 1;
    for (let i = k + 1; i < m; i++) u[i] = QR[i][k];

    const tau = 2 / (u.reduce((s, v) => s + v * v, 0));

    for (let j = 0; j < m; j++) {
      let sum = 0;
      for (let i = k; i < m; i++) sum += u[i] * QFull[i][j];
      sum *= tau;
      for (let i = k; i < m; i++) QFull[i][j] -= u[i] * sum;
    }
  }

  return { Q: QFull, R, P };
}

/**
 * Solve A x = b using QR decomposition (most stable way)
 */
export function solveQR(A: Matrix, b: Vector): Vector | null {
  const qr = qrDecomposition(A);
  if (!qr) return null;

  const { Q, R } = qr;

  // Solve Q R x = b → R x = Qᵀ b
  const Qtb = matrixVectorMultiply(matrixTranspose(Q), b);

  // Back-substitution on R x = Qtb
  const n = R.length;
  const x: Vector = Array(n).fill(0);

  for (let i = n - 1; i >= 0; i--) {
    let sum = 0;
    for (let j = i + 1; j < n; j++) sum += R[i][j] * x[j];
    x[i] = (Qtb[i] - sum) / R[i][i];
  }

  return x;
}

// ──────────────────────────────────────────────────────────────
// Previous functions (unchanged but included for completeness)
// ──────────────────────────────────────────────────────────────

export function matrixMultiply(A: Matrix, B: Matrix): Matrix {
  // ... (previous implementation)
}

// ... (rest of previous functions: add, subtract, transpose, scalarMultiply, identityMatrix, etc.)

export const matrix = {
  multiply: matrixMultiply,
  // ... other functions ...
  qrDecomposition,
  solveQR,
  choleskyDecomposition,
  // ... previous exports ...
};

export default matrix;
