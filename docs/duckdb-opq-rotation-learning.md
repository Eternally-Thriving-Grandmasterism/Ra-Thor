# DuckDB Vector Extension – OPQ Rotation Learning Deep Dive (Rathor.ai 2026 Edition)

## What is OPQ?
Optimized Product Quantization (OPQ) adds a learnable orthogonal rotation matrix R before standard PQ. This rotation aligns the vector space so that sub-vector quantization error is minimized.

## Mathematical Foundation
Given a vector \(\mathbf{x} \in \mathbb{R}^d\):

1. Apply rotation: \(\mathbf{x}' = R \mathbf{x}\)
2. Split into m sub-vectors of dimension d/m
3. Quantize each sub-vector independently to a codebook of size k

The rotation R is learned by minimizing the quantization distortion:
\[
R^* = \arg\min_R \sum_i \| \mathbf{x}_i - Q(R \mathbf{x}_i) \|^2
\]
where Q is the product quantizer.

## Training the Rotation (Iterative Algorithm)
1. Initialize R as identity matrix
2. Run standard PQ on rotated vectors
3. Update R via singular value decomposition (SVD) on the covariance of residuals
4. Repeat until convergence (usually 10–20 iterations)

## DuckDB SQL Usage
```sql
-- Create OPQ index (DuckDB automatically learns the rotation)
CREATE INDEX lumenas_opq ON lumenas_vectors 
USING hnsw (embedding) 
WITH (quantization = 'opq', m = 8, k = 256, rotation_iters = 15);
