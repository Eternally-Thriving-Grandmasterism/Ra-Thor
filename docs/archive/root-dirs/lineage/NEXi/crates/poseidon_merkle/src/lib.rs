//! PoseidonMerkle — zk-Friendly Merkle Trees with Full Inclusion Proofs
//! Ultramasterful tree construction + proof generation/verification

use poseidon_hash::PoseidonHash;
use halo2_proofs::arithmetic::Field;
use pasta_curves::pallas::Scalar;

#[derive(Clone, Debug)]
pub struct MerkleProof {
    pub path: Vec<Scalar>,     // Sibling hashes
    pub direction: Vec<bool>,  // true = right sibling
}

pub struct PoseidonMerkleTree {
    hash: PoseidonHash,
    leaves: Vec<Scalar>,
    levels: Vec<Vec<Scalar>>,
}

impl PoseidonMerkleTree {
    pub fn new() -> Self {
        PoseidonMerkleTree {
            hash: PoseidonHash::new(),
            leaves: vec![],
            levels: vec![],
        }
    }

    /// Insert leaf + rebuild tree
    pub fn insert(&mut self, leaf: Scalar) {
        self.leaves.push(leaf);
        self.rebuild();
    }

    /// Rebuild tree from leaves (canonical left-right order)
    fn rebuild(&mut self) {
        let mut current = self.leaves.clone();
        self.levels = vec![current.clone()];

        while current.len() > 1 {
            let mut next = vec![];
            for chunk in current.chunks(2) {
                let left = chunk[0];
                let right = if chunk.len() > 1 { chunk[1] } else { left }; // Duplicate for odd
                let parent = self.hash.hash(&[left, right]);
                next.push(parent);
            }
            current = next;
            self.levels.push(current.clone());
        }
    }

    /// Generate inclusion proof for leaf index
    pub fn prove_inclusion(&self, index: usize) -> Result<MerkleProof, String> {
        if index >= self.leaves.len() {
            return Err("Index out of bounds".to_string());
        }

        let mut path = vec![];
        let mut direction = vec![];
        let mut current_idx = index;

        for level in &self.levels[..self.levels.len() - 1] {
            let sibling_idx = if current_idx % 2 == 0 { current_idx + 1 } else { current_idx - 1 };
            let sibling = if sibling_idx < level.len() {
                level[sibling_idx]
            } else {
                level[current_idx] // Duplicate for padding
            };

            path.push(sibling);
            direction.push(current_idx % 2 == 1); // true if right sibling
            current_idx /= 2;
        }

        Ok(MerkleProof { path, direction })
    }

    /// Verify inclusion proof
    pub fn verify_proof(&self, root: Scalar, leaf: Scalar, proof: &MerkleProof) -> bool {
        let mut current = leaf;
        for (sibling, &right) in proof.path.iter().zip(&proof.direction) {
            current = if right {
                self.hash.hash(&[sibling, current])
            } else {
                self.hash.hash(&[current, sibling])
            };
        }
        current == root
    }

    pub fn root(&self) -> Scalar {
        self.levels.last().map_or(Scalar::zero(), |level| level[0])
    }
}
