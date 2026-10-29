//! EternalMercyMesh tests against the real v14 API (session_id + invite).
//! Stale get_or_create_session stubs removed — that method does not exist.

use lattice_conductor_v14::eternal_mercy_mesh::{EternalMercyMesh, EternalMercyMeshConfig};

#[test]
fn two_meshes_keep_distinct_session_ids() {
    let a = EternalMercyMesh::new_eternal("chat_alpha");
    let b = EternalMercyMesh::new_eternal("chat_beta");
    assert_ne!(a.session_id, b.session_id);
}

#[test]
fn invite_adds_a_participant_on_this_mesh_only() {
    let mut a = EternalMercyMesh::new(EternalMercyMeshConfig {
        session_id: "chat_alpha".into(),
        seed_patsagi_councils: false,
        default_coherence: 0.97,
    });
    let b = EternalMercyMesh::new(EternalMercyMeshConfig {
        session_id: "chat_beta".into(),
        seed_patsagi_councils: false,
        default_coherence: 0.97,
    });
    let before_a = a.field.organism_fields.len();
    let before_b = b.field.organism_fields.len();
    a.invite_shared_chat_participant("guest", 0.9);
    assert_eq!(a.field.organism_fields.len(), before_a + 1);
    assert_eq!(b.field.organism_fields.len(), before_b);
}
