// mercy_orchestrator/src/hypergraph_integration.rs — HyperGraphDB Hypergraph Persistence via JNI
use jni::objects::{JClass, JObject, JString, JValue};
use jni::sys::{jlong, jobject};
use jni::{JNIEnv, JavaVM};
use std::error::Error;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum HyperError {
    #[error("JNI error: {0}")]
    Jni(#[from] jni::errors::Error),
    #[error("HyperGraphDB init failed")]
    InitFailed,
    #[error("Invalid handle")]
    InvalidHandle,
}

pub struct HyperGraphDB {
    jvm: JavaVM,
    graph: JObject<'static>,  // HGHandle to HyperGraph
}

impl HyperGraphDB {
    pub fn new(location: &Path) -> Result<Self, Box<dyn Error>> {
        // Attach to existing JVM or create new (simplified; in prod use managed JVM)
        let jvm = unsafe { JavaVM::from_raw(std::ptr::null_mut())? }; // Placeholder: init properly via jni::JavaVM::attach_current_thread or launch
        let env = jvm.attach_current_thread()?;

        // Load HyperGraphDB classes (assume JAR in classpath)
        let config_class: JClass = env.find_class("org/hypergraphdb/HGConfiguration")?;
        let config: JObject = env.new_object(config_class, "()V", &[])?;

        // Set location
        let location_str: JString = env.new_string(location.to_str().unwrap_or(""))?;
        env.call_method(&config, "setStoreLocation", "(Ljava/lang/String;)V", &[location_str.into()])?;

        // Open graph
        let hg_class: JClass = env.find_class("org/hypergraphdb/HyperGraph")?;
        let graph_obj: JObject = env.call_static_method(
            hg_class,
            "open",
            "(Lorg/hypergraphdb/HGConfiguration;)Lorg/hypergraphdb/HyperGraph;",
            &[config.into()],
        )?.l()?;

        Ok(HyperGraphDB {
            jvm,
            graph: graph_obj.into(),
        })
    }

    pub fn add_metta_atom(&self, atom: &str, valence: f64) -> Result<jlong, HyperError> {
        let env = self.jvm.attach_current_thread()?;
        let atom_str: JString = env.new_string(atom)?;

        // Example: add simple value (extend to custom types/links)
        let handle: JObject = env.call_method(
            &self.graph,
            "add",
            "(Ljava/lang/Object;)Lorg/hypergraphdb/HGHandle;",
            &[atom_str.into()],
        )?.l()?;

        // Add valence property link (simplified)
        let valence_obj: JObject = env.new_object("java/lang/Double", "(D)V", &[JValue::Double(valence)])?;
        env.call_method(&self.graph, "add", "(Ljava/lang/Object;)Lorg/hypergraphdb/HGHandle;", &[valence_obj.into()])?;

        let handle_long: jlong = env.call_method(&handle, "getPersistentHandle", "()J", &[])?.j()?;
        Ok(handle_long)
    }

    // TODO: Extend with HGQuery for valence rules, traversal, etc.
    // e.g. pub fn query_high_valence(&self, min: f64) -> Result<Vec<String>, HyperError> { ... }
}

impl Drop for HyperGraphDB {
    fn drop(&mut self) {
        // Close graph on drop (call close() via JNI)
        if let Ok(env) = self.jvm.attach_current_thread() {
            let _ = env.call_method(&self.graph, "close", "()V", &[]);
        }
    }
}
