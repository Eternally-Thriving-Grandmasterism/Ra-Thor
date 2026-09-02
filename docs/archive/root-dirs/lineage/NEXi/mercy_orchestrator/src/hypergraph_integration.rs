// mercy_orchestrator/src/hypergraph_integration.rs — HyperGraphDB with HGQuery for Valence Rules
use jni::objects::{JClass, JObject, JString, JValue};
use jni::sys::{jboolean, jdouble, jint, jlong, jobject};
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
    #[error("Query execution failed")]
    QueryFailed,
}

pub struct HyperGraphDB {
    jvm: JavaVM,
    graph: JObject<'static>, // HGHandle? Actually HyperGraph instance
}

impl HyperGraphDB {
    pub fn new(location: &Path) -> Result<Self, Box<dyn Error>> {
        // JVM init (simplified; in real use Launch JVM or attach to existing)
        // For demo: assume JVM launched externally, use from_raw or proper attach
        let jvm = unsafe { JavaVM::from_raw(std::ptr::null_mut())? }; // REPLACE with proper init!
        let env = jvm.attach_current_thread()?;

        let config_class: JClass = env.find_class("org/hypergraphdb/HGConfiguration")?;
        let config: JObject = env.new_object(config_class, "()V", &[])?;

        let location_str: JString = env.new_string(location.to_str().unwrap_or(""))?;
        env.call_method(&config, "setStoreLocation", "(Ljava/lang/String;)V", &[location_str.into()])?;

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

        // Add as value (String)
        let atom_handle: JObject = env.call_method(
            &self.graph,
            "add",
            "(Ljava/lang/Object;)Lorg/hypergraphdb/HGHandle;",
            &[atom_str.into()],
        )?.l()?;

        // Add valence as linked value (Double)
        let valence_obj: JObject = env.new_object("java/lang/Double", "(D)V", &[JValue::Double(valence)])?;
        let valence_handle: JObject = env.call_method(
            &self.graph,
            "add",
            "(Ljava/lang/Object;)Lorg/hypergraphdb/HGHandle;",
            &[valence_obj.into()],
        )?.l()?;

        // Link them (e.g., as ordered pair or custom link)
        let link_class: JClass = env.find_class("org/hypergraphdb/HGPlainLink")?;
        let link_args = env.new_object_array(2, "org/hypergraphdb/HGHandle", None)?;
        env.set_object_array_element(&link_args, 0, &atom_handle)?;
        env.set_object_array_element(&link_args, 1, &valence_handle)?;
        let link_handle: JObject = env.new_object(
            link_class,
            "([Lorg/hypergraphdb/HGHandle;)V",
            &[link_args.into()],
        )?;
        let link_persist: JObject = env.call_method(&self.graph, "add", "(Ljava/lang/Object;)Lorg/hypergraphdb/HGHandle;", &[link_handle.into()])?.l()?;

        let handle_long: jlong = env.call_method(&link_persist, "getPersistentHandle", "()J", &[])?.j()?;
        Ok(handle_long)
    }

    // NEW: Query high-valence atoms using HGQuery.hg
    pub fn query_high_valence(&self, min_valence: f64) -> Result<Vec<(String, f64)>, HyperError> {
        let env = self.jvm.attach_current_thread()?;

        // Import statics from HGQuery.hg
        let hg_class: JClass = env.find_class("org/hypergraphdb/HGQuery$hg")?;

        // Build condition: atom with valence property >= min
        // Simplified: assume valence stored as linked Double, query links where target is Double >= min, source is String
        // Real: use AtomPartCondition, ComparisonOperator, etc.

        // Example Java equiv: hg.and(hg.targetValue(hg.eq(min_valence)), hg.type(Double.class))
        // But for >= : hg.targetValue(hg.ge(min_valence))

        let ge_method = env.get_static_method_id(hg_class, "ge", "(D)Lorg/hypergraphdb/query/ComparisonCondition;")?;
        let ge_cond: JObject = env.call_static_method_unchecked(
            hg_class,
            ge_method,
            "(D)Lorg/hypergraphdb/query/ComparisonCondition;",
            &[JValue::Double(min_valence)],
        )?.l()?;

        let target_value_method = env.get_static_method_id(hg_class, "targetValue", "(Lorg/hypergraphdb/query/HGQueryCondition;)Lorg/hypergraphdb/query/AtomPartCondition;")?;
        let valence_cond: JObject = env.call_static_method_unchecked(
            hg_class,
            target_value_method,
            "(Lorg/hypergraphdb/query/HGQueryCondition;)Lorg/hypergraphdb/query/AtomPartCondition;",
            &[ge_cond.into()],
        )?.l()?;

        // And with source being String (MettaAtom)
        let string_class: JClass = env.find_class("java/lang/String")?;
        let type_cond: JObject = env.call_static_method(
            hg_class,
            "type",
            "(Ljava/lang/Class;)Lorg/hypergraphdb/query/AtomTypeCondition;",
            &[string_class.into()],
        )?.l()?;

        let and_method = env.get_static_method_id(hg_class, "and", "([Lorg/hypergraphdb/query/HGQueryCondition;)Lorg/hypergraphdb/query/And;")?;
        let cond_array = env.new_object_array(2, "org/hypergraphdb/query/HGQueryCondition", None)?;
        env.set_object_array_element(&cond_array, 0, &valence_cond)?;
        env.set_object_array_element(&cond_array, 1, &type_cond)?;
        let and_cond: JObject = env.call_static_method_unchecked(
            hg_class,
            and_method,
            "([Lorg/hypergraphdb/query/HGQueryCondition;)Lorg/hypergraphdb/query/And;",
            &[cond_array.into()],
        )?.l()?;

        // Execute query
        let make_method = env.get_static_method_id(hg_class, "make", "(Lorg/hypergraphdb/HyperGraph;Lorg/hypergraphdb/query/HGQueryCondition;)Lorg/hypergraphdb/HGQuery;")?;
        let query: JObject = env.call_static_method_unchecked(
            hg_class,
            make_method,
            "(Lorg/hypergraphdb/HyperGraph;Lorg/hypergraphdb/query/HGQueryCondition;)Lorg/hypergraphdb/HGQuery;",
            &[self.graph.into(), and_cond.into()],
        )?.l()?;

        let execute_method = env.get_method_id("org/hypergraphdb/HGQuery", "execute", "()Lorg/hypergraphdb/HGSearchResult;")?;
        let result: JObject = env.call_method_unchecked(&query, execute_method, "()Lorg/hypergraphdb/HGSearchResult;", &[])?.l()?;

        // Iterate results (handles), fetch values
        let mut results = Vec::new();
        let has_next_method = env.get_method_id("org/hypergraphdb/HGSearchResult", "hasNext", "()Z")?;
        let next_method = env.get_method_id("org/hypergraphdb/HGSearchResult", "next", "()Lorg/hypergraphdb/HGHandle;")?;
        let get_method = env.get_method_id("org/hypergraphdb/HyperGraph", "get", "(Lorg/hypergraphdb/HGHandle;)Ljava/lang/Object;")?;

        while env.call_method_unchecked(&result, has_next_method, "()Z", &[])?.z()? {
            let handle: JObject = env.call_method_unchecked(&result, next_method, "()Lorg/hypergraphdb/HGHandle;", &[])?.l()?;
            let value: JObject = env.call_method_unchecked(&self.graph, get_method, "(Lorg/hypergraphdb/HGHandle;)Ljava/lang/Object;", &[handle.into()])?.l()?;

            // Assume value is String atom; valence from linked or prop (simplified parse)
            let atom_str: String = env.get_string(value.into())?.into();
            // Placeholder valence fetch (in real: traverse link to Double)
            let valence = min_valence; // TODO: actual fetch

            results.push((atom_str, valence));
        }

        Ok(results)
    }
}

impl Drop for HyperGraphDB {
    fn drop(&mut self) {
        if let Ok(env) = self.jvm.attach_current_thread() {
            let _ = env.call_method(&self.graph, "close", "()V", &[]);
        }
    }
}
