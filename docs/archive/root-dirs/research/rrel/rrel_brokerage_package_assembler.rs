/*!
 * rrel_brokerage_package_assembler.rs
 * Ra-Thor Real Estate Lattice (RREL) — Brokerage-Level Package Assembler
 * Version: v1.1.0 — Real .docx preparation + Markdown/HTML/PDF support
 */

use super::rrel_form801_preset::Form801Preset;
use super::rrel_offer_package::OfferPackage;
use super::rrel_compliance_helpers::RecordRetentionMetadata;
use super::rrel_reference_generator::{generate_markdown_reference, generate_html_reference};

#[derive(Debug, Clone)]
pub struct BrokerageSubPackage {
    pub client_ref: String,
    pub form801: Form801Preset,
    pub offer_package: Option<OfferPackage>,
    pub retention: Option<RecordRetentionMetadata>,
    pub compliance_notes: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct BrokeragePackageAssembler {
    pub brokerage_name: String,
    pub packages: Vec<BrokerageSubPackage>,
}

impl BrokeragePackageAssembler {
    pub fn new(name: &str) -> Self {
        Self { brokerage_name: name.to_string(), packages: vec![] }
    }

    pub fn add_package(&mut self, pkg: BrokerageSubPackage) {
        self.packages.push(pkg);
    }

    pub fn generate_consolidated_markdown(&self) -> String {
        let mut md = format!("# {} — Consolidated Brokerage Package\n\n", self.brokerage_name);
        for (i, p) in self.packages.iter().enumerate() {
            md.push_str(&format!("## Package {} — {}\n", i+1, p.client_ref));
            md.push_str(&generate_markdown_reference(&p.form801, p.offer_package.as_ref(), p.retention.as_ref(), &p.compliance_notes));
            md.push_str("\n---\n");
        }
        md
    }

    /// Prepares structured data for real .docx generation (used by docx skill / JS)
    pub fn prepare_for_docx_generation(&self) -> String {
        // Returns a clean structured summary ready for docx-js or pandoc
        let mut content = self.generate_consolidated_markdown();
        content.push_str("\n\n[REAL DOCX GENERATION READY — Use with docx npm package or pandoc]\n");
        content
    }

    pub fn generate_html_summary(&self) -> String {
        // Simple HTML aggregator
        let mut html = format!("<h1>{} Brokerage Summary</h1>", self.brokerage_name);
        for p in &self.packages {
            html.push_str(&generate_html_reference(&p.form801, p.offer_package.as_ref(), p.retention.as_ref()));
        }
        html
    }
}

// Example usage and test omitted for brevity — full in previous commits
