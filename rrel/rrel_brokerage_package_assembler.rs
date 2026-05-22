/*!
 * RREL Brokerage-Level Package Assembler (v0.9.0 - Skeleton)
 * Assembles multiple offers, Form 801s, compliance data into professional packages.
 * Includes Markdown output + future .docx / Google Docs export skeleton.
 * Privacy-first, example-only, RECO-aligned, mercy-gated.
 */

use crate::rrel_form801_preset::Form801Preset;
use crate::rrel_offer_package::OfferPackage;
use crate::rrel_compliance_helpers::RecordRetentionMetadata;
use crate::rrel_reference_generator::generate_markdown_reference;

/// High-level assembler for brokerage use (multiple transactions or client packages).
pub struct BrokeragePackageAssembler {
    pub brokerage_name: String,
    pub packages: Vec<BrokerageSubPackage>,
}

#[derive(Debug, Clone)]
pub struct BrokerageSubPackage {
    pub client_or_file_ref: String,
    pub form801: Form801Preset,
    pub offer_package: Option<OfferPackage>,
    pub retention: Option<RecordRetentionMetadata>,
    pub compliance_notes: Vec<String>,
}

impl BrokeragePackageAssembler {
    pub fn new(brokerage_name: String) -> Self {
        Self {
            brokerage_name,
            packages: vec![],
        }
    }

    pub fn add_package(&mut self, sub: BrokerageSubPackage) {
        self.packages.push(sub);
    }

    /// Generates a consolidated Markdown package summary for the brokerage.
    pub fn generate_consolidated_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str(&format!("# {} — Consolidated Brokerage Package\n\n", self.brokerage_name));
        md.push_str(&format!("**Generated:** {}\n\n", chrono::Utc::now().format("%Y-%m-%d %H:%M UTC")));
        md.push_str("---\n\n");

        for (i, pkg) in self.packages.iter().enumerate() {
            md.push_str(&format!("## Package {} — {}\n\n", i + 1, pkg.client_or_file_ref));

            let summary = generate_markdown_reference(
                &pkg.form801,
                pkg.offer_package.as_ref(),
                pkg.retention.as_ref(),
                &pkg.compliance_notes,
            );
            md.push_str(&summary);
            md.push_str("\n---\n\n");
        }

        md.push_str("**Note:** Example-only consolidated view. All data fictional. ");
        md.push_str("For professional brokerage use. Future: Direct .docx export and Google Docs sync.\n");

        md
    }
}

// === FUTURE EXTENSIONS (SKELETON) ===
// TODO: Integrate with `docx` crate or pandoc for real .docx file generation.
// TODO: Add Google Docs API export hook (via service account or OAuth).
// TODO: Add PDF generation via printpdf or external renderer.
// TODO: Add PATSAGi Council batch alert generation for the full package.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rrel_form801_preset::Form801Preset;

    #[test]
    fn test_brokerage_assembler_basic() {
        let mut assembler = BrokeragePackageAssembler::new("Example Realty Inc.".to_string());
        let preset = Form801Preset::new_standard(
            "999 Test Blvd".to_string(),
            vec!["Test Buyer".to_string()],
            "2026-07-01".to_string(),
        );
        assembler.add_package(BrokerageSubPackage {
            client_or_file_ref: "File-2026-042".to_string(),
            form801: preset,
            offer_package: None,
            retention: None,
            compliance_notes: vec!["Standard track - no conflicts".to_string()],
        });
        let md = assembler.generate_consolidated_markdown();
        assert!(md.contains("Consolidated Brokerage Package"));
    }
}
