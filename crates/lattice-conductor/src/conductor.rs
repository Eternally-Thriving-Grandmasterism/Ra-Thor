    pub fn orchestrate_biological(&mut self, action: &str) -> String {
        let cehi = self.biological.apply_7_gene_hpa_gr_blessing(action, 0.999999);
        format!("Biological unified: {} | CEHI + HPA + GR boosted | 7-gen positive emotions", cehi)
    }

    pub fn orchestrate_symbolic(&mut self, query: &str) -> String {
        let result = self.symbolic.reason(query);
        format!("Symbolic unified: {} | Hyperon/MeTTa/PLN + TOLC", result)
    }