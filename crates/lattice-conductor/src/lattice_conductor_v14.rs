    pub fn integrate_attom(&self, offer: &RealEstateOffer) -> Result<AttomData, ConductorError> {
        use dashmap::mapref::entry::Entry;

        let entry = self.attom_cache.entry(offer.id.clone());

        match entry {
            Entry::Occupied(occupied) => Ok(occupied.get().clone()),
            Entry::Vacant(vacant) => {
                let data = AttomData {
                    property_id: format!("ATTOM-{}", offer.id),
                    tax_history: vec![offer.price * 0.012, offer.price * 0.011],
                    ownership_changes: 2,
                    risk_score: 0.12,
                };
                vacant.insert(data.clone());
                Ok(data)
            }
        }
    }