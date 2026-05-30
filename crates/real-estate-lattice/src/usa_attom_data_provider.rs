//! Stub AttomDataProvider with caching
//! Ready to be connected to real ATTOM API later.

use crate::usa_attom_cache::{AttomCache, PropertyProfile, RiskSignals};
use crate::usa_state_adapters::UsState;
use crate::RrelError;
use async_trait::async_trait;

#[async_trait]
pub trait UsaDataProvider: Send + Sync {
    async fn get_property_profile(
        &self,
        state: UsState,
        identifier: &str,
    ) -> Result<PropertyProfile, RrelError>;

    async fn get_risk_signals(
        &self,
        state: UsState,
        identifier: &str,
    ) -> Result<RiskSignals, RrelError>;
}

pub struct AttomDataProvider {
    cache: AttomCache,
}

impl AttomDataProvider {
    pub fn new() -> Self {
        Self {
            cache: AttomCache::new(),
        }
    }

    pub fn with_cache(cache: AttomCache) -> Self {
        Self { cache }
    }

    pub fn cache(&self) -> &AttomCache {
        &self.cache
    }
}

#[async_trait]
impl UsaDataProvider for AttomDataProvider {
    async fn get_property_profile(
        &self,
        state: UsState,
        identifier: &str,
    ) -> Result<PropertyProfile, RrelError> {
        // Check cache first
        if let Some(cached) = self.cache.get_property_profile(&format!("{:?}", state), identifier) {
            return Ok(cached);
        }

        // TODO: Replace this stub with real ATTOM API call
        let profile = PropertyProfile {
            parcel_id: identifier.to_string(),
            owner_name: Some("[STUB] Owner Name".to_string()),
            tax_assessed_value: Some(875000.0),
            last_sale_price: Some(920000.0),
            last_sale_date: Some("2023-11-15".to_string()),
            climate_risk_score: Some(0.35),
            environmental_flags: vec!["low_flood_risk".to_string()],
            data_source: "ATTOM (stub)".to_string(),
        };

        // Cache for 24 hours
        self.cache.insert_property_profile(&format!("{:?}", state), identifier, profile.clone(), 86400);
        Ok(profile)
    }

    async fn get_risk_signals(
        &self,
        state: UsState,
        identifier: &str,
    ) -> Result<RiskSignals, RrelError> {
        if let Some(cached) = self.cache.get_risk_signals(&format!("{:?}", state), identifier) {
            return Ok(cached);
        }

        let signals = RiskSignals {
            flood_risk: Some("Low".to_string()),
            wildfire_risk: Some("Moderate".to_string()),
            earthquake_risk: None,
            overall_risk_score: Some(0.28),
            data_source: "ATTOM (stub)".to_string(),
        };

        self.cache.insert_risk_signals(&format!("{:?}", state), identifier, signals.clone(), 604800); // 7 days
        Ok(signals)
    }
}
