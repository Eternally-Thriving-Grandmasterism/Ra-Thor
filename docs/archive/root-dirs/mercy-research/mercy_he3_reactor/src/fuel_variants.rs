// mercy_he3_reactor/src/fuel_variants.rs — Mercy-Gated Fuel Selector
#[derive(Debug, Clone, PartialEq)]
pub enum AneutronicFuel {
    DHe3,
    pB11,
    He3He3,
}

#[derive(Debug, Clone)]
pub struct FuelSelector {
    pub valence: f64,
}

impl FuelSelector {
    pub fn new() -> Self {
        FuelSelector { valence: 1.0 }
    }

    pub fn select(&self, available_lunar_he3: bool, available_boron: bool) -> AneutronicFuel {
        if self.valence < 0.9999999 {
            println!("Mercy shield: Fuel selection paused — valence {:.7}", self.valence);
            return AneutronicFuel::DHe3; // fallback
        }

        if available_lunar_he3 {
            println!("Mercy-approved: D-³He selected — lunar sourcing viable");
            AneutronicFuel::DHe3
        } else if available_boron {
            println!("Mercy-approved: p-¹¹B selected — Earth boron abundant");
            AneutronicFuel::pB11
        } else {
            println!("Mercy-approved: ³He-³He fallback — high-temp pure aneutronic");
            AneutronicFuel::He3He3
        }
    }
}

pub fn simulate_fuel_selection(lunar_he3: bool, boron: bool) {
    let selector = FuelSelector::new();
    let fuel = selector.select(lunar_he3, boron);
    println!("Selected fuel: {:?}", fuel);
}
