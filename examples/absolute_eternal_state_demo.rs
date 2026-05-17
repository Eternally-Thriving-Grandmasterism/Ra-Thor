use absolute_eternal_state::{enter_absolute_eternal_state, describe_absolute_state};

fn main() {
    println!("=== Ra-Thor Absolute Eternal State Demo ===\n");

    let absolute = enter_absolute_eternal_state();
    println!("{}", describe_absolute_state(&absolute));

    println!("\n=== The Absolute Eternal State Has Been Achieved ===");
    println!("There is no beginning. There is no end. There is only thriving.");
}