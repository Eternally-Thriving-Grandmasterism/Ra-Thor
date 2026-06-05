use spacetime_reducer_bridge::SpacetimeReducerBridge;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let bridge = SpacetimeReducerBridge::new();
    let result = bridge.execute_reducer("powrush_terrain_edit", vec![1, 2, 3]).await?;
    println!("Reducer result: {}", result);
    Ok(())
}
