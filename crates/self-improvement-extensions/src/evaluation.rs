fn extract_json_from_response(response: &str) -> String {
    let trimmed = response.trim();

    // Try ```json ... ``` or ``` ... ```
    let re_code = Regex::new(r"```(?:json)?\s*([\s\S]*?)\s*```").unwrap();
    if let Some(caps) = re_code.captures(trimmed) {
        if let Some(m) = caps.get(1) {
            let candidate = m.as_str().trim();
            if candidate.starts_with('{') && candidate.ends_with('}') {
                return candidate.to_string();
            }
        }
    }

    // Try to find the first { ... } block
    let re_json = Regex::new(r"(\{[\s\S]*\})").unwrap();
    if let Some(caps) = re_json.captures(trimmed) {
        if let Some(m) = caps.get(1) {
            return m.as_str().to_string();
        }
    }

    trimmed.to_string()
}