/// WCAG AA Accessibility Scorer
///
/// Provides a practical heuristic score (0–100) based on common WCAG 2.1 AA success criteria.
/// This is not a full automated accessibility audit, but a useful signal for generated content.

pub struct WcagAaScore {
    pub score: f32,           // 0.0 – 100.0
    pub issues: Vec<String>,
    pub grade: String,        // A, B, C, D, F
}

/// Calculates a WCAG AA accessibility score for the given HTML.
pub fn calculate_wcag_aa_score(html: &str) -> WcagAaScore {
    let mut issues = Vec::new();
    let mut deductions: f32 = 0.0;

    // === Critical Checks (higher weight) ===

    if html.contains("<img") && !html.contains("alt=") {
        issues.push("Images missing alt text".to_string());
        deductions += 15.0;
    }

    if html.contains("<input") && !html.contains("<label") {
        issues.push("Form inputs missing associated labels".to_string());
        deductions += 12.0;
    }

    if !html.contains("<h1") {
        issues.push("Missing primary heading (<h1>)".to_string());
        deductions += 8.0;
    }

    // === Important Checks ===

    if html.contains("onclick=") && !html.contains("tabindex=") {
        issues.push("Interactive elements may lack keyboard support".to_string());
        deductions += 8.0;
    }

    if !html.contains("<main") && !html.contains("role=\"main\"") {
        issues.push("Missing main landmark".to_string());
        deductions += 7.0;
    }

    // === Moderate Checks ===

    if html.contains("<button") && !html.contains(">") {
        issues.push("Buttons may lack accessible names".to_string());
        deductions += 6.0;
    }

    // Calculate final score
    let mut score = 100.0 - deductions;
    if score < 0.0 { score = 0.0; }

    let grade = match score {
        s if s >= 90.0 => "A",
        s if s >= 80.0 => "B",
        s if s >= 70.0 => "C",
        s if s >= 60.0 => "D",
        _ => "F",
    };

    WcagAaScore {
        score,
        issues,
        grade: grade.to_string(),
    }
}
