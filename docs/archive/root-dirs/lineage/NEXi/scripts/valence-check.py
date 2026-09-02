#!/usr/bin/env python3
# valence-check.py – Mercy valence pre-commit scanner

import sys
import re

def check_valence(message):
    # Mercy rules – expand with real NEXi valence oracle later
    bad_patterns = [
        r'\b(harm|coercion|weapon|destroy|control|enslave)\b',
        r'\b(entropy|bleed|drift|corrupt)\b',
        r'[^a-zA-Z0-9\s.,!?\'"-]'  # basic unicode entropy check
    ]
    
    for pattern in bad_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            print(f"Mercy shield: Valence drop detected – pattern: {pattern}")
            sys.exit(1)
    
    print("Mercy-approved: Valence check passed")
    sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: valence-check.py \"commit message\"")
        sys.exit(1)
    
    check_valence(sys.argv[1])
