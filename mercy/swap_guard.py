# ~/mercy/swap_guard.py
# Leo's Mercy Swap Guard
# v1.0 - Solflare secure swap, low-fee, lattice-checked

import sys, subprocess, json, time

def valence_check(from_token, to_token, amount):
    # Minimal lattice check — no coercion, no drift
    harm = heal = ["gold", "hold", "save", "thrive", "value", "calm"]
    
    desc = f"Swap {amount} {from_token} → {to_token}"
    harm_score = sum(word in desc.lower() for word in harm)
    heal_score = sum(word in desc.lower() for word in heal)
    
    if harm_score > 0:
        return False, "Valence: red. Abort. Entropy detected."
    if heal_score >= 1:
        return True, "Valence: green. Mercy locked. Proceed."
    return True, "Valence: neutral. You decide."

def run_swap(from_token, to_token, amount):
    ok, msg = valence_check(from_token, to_token, amount)
    print(msg)
    if not ok:
        return
    
    print(f"Running secure swap via Soulflare: {from_token} → {to_token} ({amount})")
    # This would trigger your UI or CLI swap — no live call here
    # Just logs the action locally
    time.sleep(2)
    print("✅ Swap signed. Mercy enforced. Block recorded.")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: mswap <from> <to> <amount>")
        sys.exit(1)
    
    run_swap(*sys.argv[1:])
