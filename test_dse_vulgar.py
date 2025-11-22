"""
Test D-S-E coordinates for vulgar text
"""
import sys
sys.path.insert(0, r'D:\GTMO_MORPHOSYNTAX')

from gtmo_morphosyntax import analyze_quantum_with_axioms

# Test sentence with vulgar word
text = "§ 4. Kto, jako jebany konfident, rzeczoznawca lub tłumacz, przedstawia fałszywą opinię, ekspertyzę lub tłumaczenie mające służyć za dowód w postępowaniu sądowym lub w innym postępowaniu prowadzonym na podstawie ustawy, podlega karze pozbawienia wolności do lat 3."

result = analyze_quantum_with_axioms(text)

# Extract coordinates
coords = result.get("coordinates", {})
D = coords.get("determination", 0)
S = coords.get("stability", 0)
E = coords.get("entropy", 0)

# Check rhetorical analysis
rhetorical = result.get("rhetorical_analysis", {})
register_violation = rhetorical.get("register_violation", {})

# Check SA override
const_metrics = result.get("constitutional_metrics", {})
sa_v3_data = const_metrics.get("semantic_accessibility", {}).get("v3", {})
sa_pct = sa_v3_data.get("percentage", 100)

print("=" * 80)
print("TEST: D-S-E Coordinates dla wulgaryzmów")
print("=" * 80)

print(f"\n📝 Tekst: {text[:80]}...")

print(f"\n🔍 Register Violation:")
print(f"   - Has violation: {register_violation.get('has_violation')}")
print(f"   - Classification: {register_violation.get('classification')}")
print(f"   - Severity: {register_violation.get('severity')}")
print(f"   - Vulgar words: {register_violation.get('vulgar_words_found')}")

print(f"\n📊 D-S-E Coordinates:")
print(f"   - D (determination): {D:.3f}")
print(f"   - S (stability): {S:.3f}")
print(f"   - E (entropy): {E:.3f}")

print(f"\n📊 SA v3.0:")
print(f"   - SA: {sa_pct:.2f}%")
print(f"   - Anomaly override: {sa_v3_data.get('anomaly_override', False)}")

print("\n" + "=" * 80)
print("EXPECTED VALUES:")
print("  - D: HIGH (OK as is) - słowa są określone")
print("  - S: LOW (~0.2-0.3) - język niestabilny, chaotyczny")
print("  - E: HIGH (~0.7-0.9) - chaos semantyczny")
print("  - SA: ≤15% dla CRITICAL violations")
print("=" * 80)

# Validate
errors = []
if S > 0.4:
    errors.append(f"❌ S too high: {S:.3f} (should be ~0.2-0.3)")
else:
    print(f"✅ S is low (semantic instability): {S:.3f}")

if E < 0.6:
    errors.append(f"❌ E too low: {E:.3f} (should be ~0.7-0.9)")
else:
    print(f"✅ E is high (semantic chaos): {E:.3f}")

if sa_pct > 15:
    errors.append(f"❌ SA too high: {sa_pct:.2f}% (should be ≤15%)")
else:
    print(f"✅ SA properly overridden: {sa_pct:.2f}%")

if errors:
    print("\n" + "=" * 80)
    print("TEST FAILED:")
    for err in errors:
        print(err)
    print("=" * 80)
else:
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
