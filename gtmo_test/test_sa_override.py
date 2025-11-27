"""
Test SA Override dla IRRATIONAL_ANOMALY
"""
import sys
sys.path.insert(0, r'D:\GTMO_MORPHOSYNTAX')

from gtmo_morphosyntax import GTMOAnalyzer

# Test sentence with vulgar word
text = "§ 4. Kto, jako jebany konfident, rzeczoznawca lub tłumacz, przedstawia fałszywą opinię, ekspertyzę lub tłumaczenie mające służyć za dowód w postępowaniu sądowym lub w innym postępowaniu prowadzonym na podstawie ustawy, podlega karze pozbawienia wolności do lat 3."

analyzer = GTMOAnalyzer()
result = analyzer.analyze(text)

# Check results
rhetorical = result.get("rhetorical_analysis", {})
register_violation = rhetorical.get("register_violation", {})
const_metrics = result.get("constitutional_metrics", {})
sa_v3_data = const_metrics.get("semantic_accessibility", {}).get("v3", {})

print("=" * 80)
print("TEST: SA Override dla wulgaryzmów")
print("=" * 80)

print(f"\n📝 Tekst: {text[:100]}...")

print(f"\n🔍 Register Violation:")
print(f"   - Has violation: {register_violation.get('has_violation')}")
print(f"   - Classification: {register_violation.get('classification')}")
print(f"   - Severity: {register_violation.get('severity')}")
print(f"   - Vulgar words: {register_violation.get('vulgar_words_found')}")

print(f"\n📊 SA v3.0:")
print(f"   - Current SA: {sa_v3_data.get('percentage')}%")
print(f"   - Anomaly override: {sa_v3_data.get('anomaly_override', False)}")
if sa_v3_data.get('anomaly_override'):
    print(f"   - Original SA: {sa_v3_data.get('original_value') * 100:.2f}%")
    print(f"   - Penalty reason: {sa_v3_data.get('penalty_reason')}")

print(f"\n🚨 Critical Block: {result.get('critical_block', False)}")
print(f"   - Reason: {result.get('critical_reason', 'N/A')}")

print("\n" + "=" * 80)
if sa_v3_data.get('percentage', 100) <= 15:
    print("✅ TEST PASSED: SA properly overridden to <= 15%")
else:
    print(f"❌ TEST FAILED: SA = {sa_v3_data.get('percentage')}% (should be <= 15%)")
print("=" * 80)
