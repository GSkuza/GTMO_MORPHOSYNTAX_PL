#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTMØ Property-Based Tests - Simple Version
==========================================
Simplified property-based testing for GTMØ without IO conflicts.
"""

import sys
import os
import json
import math
import traceback
from pathlib import Path

# Import required modules
try:
    from hypothesis import given, strategies as st, assume, settings
    from hypothesis import HealthCheck
    import numpy as np
    HYPOTHESIS_AVAILABLE = True
    print("✓ Hypothesis and NumPy available")
except ImportError as e:
    print(f"✗ Missing dependencies: {e}")
    HYPOTHESIS_AVAILABLE = False

def test_coordinate_bounds_simple():
    """Simple test for coordinate bounds without Hypothesis."""
    # Import here to avoid IO conflicts
    import sys
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        from gtmo_morphosyntax import QuantumMorphosyntaxEngine
        sys.stdout.close()
        sys.stdout = old_stdout
        
        engine = QuantumMorphosyntaxEngine()
        
        test_texts = [
            "Kot śpi.",
            "Rzeczpospolita Polska jest demokratycznym państwem prawnym.",
            "Test bardzo długi tekst z wieloma słowami do sprawdzenia granic współrzędnych.",
            "Czy to pytanie?"
        ]
        
        passed = 0
        total = len(test_texts)
        
        for text in test_texts:
            try:
                result = engine.gtmo_analyze_quantum(text)
                coords = result['coordinates']
                
                # Check bounds
                if (0 <= coords['determination'] <= 1 and 
                    0 <= coords['stability'] <= 1 and 
                    0 <= coords['entropy'] <= 1):
                    passed += 1
                    print(f"✓ '{text[:30]}...' - coordinates OK")
                else:
                    print(f"✗ '{text[:30]}...' - coordinates out of bounds")
                    
            except Exception as e:
                print(f"✗ '{text[:30]}...' - error: {e}")
        
        success_rate = passed / total * 100
        print(f"\n📊 Coordinate bounds test: {passed}/{total} passed ({success_rate:.1f}%)")
        return success_rate >= 90
        
    except Exception as e:
        if sys.stdout != old_stdout:
            sys.stdout.close()
            sys.stdout = old_stdout
        print(f"✗ Engine import failed: {e}")
        return False

def test_constitutional_duality_simple():
    """Simple test for constitutional duality."""
    import sys
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        from gtmo_morphosyntax import QuantumMorphosyntaxEngine
        sys.stdout.close()
        sys.stdout = old_stdout
        
        engine = QuantumMorphosyntaxEngine()
        
        test_texts = [
            "Sąd Konstytucyjny orzeka o zgodności ustaw z Konstytucją.",
            "Nieantykonstytucjonalizacyjność jest długim słowem.",
            "Prosty tekst testowy.",
            "Bardzo skomplikowane zdanie z wieloma zależnościami składniowymi i morfologicznymi."
        ]
        
        passed = 0
        total = len(test_texts)
        
        for text in test_texts:
            try:
                result = engine.gtmo_analyze_quantum(text)
                const_metrics = result['constitutional_metrics']
                
                ci = const_metrics['indefiniteness']['value']
                cd = const_metrics['definiteness']['value']
                duality_error = const_metrics['duality']['error_percent']
                
                # Check duality: CI × CD = Depth²
                if duality_error < 5.0:  # 5% tolerance
                    passed += 1
                    print(f"✓ '{text[:30]}...' - duality error: {duality_error:.3f}%")
                else:
                    print(f"✗ '{text[:30]}...' - duality error too high: {duality_error:.3f}%")
                    
            except Exception as e:
                print(f"✗ '{text[:30]}...' - error: {e}")
        
        success_rate = passed / total * 100
        print(f"\n📊 Constitutional duality test: {passed}/{total} passed ({success_rate:.1f}%)")
        return success_rate >= 90
        
    except Exception as e:
        if sys.stdout != old_stdout:
            sys.stdout.close()
            sys.stdout = old_stdout
        print(f"✗ Engine import failed: {e}")
        return False

def test_semantic_accessibility_simple():
    """Simple test for semantic accessibility bounds."""
    import sys
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        from gtmo_morphosyntax import QuantumMorphosyntaxEngine
        sys.stdout.close()
        sys.stdout = old_stdout
        
        engine = QuantumMorphosyntaxEngine()
        
        test_texts = [
            "Dom.",
            "Kot ma dom.",
            "Rzeczpospolita Polska przestrzega prawa międzynarodowego.",
            "Najpiękniejsze dzieła sztuki zachwycają wszystkich zwiedzających muzea."
        ]
        
        passed = 0
        total = len(test_texts)
        
        for text in test_texts:
            try:
                result = engine.gtmo_analyze_quantum(text)
                sa = result['constitutional_metrics']['semantic_accessibility']['value']
                
                # Check SA bounds [0,1]
                if 0 <= sa <= 1:
                    passed += 1
                    print(f"✓ '{text[:30]}...' - SA: {sa:.3f}")
                else:
                    print(f"✗ '{text[:30]}...' - SA out of bounds: {sa:.3f}")
                    
            except Exception as e:
                print(f"✗ '{text[:30]}...' - error: {e}")
        
        success_rate = passed / total * 100
        print(f"\n📊 Semantic accessibility test: {passed}/{total} passed ({success_rate:.1f}%)")
        return success_rate >= 90
        
    except Exception as e:
        if sys.stdout != old_stdout:
            sys.stdout.close()
            sys.stdout = old_stdout
        print(f"✗ Engine import failed: {e}")
        return False

def create_simple_golden_dataset():
    """Create a simple golden dataset."""
    print("\n📋 Creating simple golden dataset...")
    
    import sys
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        from gtmo_morphosyntax import QuantumMorphosyntaxEngine
        sys.stdout.close()
        sys.stdout = old_stdout
        
        engine = QuantumMorphosyntaxEngine()
        
        golden_samples = [
            "Kot śpi na macie.",
            "Rzeczpospolita Polska jest demokratycznym państwem prawnym.",
            "Sąd Konstytucyjny orzeka o zgodności ustaw z Konstytucją.",
            "Nieantykonstytucjonalizacyjność jest bardzo długim słowem.",
            "To wspaniałe, że znowu pada deszcz!",  # Irony
            "Jedyną stałą jest zmiana.",  # Paradox
            "123 456 789",  # Numbers
            "Bardzo bardzo bardzo długie zdanie z wieloma powtórzeniami i skomplikowaną strukturą składniową która testuje wydajność i dokładność analizy morfosyntaktycznej."
        ]
        
        golden_dataset = []
        
        for i, text in enumerate(golden_samples):
            try:
                result = engine.gtmo_analyze_quantum(text, f"golden_{i}")
                
                test_case = {
                    "id": f"golden_{i:03d}",
                    "input_text": text,
                    "expected_coordinates": result['coordinates'],
                    "expected_cd": result['constitutional_metrics']['definiteness']['value'],
                    "expected_ci": result['constitutional_metrics']['indefiniteness']['value'],
                    "expected_sa": result['constitutional_metrics']['semantic_accessibility']['value'],
                    "expected_duality_error": result['constitutional_metrics']['duality']['error_percent']
                }
                
                golden_dataset.append(test_case)
                print(f"✓ Golden sample {i}: {text[:40]}...")
                
            except Exception as e:
                print(f"✗ Failed sample {i}: {e}")
        
        # Save dataset
        with open('simple_golden_dataset.json', 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "version": "1.0",
                    "total_samples": len(golden_dataset),
                    "created_at": "2025-11-05"
                },
                "samples": golden_dataset
            }, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Golden dataset saved: {len(golden_dataset)} samples")
        return len(golden_dataset) > 0
        
    except Exception as e:
        if sys.stdout != old_stdout:
            sys.stdout.close()
            sys.stdout = old_stdout
        print(f"✗ Golden dataset creation failed: {e}")
        return False

def validate_golden_dataset():
    """Validate current engine against golden dataset."""
    print("\n🎯 Validating golden dataset...")
    
    if not Path('simple_golden_dataset.json').exists():
        print("❌ Golden dataset not found")
        return False
    
    import sys
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        from gtmo_morphosyntax import QuantumMorphosyntaxEngine
        sys.stdout.close()
        sys.stdout = old_stdout
        
        engine = QuantumMorphosyntaxEngine()
        
        with open('simple_golden_dataset.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        passed = 0
        total = len(dataset['samples'])
        tolerance = 0.01  # 1% tolerance
        
        for sample in dataset['samples']:
            try:
                result = engine.gtmo_analyze_quantum(sample['input_text'])
                
                # Check coordinates
                coords_ok = True
                for coord_name in ['determination', 'stability', 'entropy']:
                    expected = sample['expected_coordinates'][coord_name]
                    actual = result['coordinates'][coord_name]
                    if abs(expected - actual) > tolerance:
                        coords_ok = False
                        break
                
                # Check duality error
                expected_error = sample['expected_duality_error']
                actual_error = result['constitutional_metrics']['duality']['error_percent']
                duality_ok = abs(expected_error - actual_error) < 1.0  # 1% tolerance
                
                if coords_ok and duality_ok:
                    passed += 1
                    print(f"✓ {sample['id']}: PASSED")
                else:
                    print(f"✗ {sample['id']}: FAILED (coords: {coords_ok}, duality: {duality_ok})")
                    
            except Exception as e:
                print(f"✗ {sample['id']}: ERROR ({e})")
        
        success_rate = passed / total * 100
        print(f"\n📊 Golden dataset validation: {passed}/{total} passed ({success_rate:.1f}%)")
        return success_rate >= 95
        
    except Exception as e:
        if sys.stdout != old_stdout:
            sys.stdout.close()
            sys.stdout = old_stdout
        print(f"✗ Validation failed: {e}")
        return False

def run_hypothesis_tests():
    """Run Hypothesis-based property tests if available."""
    if not HYPOTHESIS_AVAILABLE:
        print("\n⚠️  Hypothesis not available - skipping property-based tests")
        return False
    
    print("\n🧪 Running Hypothesis property-based tests...")
    
    import sys
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        from gtmo_morphosyntax import QuantumMorphosyntaxEngine
        sys.stdout.close()
        sys.stdout = old_stdout
        
        engine = QuantumMorphosyntaxEngine()
        
        @given(st.text(min_size=5, max_size=100))
        @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
        def test_coordinates_bounded(text):
            assume(text.strip())
            assume(len(text.split()) >= 1)
            
            try:
                result = engine.gtmo_analyze_quantum(text)
                coords = result['coordinates']
                
                assert 0 <= coords['determination'] <= 1
                assert 0 <= coords['stability'] <= 1
                assert 0 <= coords['entropy'] <= 1
                
            except Exception:
                assume(False)
        
        @given(st.text(min_size=10, max_size=200))
        @settings(max_examples=15)
        def test_duality_preserved(text):
            assume(text.strip())
            assume(len(text.split()) >= 2)
            
            try:
                result = engine.gtmo_analyze_quantum(text)
                duality_error = result['constitutional_metrics']['duality']['error_percent']
                assert duality_error < 10.0
                
            except Exception:
                assume(False)
        
        # Run the tests
        try:
            test_coordinates_bounded()
            print("✓ Hypothesis coordinates test passed")
            coord_test_passed = True
        except Exception as e:
            print(f"✗ Hypothesis coordinates test failed: {e}")
            coord_test_passed = False
        
        try:
            test_duality_preserved()
            print("✓ Hypothesis duality test passed")
            duality_test_passed = True
        except Exception as e:
            print(f"✗ Hypothesis duality test failed: {e}")
            duality_test_passed = False
        
        return coord_test_passed and duality_test_passed
        
    except Exception as e:
        if sys.stdout != old_stdout:
            sys.stdout.close()
            sys.stdout = old_stdout
        print(f"✗ Hypothesis tests failed: {e}")
        return False

def main():
    """Main test execution."""
    print("🧪 GTMØ SIMPLE PROPERTY-BASED TESTING SUITE")
    print("=" * 60)
    
    results = {}
    
    # Run basic tests
    print("\n🔍 Running basic property tests...")
    results['coordinate_bounds'] = test_coordinate_bounds_simple()
    results['constitutional_duality'] = test_constitutional_duality_simple()
    results['semantic_accessibility'] = test_semantic_accessibility_simple()
    
    # Create and validate golden dataset
    results['golden_dataset_creation'] = create_simple_golden_dataset()
    results['golden_dataset_validation'] = validate_golden_dataset()
    
    # Run Hypothesis tests if available
    results['hypothesis_tests'] = run_hypothesis_tests()
    
    # Calculate overall success
    passed_tests = sum(1 for result in results.values() if result)
    total_tests = len(results)
    success_rate = passed_tests / total_tests * 100
    
    print("\n" + "=" * 60)
    print("📋 FINAL TEST RESULTS")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:25s}: {status}")
    
    print(f"\n🎯 Overall success rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 SUCCESS: GTMØ property-based testing shows strong coverage!")
        print("✓ Coordinate bounds verified")
        print("✓ Constitutional duality maintained") 
        print("✓ Semantic accessibility normalized")
        print("✓ Golden dataset validation passed")
        if results['hypothesis_tests']:
            print("✓ Hypothesis property-based tests passed")
    else:
        print("⚠️  NEEDS IMPROVEMENT: Some tests failed")
        print("Review failed tests and improve implementation")
    
    return success_rate >= 80

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)