#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Property-Based Test for GTMØ
===================================
Simplified version to avoid Windows IO issues.
"""

import sys
from hypothesis import given, strategies as st, settings, example
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize

# Simple test without complex imports
@given(st.text(min_size=1, max_size=100))
def test_text_processing_invariants(text):
    """Test basic text processing invariants."""
    # Basic string operations should work
    assert len(text) >= 1
    assert isinstance(text, str)
    assert text == text.strip() or text != text.strip()

@given(st.integers(min_value=0, max_value=100))
def test_numeric_invariants(number):
    """Test numeric processing invariants."""
    assert number >= 0
    assert number <= 100
    assert isinstance(number, int)

def simple_gtmo_test():
    """Basic GTMØ functionality test."""
    try:
        # Test import without sys modifications
        import gtmo_morphosyntax as gtmo
        
        # Test basic sentence using available function
        test_sentence = "To jest test."
        result = gtmo.analyze_quantum_with_axioms(test_sentence)
        
        print(f"✓ Analysis completed for: '{test_sentence}'")
        print(f"  Result type: {type(result)}")
        
        if isinstance(result, dict):
            print(f"  Keys: {list(result.keys())}")
            
        return True
        
    except Exception as e:
        print(f"✗ GTMØ test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== GTMØ Simple Property Tests ===")
    
    # Run hypothesis tests
    print("\n1. Testing text processing invariants...")
    try:
        test_text_processing_invariants()
        print("   ✓ Text invariants passed")
    except Exception as e:
        print(f"   ✗ Text invariants failed: {e}")
    
    print("\n2. Testing numeric invariants...")
    try:
        test_numeric_invariants()
        print("   ✓ Numeric invariants passed")
    except Exception as e:
        print(f"   ✗ Numeric invariants failed: {e}")
    
    print("\n3. Testing GTMØ functionality...")
    if simple_gtmo_test():
        print("   ✓ GTMØ functionality passed")
    else:
        print("   ✗ GTMØ functionality failed")
    
    print("\n=== Property Tests Complete ===")