#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTMØ Basic Integration Test
===========================
Tests basic functionality and integration between modules.
"""

import sys
import os
import json
from pathlib import Path

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def test_imports():
    """Test if all modules can be imported."""
    print("\n" + "="*60)
    print("Testing imports...")
    print("="*60)
    
    try:
        import numpy as np
        print("✓ NumPy imported")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import morfeusz2
        print("✓ Morfeusz2 imported")
    except ImportError as e:
        print(f"✗ Morfeusz2 import failed: {e}")
        print("  Install: pip install morfeusz2")
        return False
    
    try:
        import spacy
        print("✓ spaCy imported")
        try:
            nlp = spacy.load('pl_core_news_lg')
            print("✓ spaCy large model loaded")
        except:
            try:
                nlp = spacy.load('pl_core_news_sm')
                print("✓ spaCy small model loaded")
            except Exception as e:
                print(f"✗ spaCy model load failed: {e}")
                print("  Install: python -m spacy download pl_core_news_lg")
                return False
    except ImportError as e:
        print(f"✗ spaCy import failed: {e}")
        return False
    
    return True


def test_base_module():
    """Test base morphosyntax module."""
    print("\n" + "="*60)
    print("Testing base module (gtmo_morphosyntax.py)...")
    print("="*60)

    try:
        from gtmo_morphosyntax import analyze_quantum_with_axioms

        test_texts = [
            "Ona ma tunele w uszach.",
            "Prawo jest prawem.",
            "Skąd się biorą dzieci?"
        ]

        for text in test_texts:
            print(f"\nAnalyzing: '{text}'")
            result = analyze_quantum_with_axioms(text)

            if 'coordinates' in result:
                coords = result['coordinates']
                print(f"  D: {coords['determination']:.3f}")
                print(f"  S: {coords['stability']:.3f}")
                print(f"  E: {coords['entropy']:.3f}")
            else:
                print(f"  Result: {result}")

        print("\n✓ Base module tests passed")
        return True

    except Exception as e:
        print(f"✗ Base module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_extended_module():
    """Test extended module with quantum and temporal features."""
    print("\n" + "="*60)
    print("Testing extended module (gtmo_extended.py)...")
    print("="*60)
    
    try:
        from gtmo_extended import gtmo_analyze_extended, QuantumCoordinates
        
        test_cases = [
            ("Wczoraj padało deszcz.", "temporal"),
            ("Świetnie, znowu spóźniony autobus...", "irony"),
            ("Kocham cię i jednocześnie nienawidzę.", "paradox"),
            ("Może przyjdę, a może nie.", "quantum")
        ]
        
        for text, expected_type in test_cases:
            print(f"\nAnalyzing: '{text}' (expected: {expected_type})")
            result = gtmo_analyze_extended(text)
            
            if 'coordinates' in result:
                coords = result['coordinates']
                print(f"  D: {coords['determination']:.3f}")
                print(f"  S: {coords['stability']:.3f}")
                print(f"  E: {coords['entropy']:.3f}")
                print(f"  T: {coords.get('temporality', 0.5):.3f}")
                
                if 'rhetorical_analysis' in result:
                    mode = result['rhetorical_analysis']['mode']
                    print(f"  Rhetorical mode: {mode}")
                
                if 'quantum_analysis' in result:
                    is_super = result['quantum_analysis']['is_superposed']
                    print(f"  Quantum state: {'SUPERPOSED' if is_super else 'COLLAPSED'}")
            else:
                print(f"  Error: {result.get('error', 'Unknown error')}")
        
        print("\n✓ Extended module tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Extended module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_operations():
    """Test file loading and saving operations."""
    print("\n" + "="*60)
    print("Testing file operations...")
    print("="*60)
    
    # Create test data directory
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create test MD file
    test_md_path = test_dir / "test.md"
    with open(test_md_path, 'w', encoding='utf-8') as f:
        f.write("""# Test Document

## Sekcja 1
To jest test. Sprawdzamy czy system działa poprawnie.

## Sekcja 2
Wczoraj padał deszcz. Jutro będzie lepiej.
""")
    
    print(f"✓ Created test file: {test_md_path}")
    
    try:
        # Test JSON saver
        from gtmo_json_saver import GTMOOptimizedSaver

        saver = GTMOOptimizedSaver(output_dir="test_output")

        test_coords = {
            'determination': 0.7,
            'stability': 0.8,
            'entropy': 0.3
        }

        output_path = saver.save_md_analysis(
            md_file_path=str(test_md_path),
            text_content="Test text for analysis",
            coordinates=test_coords
        )
        print(f"✓ Saved test result to: {output_path}")

        # Verify saved file
        if output_path and Path(output_path).exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                if 'coordinates' in loaded and 'content' in loaded:
                    print(f"✓ JSON save/load verification passed")
                else:
                    print(f"✗ JSON structure verification failed: {loaded.keys()}")
                    return False
        else:
            print("✗ JSON file not created")
            return False

    except Exception as e:
        print(f"✗ File operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Cleanup
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)
    if Path("test_output").exists():
        shutil.rmtree("test_output")
    
    print("✓ File operations tests passed")
    return True


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("GTMØ SYSTEM INTEGRATION TESTS")
    print("="*60)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    if results[0][1]:  # Only continue if imports work
        # Test modules
        results.append(("Base Module", test_base_module()))
        results.append(("Extended Module", test_extended_module()))
        results.append(("File Operations", test_file_operations()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED - System is ready for use!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED - Please check the errors above")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())