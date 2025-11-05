#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug Unicode AssertionError
"""

import gtmo_morphosyntax as gtmo
import traceback

# Test problematic Unicode characters that caused AssertionError
test_cases = [
    "ÃéśɃⴇĖ",
    "Ĺ1u",
    "inverted",
    "𑢲ὠŒáļ§Ģ𑗏&ĿáâŞūłŝíÇŚzƮÌrĀõėlõ",
    "𞤤ꙇsù𐐐Թ",
    "Ãę𑇈ÎR𑙪𑇙Ĳ𖺗𝘓jĆµľĬ౮",
    "ĩ/!",
    "W",
    "Ò"
]

for i, text in enumerate(test_cases, 1):
    print(f"\n=== Test Case {i}: '{text}' ===")
    
    try:
        result = gtmo.analyze_quantum_with_axioms(text)
        print(f"✅ Success: Analysis completed")
        
        # Check if we have the expected structure
        assert isinstance(result, dict), "Result must be a dictionary"
        assert 'coordinates' in result, "Must have coordinates"
        assert 'content' in result, "Must have content"
        
        # Check content structure (it should be a dict with 'text' field)
        content = result['content']
        assert isinstance(content, dict), "Content must be a dictionary"
        assert 'text' in content, "Content must have 'text' field"
        assert content['text'] == text, f"Text mismatch: got '{content['text']}', expected '{text}'"
        
        print(f"✅ All assertions passed for: '{text}'")
        
    except AssertionError as e:
        print(f"❌ AssertionError: {e}")
        print(f"   This is the error we need to fix!")
        traceback.print_exc()
        
    except Exception as e:
        print(f"🔍 Other error ({type(e).__name__}): {e}")

print("\n" + "="*60)
print("Debug complete - we can see which assertions fail!")