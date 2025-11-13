#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug test for GTMØ import issues
"""

import sys
import io

print("Starting import debug...")

# Test basic imports first
try:
    import numpy as np
    print("✓ NumPy OK")
except ImportError as e:
    print(f"✗ NumPy failed: {e}")

try:
    import morfeusz2
    print("✓ Morfeusz2 OK")
except ImportError as e:
    print(f"✗ Morfeusz2 failed: {e}")

try:
    import spacy
    print("✓ spaCy OK")
except ImportError as e:
    print(f"✗ spaCy failed: {e}")

# Test GTMØ modules one by one
try:
    import gtmo_pure_rhetoric
    print("✓ gtmo_pure_rhetoric OK")
except Exception as e:
    print(f"✗ gtmo_pure_rhetoric failed: {e}")

try:
    import gtmo_extended
    print("✓ gtmo_extended OK")
except Exception as e:
    print(f"✗ gtmo_extended failed: {e}")

# Finally test main module
try:
    print("Attempting to import gtmo_morphosyntax...")
    import gtmo_morphosyntax
    print("✓ gtmo_morphosyntax OK")
except Exception as e:
    print(f"✗ gtmo_morphosyntax failed: {e}")
    import traceback
    traceback.print_exc()

print("Import debug complete.")