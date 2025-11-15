#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper for gtmo_verdict_analyzer.py with proper UTF-8 encoding for Windows
"""
import sys
import os

# Force UTF-8 encoding for stdout/stderr
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Import and run the main script
if __name__ == '__main__':
    from gtmo_verdict_analyzer import main
    main()
