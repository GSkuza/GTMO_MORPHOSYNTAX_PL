#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script for GTMØ API
"""

import requests
import sys
from pathlib import Path

def test_api(api_url: str, test_file: str):
    """Test the GTMØ API with a sample file"""

    print(f"🧪 Testing GTMØ API")
    print(f"   URL: {api_url}")
    print(f"   File: {test_file}")
    print("=" * 60)

    # 1. Health check
    print("\n1️⃣ Testing health endpoint...")
    try:
        response = requests.get(f"{api_url}/health")
        if response.ok:
            data = response.json()
            print(f"   ✓ API is healthy")
            print(f"   Components: {data['components']}")
        else:
            print(f"   ✗ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return

    # 2. Test analysis
    print("\n2️⃣ Testing analysis endpoint...")

    if not Path(test_file).exists():
        print(f"   ✗ Test file not found: {test_file}")
        return

    try:
        with open(test_file, 'rb') as f:
            files = {'file': (Path(test_file).name, f, 'text/plain')}
            params = {'use_llm': False}  # Disable LLM for faster testing

            print(f"   Uploading file...")
            response = requests.post(
                f"{api_url}/analyze",
                files=files,
                params=params,
                timeout=600  # 10 minutes
            )

        if response.ok:
            data = response.json()

            if data.get('success'):
                print(f"   ✓ Analysis successful!")

                # Display stats
                stats = data.get('aggregate_stats', {})
                print(f"\n   📊 Results:")
                print(f"      Articles: {stats.get('total_articles', 0)}")
                print(f"      Sentences: {stats.get('total_sentences', 0)}")
                print(f"      Average SA: {stats.get('average_SA', 0)}%")
                print(f"      Critical: {stats.get('critical_sentences', 0)}")
                print(f"      Warnings: {stats.get('warning_sentences', 0)}")

                # Show first few rows
                metrics = data.get('metrics_table', [])
                if metrics:
                    print(f"\n   📈 First 3 sentences:")
                    for i, row in enumerate(metrics[:3], 1):
                        print(f"      {i}. SA: {row['SA']}% | {row['text_preview'][:60]}...")

                print(f"\n   ✅ API test PASSED")
                return True
            else:
                print(f"   ✗ Analysis failed: {data.get('message')}")
                return False
        else:
            print(f"   ✗ Request failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Default values
    api_url = "http://localhost:8000"
    test_file = "../gtmo_results/analysis_21112025_no1_NEW_ustawa_o_trzezwosci/sample.txt"

    # Parse command line args
    if len(sys.argv) > 1:
        api_url = sys.argv[1]
    if len(sys.argv) > 2:
        test_file = sys.argv[2]

    success = test_api(api_url, test_file)
    sys.exit(0 if success else 1)
