#!/usr/bin/env python3
"""
Quick test script for all_in_one.py server
Tests the /analyze endpoint with a small text sample
"""
import requests
import time
from io import BytesIO

# Test text - short legal sample
TEST_TEXT = """Art. 1. Ustawa określa zasady ochrony zdrowia publicznego.

Art. 2. Ilekroć w ustawie jest mowa o napojach alkoholowych, rozumie się przez to napoje zawierające alkohol etylowy."""

def test_analyze_endpoint():
    """Test the /analyze endpoint with text input"""
    url = "http://127.0.0.1:8000/analyze"

    # Create a file-like object from text
    file_data = BytesIO(TEST_TEXT.encode('utf-8'))
    files = {'file': ('test.txt', file_data, 'text/plain')}

    print("=" * 60)
    print("🧪 Testing /analyze endpoint...")
    print("=" * 60)
    print(f"URL: {url}")
    print(f"Text length: {len(TEST_TEXT)} chars")
    print(f"Sample: {TEST_TEXT[:100]}...")
    print("\n⏳ Sending request (this may take 2-5 minutes)...\n")

    try:
        start_time = time.time()
        response = requests.post(url, files=files)
        elapsed = time.time() - start_time

        print(f"✓ Response received in {elapsed:.1f} seconds")
        print(f"Status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            print("\n" + "=" * 60)
            print("📊 RESULTS:")
            print("=" * 60)

            if data.get('success'):
                print("✅ Analysis successful!")

                # Document metadata
                metadata = data.get('document_metadata', {})
                print(f"\n📄 Document: {metadata.get('total_units', 0)} articles, "
                      f"{metadata.get('total_words', 0)} words")

                # Aggregate stats
                stats = data.get('aggregate_stats', {})
                print(f"\n📈 Statistics:")
                print(f"  - Total sentences: {stats.get('total_sentences', 0)}")
                print(f"  - Average SA: {stats.get('average_SA', 0):.2f}%")
                print(f"  - Critical (SA<10%): {stats.get('critical_sentences', 0)}")
                print(f"  - Warning (10%≤SA<30%): {stats.get('warning_sentences', 0)}")

                # Show first few metrics
                metrics = data.get('metrics_table', [])
                if metrics:
                    print(f"\n📋 First sentence metrics:")
                    first = metrics[0]
                    print(f"  - Text: {first.get('text_preview', '')[:80]}...")
                    print(f"  - SA: {first.get('SA', 0):.2f}%")
                    print(f"  - D: {first.get('D', 0):.3f}")
                    print(f"  - S: {first.get('S', 0):.3f}")
                    print(f"  - E: {first.get('E', 0):.3f}")
                    print(f"  - Classification: {first.get('classification', 'N/A')}")

                # Recommendations
                recs = data.get('recommendations', [])
                print(f"\n💡 Recommendations: {len(recs)} generated")
                if recs and len(recs) > 0:
                    rec = recs[0]
                    print(f"  - First recommendation for sentence #{rec.get('sentence_id', 'N/A')}")
                    print(f"    SA: {rec.get('SA_percent', 0):.2f}%")
                    print(f"    Severity: {rec.get('severity', 'N/A')}")

                print("\n" + "=" * 60)
                print("✅ ALL TESTS PASSED!")
                print("=" * 60)
                return True

            else:
                print(f"❌ Analysis failed: {data.get('message', 'Unknown error')}")
                return False
        else:
            print(f"❌ HTTP Error {response.status_code}")
            print(response.text[:500])
            return False

    except requests.exceptions.ConnectionError:
        print("❌ Connection error - is the server running?")
        print("   Run: python demo_webapp/api/all_in_one.py")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_analyze_endpoint()
    exit(0 if success else 1)
