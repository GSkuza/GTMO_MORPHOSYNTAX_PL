#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch processor for GTMØ Verdict Analyzer - processes all article JSONs in a folder.
"""

import sys
import os
from pathlib import Path
import subprocess
import json

def process_all_articles(analysis_folder: str):
    """
    Process all article_*.json files in the analysis folder.

    Args:
        analysis_folder: Path to the analysis folder containing article_*.json files
    """
    analysis_path = Path(analysis_folder)

    if not analysis_path.exists():
        print(f"ERROR: Folder not found: {analysis_folder}")
        return

    # Find all article JSON files
    article_files = sorted(analysis_path.glob("article_*.json"))

    if not article_files:
        print(f"ERROR: No article_*.json files found in {analysis_folder}")
        return

    print("="*80)
    print(f"BATCH PROCESSING: {len(article_files)} articles")
    print("="*80)

    # Process each article
    results = []
    for i, article_file in enumerate(article_files, 1):
        print(f"\n[{i}/{len(article_files)}] Processing: {article_file.name}")

        try:
            # Run analyzer on this article
            cmd = [
                sys.executable,
                "gtmo_results_analyse/gtmo_verdict_analyzer.py",
                str(article_file)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                print(f"   ✓ SUCCESS")
                results.append({"file": article_file.name, "status": "SUCCESS"})
            else:
                print(f"   ✗ FAILED: {result.stderr[:200]}")
                results.append({"file": article_file.name, "status": "FAILED", "error": result.stderr[:200]})

        except subprocess.TimeoutExpired:
            print(f"   ✗ TIMEOUT")
            results.append({"file": article_file.name, "status": "TIMEOUT"})
        except Exception as e:
            print(f"   ✗ ERROR: {e}")
            results.append({"file": article_file.name, "status": "ERROR", "error": str(e)})

    # Summary
    print("\n" + "="*80)
    print("BATCH PROCESSING SUMMARY")
    print("="*80)

    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    failed_count = len(results) - success_count

    print(f"Total articles: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")

    if failed_count > 0:
        print("\nFailed articles:")
        for r in results:
            if r["status"] != "SUCCESS":
                print(f"   • {r['file']}: {r['status']}")

    # Save summary
    summary_file = analysis_path / "batch_processing_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total": len(results),
            "success": success_count,
            "failed": failed_count,
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Summary saved to: {summary_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_all_articles.py <analysis_folder>")
        print("Example: python process_all_articles.py gtmo_results/analysis_20112025_no1_RP_KONSTYTUCJA_20112025")
        sys.exit(1)

    analysis_folder = sys.argv[1]
    process_all_articles(analysis_folder)
