#!/usr/bin/env python3
"""
ALL-IN-ONE GTMØ Backend + Frontend
Serwuje frontend i backend z jednego portu - BEZ PROBLEMÓW Z CORS!
"""
import os
import sys
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

app = FastAPI(title="GTMØ All-in-One")

# Serve static files (CSS, JS)
DOCS_DIR = Path(__file__).parent.parent / "docs"
app.mount("/css", StaticFiles(directory=str(DOCS_DIR / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(DOCS_DIR / "js")), name="js")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the main page with real backend integration"""
    index_path = DOCS_DIR / "index_backend.html"

    if not index_path.exists():
        return HTMLResponse(content="<h1>Error: index_backend.html not found</h1>", status_code=500)

    with open(index_path, 'r', encoding='utf-8') as f:
        html = f.read()

    return HTMLResponse(content=html)

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "message": "All-in-one backend - no CORS issues!"
    }

@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    use_llm: bool = False  # Default False for faster testing
):
    """
    REAL analyze endpoint - runs GTMØ morphosyntax analysis

    Note: This takes 2-5 minutes!
    """
    import tempfile
    import subprocess
    import json
    from pathlib import Path

    GTMO_ROOT = Path(__file__).parent.parent.parent
    GTMO_MORPHOSYNTAX_SCRIPT = GTMO_ROOT / "gtmo_morphosyntax.py"
    RESULTS_DIR = GTMO_ROOT / "gtmo_results"

    temp_file_path = None
    result_json_path = None

    try:
        # Validate file
        if not file.filename:
            return {"success": False, "message": "No filename provided", "error": "No filename"}

        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ['.txt', '.md']:
            return {"success": False, "message": f"Invalid file type: {file_ext}", "error": "Invalid file type"}

        # Save uploaded file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        temp_file_path = temp_file.name

        content = await file.read()
        temp_file.write(content)
        temp_file.close()

        print(f"📄 File saved: {temp_file_path}")

        # Record timestamp BEFORE analysis
        import time
        start_time = time.time()

        # Run gtmo_morphosyntax.py
        print("🔍 Running GTMØ analysis...")
        print(f"   Command: {sys.executable} {GTMO_MORPHOSYNTAX_SCRIPT} {temp_file_path}")

        result = subprocess.run(
            [sys.executable, str(GTMO_MORPHOSYNTAX_SCRIPT), temp_file_path],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes
            encoding='utf-8',
            errors='replace',
            cwd=str(GTMO_ROOT)  # Set working directory to GTMO root!
        )

        print(f"   Return code: {result.returncode}")
        if result.stdout:
            print(f"   STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"   STDERR (first 1000 chars): {result.stderr[:1000]}")

        if result.returncode != 0:
            return {
                "success": False,
                "message": f"Analysis failed: {result.stderr[:500]}",
                "error": result.stderr
            }

        # Parse folder name from stdout instead of searching by timestamp
        import re
        folder_match = re.search(r'Created analysis folder: (.+)', result.stdout)

        if not folder_match:
            return {"success": False, "message": "Could not parse folder name from output", "error": "Parse error"}

        folder_path_str = folder_match.group(1).strip()
        print(f"🔍 Parsed folder from stdout: {folder_path_str}")

        # Convert to absolute path
        result_folder = Path(folder_path_str)
        if not result_folder.is_absolute():
            result_folder = GTMO_ROOT / folder_path_str

        print(f"🔍 Full folder path: {result_folder}")
        print(f"🔍 Folder exists: {result_folder.exists()}")

        if not result_folder.exists():
            return {"success": False, "message": f"Folder not found: {result_folder}", "error": "Folder missing"}

        result_json_path = result_folder / "full_document.json"

        if not result_json_path.exists():
            return {"success": False, "message": "JSON not found", "error": "JSON not found"}

        print(f"✓ Analysis complete: {result_json_path}")

        # Load JSON
        with open(result_json_path, 'r', encoding='utf-8') as f:
            analysis_json = json.load(f)

        # Extract metrics table
        metrics_table = []
        articles = analysis_json.get('articles', [])

        for article in articles:
            article_num = article.get('article_number', 'N/A')

            # Handle TWO structures:
            # 1. With paragraphs: articles[] → paragraphs[] → sentences[]
            # 2. Without paragraphs: articles[] → sentences[]
            if 'paragraphs' in article:
                paragraphs = article.get('paragraphs', [])
                all_sentences = []
                for paragraph in paragraphs:
                    all_sentences.extend(paragraph.get('sentences', []))
            else:
                # No paragraphs - sentences directly in article
                all_sentences = article.get('sentences', [])

            for sentence in all_sentences:
                content_data = sentence.get('content', {})
                coords = sentence.get('coordinates', {})
                const_metrics = sentence.get('constitutional_metrics', {})
                depth_metrics = sentence.get('depth_metrics', {})
                additional = sentence.get('additional_metrics', {})

                # Extract SA
                sa_obj = const_metrics.get('semantic_accessibility', {})
                if isinstance(sa_obj, dict):
                    sa = (sa_obj.get('v3', {}).get('value') or
                          sa_obj.get('v2', {}).get('value') or
                          sa_obj.get('value', 0))
                else:
                    sa = sa_obj or 0

                # Extract CD, CI
                cd_obj = const_metrics.get('definiteness', {})
                cd = cd_obj.get('value', 0) if isinstance(cd_obj, dict) else cd_obj or 0

                ci_obj = const_metrics.get('indefiniteness', {})
                ci = ci_obj.get('value', 0) if isinstance(ci_obj, dict) else ci_obj or 0

                # CI decomposition
                decomp = ci_obj.get('decomposition', {}) if isinstance(ci_obj, dict) else {}
                ci_morph = decomp.get('morphological', {}).get('percentage', 0)
                ci_synt = decomp.get('syntactic', {}).get('percentage', 0)
                ci_sem = decomp.get('semantic', {}).get('percentage', 0)

                # Classification
                classification = const_metrics.get('classification', {})
                cls_type = classification.get('type', 'UNKNOWN') if isinstance(classification, dict) else 'UNKNOWN'

                text = content_data.get('text', '')

                row = {
                    'article': article_num,
                    'sentence_id': sentence.get('analysis_metadata', {}).get('sentence_number', 0),
                    'text_preview': text[:150],
                    'full_text': text,
                    'D': round(coords.get('determination', 0), 3),
                    'S': round(coords.get('stability', 0), 3),
                    'E': round(coords.get('entropy', 0), 3),
                    'SA': round(float(sa) * 100, 2) if sa else 0,
                    'CD': round(float(cd), 3) if cd else 0,
                    'CI': round(float(ci), 3) if ci else 0,
                    'CI_morph_%': round(float(ci_morph), 1) if ci_morph else 0,
                    'CI_synt_%': round(float(ci_synt), 1) if ci_synt else 0,
                    'CI_sem_%': round(float(ci_sem), 1) if ci_sem else 0,
                    'depth': depth_metrics.get('max_depth', 0),
                    'ambiguity': round(additional.get('ambiguity', 0), 2),
                    'classification': cls_type
                }

                metrics_table.append(row)

        # Aggregate stats
        total_words = analysis_json.get('document_metadata', {}).get('total_words', 0)
        avg_sa = round(sum(r['SA'] for r in metrics_table) / len(metrics_table), 2) if metrics_table else 0

        aggregate_stats = {
            'total_articles': analysis_json.get('document_metadata', {}).get('total_units', 0),
            'total_sentences': len(metrics_table),
            'total_words': total_words,
            'average_SA': avg_sa,
            'critical_sentences': len([r for r in metrics_table if r['SA'] < 10]),
            'warning_sentences': len([r for r in metrics_table if 10 <= r['SA'] < 30]),
        }

        # Generate recommendations using LLM or simple templates
        recommendations = []

        # When LLM is enabled, select sentences with LOWEST SA (regardless of threshold)
        # When LLM is disabled, only show sentences with SA < 30%
        print(f"🔍 use_llm parameter: {use_llm}")
        print(f"🔍 metrics_table length: {len(metrics_table)}")

        if use_llm:
            # Sort by SA and take 3 sentences with lowest accessibility
            sorted_sentences = sorted(metrics_table, key=lambda x: x['SA'])
            problematic_sentences = sorted_sentences[:3]
            print(f"🔍 LLM mode: Selected {len(problematic_sentences)} sentences with lowest SA")
        else:
            # Simple mode: only sentences with SA < 30%
            problematic_sentences = [r for r in metrics_table if r['SA'] < 30][:3]
            print(f"🔍 Simple mode: Selected {len(problematic_sentences)} sentences with SA < 30%")

        print(f"🔍 problematic_sentences count: {len(problematic_sentences)}")
        print(f"🔍 Checking condition: use_llm={use_llm} AND problematic_sentences={bool(problematic_sentences)}")

        if use_llm and problematic_sentences:
            try:
                # Import LLM recommendation system
                sys.path.insert(0, str(GTMO_ROOT / "gtmo_results_analyse"))
                from gtmo_verdict_analyzer import NaturalLanguageRecommendations

                # Initialize recommender with API key
                api_key = os.getenv('ANTHROPIC_API_KEY')
                recommender = NaturalLanguageRecommendations(use_llm=True, api_key=api_key)

                print(f"🤖 Generating LLM recommendations for {len(problematic_sentences)} sentences...")

                for row in problematic_sentences:
                    # Prepare sentence data for LLM
                    sentence_data = {
                        'text': row['full_text'],
                        'SA': row['SA'] / 100,  # Convert back to 0-1 range
                        'CI_morph_pct': row.get('CI_morph_%', 0),
                        'CI_synt_pct': row.get('CI_synt_%', 0),
                        'CI_sem_pct': row.get('CI_sem_%', 0),
                        'ambiguity': row.get('ambiguity', 0),
                        'depth': row.get('depth', 0),
                        'classification': row.get('classification', 'UNKNOWN')
                    }

                    # Generate LLM recommendation
                    rec = recommender.generate_recommendations(sentence_data)

                    recommendations.append({
                        'sentence_id': row['sentence_id'],
                        'text_preview': row['text_preview'],
                        'full_text': row['full_text'],
                        'SA_percent': row['SA'],
                        **rec  # Merge LLM response
                    })

                print(f"✓ LLM recommendations generated successfully")

            except Exception as e:
                print(f"⚠️ LLM recommendation failed: {str(e)}")
                print("   Falling back to simple recommendations...")
                use_llm = False  # Fallback to simple mode

        if not use_llm:
            # Simple stub recommendations (fast, no LLM)
            for row in problematic_sentences:
                recommendations.append({
                    'sentence_id': row['sentence_id'],
                    'text_preview': row['text_preview'],
                    'full_text': row['full_text'],
                    'SA_percent': row['SA'],
                    'severity': 'trudny do zrozumienia' if row['SA'] < 10 else 'średnio czytelny',
                    'main_problem_short': 'niska dostępność semantyczna',
                    'main_problem_detailed': f"SA = {row['SA']}% - tekst wymaga uproszczenia",
                    'quick_fixes': ['Rozbij zdanie na krótsze', 'Uprość słownictwo'],
                    'long_term_fixes': ['Przepisz artykuł prostszym językiem'],
                    'legal_risks': 'Ryzyko problemów z interpretacją',
                    'example_better_version': f'Oryginalny tekst: "{row["full_text"][:100]}..."\n\nProponowana poprawa: Uprość i rozbij na krótsze zdania.'
                })

        return {
            "success": True,
            "message": "Analysis completed successfully",
            "document_metadata": analysis_json.get('document_metadata', {}),
            "metrics_table": metrics_table,
            "recommendations": recommendations,
            "aggregate_stats": aggregate_stats
        }

    except subprocess.TimeoutExpired:
        return {"success": False, "message": "Analysis timeout (>5 min)", "error": "Timeout"}
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "message": f"Error: {str(e)}", "error": str(e)}
    finally:
        # Cleanup
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    import sys
    import io

    # Fix Windows console encoding for emojis
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    print("=" * 70)
    print("🎉 GTMØ ALL-IN-ONE SERVER")
    print("=" * 70)
    print("✅ Frontend + Backend on SAME PORT")
    print("✅ NO CORS ISSUES!")
    print("=" * 70)
    print("Open in browser: http://localhost:8000")
    print("                 http://127.0.0.1:8000")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8000)
