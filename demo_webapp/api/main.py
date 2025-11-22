#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTMØ Web API - FastAPI Backend
================================
API endpoint dla analizy morfosyntaktycznej ustaw i dokumentów prawnych.
"""

import os
import sys
import json
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import GTMØ modules
try:
    from gtmo_results_analyse.gtmo_verdict_analyzer import GTMOVerdictAnalyzer, NaturalLanguageRecommendations
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False
    print("⚠️  Warning: gtmo_verdict_analyzer not available")

app = FastAPI(
    title="GTMØ Constitutional Metrics API",
    description="API dla analizy morfosyntaktycznej dokumentów prawnych",
    version="1.0.0"
)

# CORS configuration - VERY PERMISSIVE for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Set to False when using allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Add request logging middleware AND manual CORS headers
@app.middleware("http")
async def log_requests_and_add_cors(request, call_next):
    print(f"📥 {request.method} {request.url}")
    origin = request.headers.get('origin', 'N/A')
    print(f"   Origin: {origin}")

    # Handle preflight
    if request.method == "OPTIONS":
        from fastapi.responses import Response
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Max-Age": "3600",
            }
        )

    response = await call_next(request)

    # Add CORS headers to every response
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Expose-Headers"] = "*"

    print(f"📤 Response: {response.status_code}")
    print(f"   CORS Headers: {response.headers.get('Access-Control-Allow-Origin', 'MISSING!')}")
    return response

# Configuration
GTMO_ROOT = Path(__file__).parent.parent.parent
GTMO_MORPHOSYNTAX_SCRIPT = GTMO_ROOT / "gtmo_morphosyntax.py"
RESULTS_DIR = GTMO_ROOT / "gtmo_results"
TEMP_DIR = Path(tempfile.gettempdir()) / "gtmo_webapp"
TEMP_DIR.mkdir(exist_ok=True)


class AnalysisResponse(BaseModel):
    """Response model for analysis results"""
    success: bool
    message: str
    document_metadata: Optional[Dict[str, Any]] = None
    metrics_table: Optional[List[Dict[str, Any]]] = None
    recommendations: Optional[List[Dict[str, Any]]] = None
    aggregate_stats: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def run_gtmo_morphosyntax(file_path: str) -> tuple[bool, str, Optional[str]]:
    """
    Run gtmo_morphosyntax.py on the uploaded file

    Returns:
        (success, message, result_json_path)
    """
    try:
        # Run the analysis script
        result = subprocess.run(
            [sys.executable, str(GTMO_MORPHOSYNTAX_SCRIPT), file_path],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout
            encoding='utf-8',
            errors='replace'
        )

        if result.returncode != 0:
            return False, f"Analysis failed: {result.stderr[:500]}", None

        # Find the generated JSON file
        # The script creates: gtmo_results/analysis_[date]_[filename]/full_document.json
        file_name = Path(file_path).stem

        # Find latest matching directory
        matching_dirs = sorted(
            RESULTS_DIR.glob(f"analysis_*_{file_name}*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if not matching_dirs:
            return False, "No analysis results found", None

        result_json = matching_dirs[0] / "full_document.json"

        if not result_json.exists():
            return False, "Analysis completed but no JSON output found", None

        return True, "Analysis completed successfully", str(result_json)

    except subprocess.TimeoutExpired:
        return False, "Analysis timed out (>5 minutes)", None
    except Exception as e:
        return False, f"Error running analysis: {str(e)}", None


def extract_metrics_table(analysis_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract key metrics from analysis JSON and format as table rows

    Returns list of dicts with metrics for each sentence
    """
    table_rows = []

    articles = analysis_json.get('articles', [])

    for article in articles:
        article_num = article.get('article_number', 'N/A')
        sentences = article.get('sentences', [])

        for sentence in sentences:
            content = sentence.get('content', {})
            coords = sentence.get('coordinates', {})
            const_metrics = sentence.get('constitutional_metrics', {})
            depth_metrics = sentence.get('depth_metrics', {})
            additional = sentence.get('additional_metrics', {})

            # Extract SA (Semantic Accessibility)
            sa_obj = const_metrics.get('semantic_accessibility', {})
            if isinstance(sa_obj, dict):
                sa = (sa_obj.get('v3', {}).get('value') or
                      sa_obj.get('v2', {}).get('value') or
                      sa_obj.get('value', 0))
            else:
                sa = sa_obj

            # Extract CD (Constitutional Definiteness)
            cd_obj = const_metrics.get('definiteness', {})
            cd = cd_obj.get('value', 0) if isinstance(cd_obj, dict) else cd_obj

            # Extract CI (Constitutional Indefiniteness)
            ci_obj = const_metrics.get('indefiniteness', {})
            ci = ci_obj.get('value', 0) if isinstance(ci_obj, dict) else ci_obj

            # CI Decomposition
            decomp = ci_obj.get('decomposition', {}) if isinstance(ci_obj, dict) else {}
            ci_morph = decomp.get('morphological', {}).get('percentage', 0)
            ci_synt = decomp.get('syntactic', {}).get('percentage', 0)
            ci_sem = decomp.get('semantic', {}).get('percentage', 0)

            # Classification
            classification = const_metrics.get('classification', {})
            cls_type = classification.get('type', 'UNKNOWN') if isinstance(classification, dict) else 'UNKNOWN'

            row = {
                'article': article_num,
                'sentence_id': sentence.get('analysis_metadata', {}).get('sentence_number', 0),
                'text_preview': content.get('text', '')[:150],
                'full_text': content.get('text', ''),
                'D': round(coords.get('determination', 0), 3),
                'S': round(coords.get('stability', 0), 3),
                'E': round(coords.get('entropy', 0), 3),
                'SA': round(float(sa) * 100, 2) if sa else 0,  # Convert to percentage
                'CD': round(float(cd), 3) if cd else 0,
                'CI': round(float(ci), 3) if ci else 0,
                'CI_morph_%': round(float(ci_morph), 1) if ci_morph else 0,
                'CI_synt_%': round(float(ci_synt), 1) if ci_synt else 0,
                'CI_sem_%': round(float(ci_sem), 1) if ci_sem else 0,
                'depth': depth_metrics.get('max_depth', 0),
                'ambiguity': round(additional.get('ambiguity', 0), 2),
                'classification': cls_type
            }

            table_rows.append(row)

    return table_rows


def generate_recommendations(
    analysis_json_path: str,
    use_llm: bool = True
) -> List[Dict[str, Any]]:
    """
    Generate natural language recommendations using gtmo_verdict_analyzer

    Returns list of recommendation dicts
    """
    if not ANALYZER_AVAILABLE:
        return [{
            'error': 'Analyzer module not available',
            'message': 'Install required dependencies: anthropic, pandas, matplotlib'
        }]

    try:
        # Initialize analyzer
        analyzer = GTMOVerdictAnalyzer(analysis_json_path)

        # Generate recommendations
        recommender = NaturalLanguageRecommendations(
            use_llm=use_llm,
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )

        # Get problematic sentences (SA < 30%)
        problematic = analyzer.df[analyzer.df['SA'] < 0.30].copy()

        if len(problematic) == 0:
            return [{
                'success': True,
                'message': 'Świetnie! Wszystkie przepisy są czytelne (SA > 30%)',
                'problematic_count': 0
            }]

        recommendations = []

        for idx, row in problematic.iterrows():
            sentence_data = {
                'text': row['full_text'],
                'SA': row['SA'],
                'CI_morph_pct': row.get('CI_morph_pct', 0),
                'CI_synt_pct': row.get('CI_synt_pct', 0),
                'CI_sem_pct': row.get('CI_sem_pct', 0),
                'ambiguity': row.get('ambiguity', 0),
                'depth': row.get('depth', 0),
                'classification': row.get('classification', 'UNKNOWN')
            }

            rec = recommender.generate_recommendations(sentence_data)

            recommendations.append({
                'sentence_id': int(row['block_id']),
                'text_preview': row['text'][:100],
                'full_text': row['full_text'],
                'SA_percent': round(row['SA'] * 100, 2),
                **rec
            })

        return recommendations

    except Exception as e:
        return [{
            'error': str(e),
            'message': f'Failed to generate recommendations: {str(e)}'
        }]


@app.get("/")
async def root():
    """Health check endpoint"""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        content={
            "status": "ok",
            "service": "GTMØ Constitutional Metrics API",
            "version": "1.0.0",
            "analyzer_available": ANALYZER_AVAILABLE
        },
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )


@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    use_llm: bool = True
):
    """
    Analyze uploaded legal document (txt or md)

    Returns:
        - Document metadata
        - Metrics table with constitutional metrics for each sentence
        - Natural language recommendations from LLM
    """
    from fastapi.responses import JSONResponse

    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ['.txt', '.md']:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file_ext}. Only .txt and .md files are supported"
        )

    temp_file_path = None
    result_json_path = None

    try:
        # Save uploaded file to temporary location
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext,
            dir=TEMP_DIR
        )
        temp_file_path = temp_file.name

        # Write uploaded content
        content = await file.read()
        temp_file.write(content)
        temp_file.close()

        print(f"📄 Uploaded file saved: {temp_file_path}")

        # Step 1: Run morphosyntax analysis
        print("🔍 Running GTMØ morphosyntax analysis...")
        success, message, result_json_path = run_gtmo_morphosyntax(temp_file_path)

        if not success:
            return AnalysisResponse(
                success=False,
                message=message,
                error=message
            )

        print(f"✓ Analysis complete: {result_json_path}")

        # Step 2: Load and parse results
        with open(result_json_path, 'r', encoding='utf-8') as f:
            analysis_json = json.load(f)

        # Step 3: Extract metrics table
        print("📊 Extracting metrics table...")
        metrics_table = extract_metrics_table(analysis_json)

        # Step 4: Generate recommendations
        print("💡 Generating recommendations...")
        recommendations = generate_recommendations(result_json_path, use_llm=use_llm)

        # Step 5: Extract aggregate stats
        aggregate_stats = {
            'total_articles': analysis_json.get('document_metadata', {}).get('total_units', 0),
            'total_sentences': len(metrics_table),
            'total_words': analysis_json.get('document_metadata', {}).get('total_words', 0),
            'average_SA': round(
                sum(row['SA'] for row in metrics_table) / len(metrics_table) if metrics_table else 0,
                2
            ),
            'critical_sentences': len([r for r in metrics_table if r['SA'] < 10]),  # SA < 10%
            'warning_sentences': len([r for r in metrics_table if 10 <= r['SA'] < 30]),  # 10% ≤ SA < 30%
        }

        return AnalysisResponse(
            success=True,
            message="Analysis completed successfully",
            document_metadata=analysis_json.get('document_metadata', {}),
            metrics_table=metrics_table,
            recommendations=recommendations,
            aggregate_stats=aggregate_stats
        )

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

        return AnalysisResponse(
            success=False,
            message=f"Error: {str(e)}",
            error=str(e)
        )

    finally:
        # Cleanup temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass


@app.get("/health")
async def health():
    """Detailed health check"""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        content={
            "status": "healthy",
            "components": {
                "gtmo_morphosyntax": GTMO_MORPHOSYNTAX_SCRIPT.exists(),
                "analyzer": ANALYZER_AVAILABLE,
                "anthropic_key": bool(os.getenv('ANTHROPIC_API_KEY'))
            }
        },
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 GTMØ Backend Starting...")
    print("=" * 60)
    print(f"Backend URL: http://127.0.0.1:8000")
    print(f"API Docs: http://127.0.0.1:8000/docs")
    print(f"CORS: Enabled for all origins")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
