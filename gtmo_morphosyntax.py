#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTMØ Polish Morphosyntax Engine - WORKING VERSION
================================================
Direct implementation. No classes. No bullshit.
"""

import numpy as np
from typing import Dict, Tuple, List
import logging
from gtmo_pure_rhetoric import integrate_with_gtmo
from gtmo_json_saver import GTMOOptimizedSaver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Required imports
try:
    import morfeusz2
    morfeusz = morfeusz2.Morfeusz()
    print("✓ Morfeusz2 loaded")
except ImportError:
    morfeusz = None
    print("✗ Morfeusz2 missing: pip install morfeusz2")

try:
    import spacy
    nlp = spacy.load('pl_core_news_lg')
    print("✓ spaCy loaded")
except:
    try:
        nlp = spacy.load('pl_core_news_sm')
        print("✓ spaCy (small) loaded")
    except:
        nlp = None
        print("✗ spaCy missing: pip install spacy && python -m spacy download pl_core_news_lg")

# GTMØ coordinates for Polish cases (corrected for better discrimination)
CASE_COORDS = {
    'nom': np.array([0.95, 0.92, 0.08]),  # Nominative - subject, highly predictable
    'gen': np.array([0.55, 0.25, 0.88]),  # Genitive - most complex, many functions
    'dat': np.array([0.72, 0.65, 0.35]),  # Dative - recipient, moderate
    'acc': np.array([0.89, 0.85, 0.15]),  # Accusative - direct object, predictable
    'ins': np.array([0.42, 0.18, 0.95]),  # Instrumental - means/manner, max entropy
    'loc': np.array([0.78, 0.95, 0.12]),  # Locative - location only, max stability
    'voc': np.array([0.65, 0.35, 0.75])   # Vocative - address, contextual
}

# GTMØ coordinates for Polish POS tags
POS_COORDS = {
    'subst': np.array([0.80, 0.85, 0.20]),  # ✓ OK
    'adj': np.array([0.65, 0.68, 0.32]),    # ↑S z 0.50→0.65, ↓E z 0.45→0.35
    'verb': np.array([0.70, 0.45, 0.65]),   # ↑S z 0.30→0.45 (fleksja stabilizuje)
    'adv': np.array([0.52, 0.38, 0.68]),    # ↓D z 0.60→0.55, ↑E z 0.55→0.65
    'num': np.array([0.95, 0.90, 0.10]),    # ✓ OK
    'pron': np.array([0.68, 0.52, 0.53]),   # ↓D z 0.75→0.70, ↑E z 0.40→0.50
    'prep': np.array([0.76, 0.75, 0.24]),   # ↑D z 0.70→0.75, ↓S z 0.90→0.75
    'conj': np.array([0.65, 0.85, 0.20]),   # ✓ OK
    'part': np.array([0.40, 0.26, 0.84]),   # ✓ OK
    'interp': np.array([0.95, 0.95, 0.05])  # ✓ OK
}

def analyze_morphology(text: str) -> Tuple[np.ndarray, Dict]:
    """Morphological analysis using Morfeusz2."""
    if not morfeusz:
        raise Exception("Morfeusz2 not available")
    
    coords_list = []
    case_counts = {}
    pos_counts = {}
    debug_info = []
    
    # Extended case mapping for full Morfeusz2 tags
    case_mapping = {
        'nom': 'nom', 'gen': 'gen', 'dat': 'dat', 'acc': 'acc', 
        'ins': 'ins', 'inst': 'ins', 'loc': 'loc', 'voc': 'voc'
    }
    
    try:
        analyses = morfeusz.analyse(text)
        
        for start, end, (form, lemma, tag, labels, qualifiers) in analyses:
            if not form or form.isspace():
                continue
                
            debug_info.append(f"{form}:{tag}")
            tag_parts = tag.split(':')
            main_pos = tag_parts[0]
            
            # Extract case - look through all tag parts
            case_found = False
            for part in tag_parts:
                # Check direct cases
                if part in CASE_COORDS:
                    coords_list.append(CASE_COORDS[part])
                    case_counts[part] = case_counts.get(part, 0) + 1
                    case_found = True
                    break
                # Check compound cases (like 'nom.acc')
                elif '.' in part:
                    sub_parts = part.split('.')
                    for sub_part in sub_parts:
                        if sub_part in CASE_COORDS:
                            coords_list.append(CASE_COORDS[sub_part])
                            case_counts[sub_part] = case_counts.get(sub_part, 0) + 1
                            case_found = True
                            break
                if case_found:
                    break
            
            # Extract POS
            if main_pos in POS_COORDS:
                coords_list.append(POS_COORDS[main_pos])
                pos_counts[main_pos] = pos_counts.get(main_pos, 0) + 1
        
        if coords_list:
            final_coords = np.mean(coords_list, axis=0)
        else:
            final_coords = np.array([0.5, 0.5, 0.5])
            
        metadata = {
            'total_analyses': len(analyses),
            'cases': case_counts,
            'pos': pos_counts,
            'ambiguity': len(analyses) / len(text.split()) if text.split() else 1.0,
            'debug_tags': debug_info[:10],  # First 10 for debugging
            'coords_count': len(coords_list)
        }
        
        return final_coords, metadata
        
    except Exception as e:
        raise Exception(f"Morphological analysis failed: {e}")

def analyze_syntax(text: str) -> Tuple[np.ndarray, Dict]:
    """Syntactic analysis using spaCy."""
    if not nlp:
        raise Exception("spaCy not available")
    
    try:
        doc = nlp(text)
        
        # Base coordinates
        coords = np.array([0.5, 0.5, 0.5])
        
        # Dependency adjustments
        dep_adjustments = {
            'ROOT': np.array([0.2, 0.1, -0.1]),
            'nsubj': np.array([0.15, 0.05, -0.05]),
            'obj': np.array([0.1, 0.0, 0.05]),
            'advmod': np.array([-0.1, -0.05, 0.15]),
            'det': np.array([0.1, 0.15, -0.1]),
            'punct': np.array([0.0, 0.2, -0.15])
        }
        
        dep_counts = {}
        max_depth = 0
        
        for token in doc:
            dep = token.dep_
            dep_counts[dep] = dep_counts.get(dep, 0) + 1
            
            if dep in dep_adjustments:
                coords += dep_adjustments[dep] * 0.1
            
            depth = len(list(token.ancestors))
            max_depth = max(max_depth, depth)
        
        # Complexity adjustment
        if max_depth > 5:
            coords[2] += (max_depth - 5) * 0.05  # Add entropy
            coords[0] -= (max_depth - 5) * 0.02  # Reduce determination
        
        coords = np.clip(coords, 0, 1)
        
        metadata = {
            'tokens': len(doc),
            'sentences': len(list(doc.sents)),
            'max_depth': max_depth,
            'dependencies': dep_counts
        }
        
        return coords, metadata
        
    except Exception as e:
        raise Exception(f"Syntactic analysis failed: {e}")

def gtmo_analyze(text: str) -> Dict:
    """
    Main GTMØ analysis function.
    
    Args:
        text: Polish text to analyze
        
    Returns:
        Dict with coordinates and metadata
    """
    if not text or not text.strip():
        raise Exception("Empty text")
    
    print(f"Analyzing: {text[:50]}...")
    
    # Run both analyses
    morph_coords, morph_meta = analyze_morphology(text)
    synt_coords, synt_meta = analyze_syntax(text)
    
    # DEBUG: Print what we found
    print(f"  MORPHOLOGY: coords_found={morph_meta.get('coords_count', 0)}, cases={morph_meta['cases']}, pos={morph_meta['pos']}")
    print(f"  MORPH_COORDS: [{morph_coords[0]:.3f}, {morph_coords[1]:.3f}, {morph_coords[2]:.3f}]")
    print(f"  SYNTAX_COORDS: [{synt_coords[0]:.3f}, {synt_coords[1]:.3f}, {synt_coords[2]:.3f}]")
    if 'debug_tags' in morph_meta:
        print(f"  MORPH_TAGS: {morph_meta['debug_tags']}")
    
    # Fusion with weights (morphology is more important in Polish)
    morph_weight = 0.64
    synt_weight = 0.36
    
    final_coords = morph_weight * morph_coords + synt_weight * synt_coords
    final_coords = np.clip(final_coords, 0, 1)
    
    result = {
        'text': text,
        'coordinates': {
            'determination': float(final_coords[0]),
            'stability': float(final_coords[1]),
            'entropy': float(final_coords[2])
        },
        'morphology': morph_meta,
        'syntax': synt_meta,
        'fusion_weights': {'morphology': morph_weight, 'syntax': synt_weight}
    }
    
    print(f"  FINAL: D={final_coords[0]:.3f}, S={final_coords[1]:.3f}, E={final_coords[2]:.3f}")
    
    # Czysta detekcja retoryczna
    try:
        from gtmo_pure_rhetoric import GTMORhetoricalAnalyzer

        analyzer = GTMORhetoricalAnalyzer()

        # Przekaż rzeczywiste współrzędne składniowe
        syntax_coords_real = synt_coords  # z linii ~92
        morph_coords_real = morph_coords  # z linii ~91

        # Analiza anomalii
        rhetorical_coords, rhetorical_mode, rhetorical_meta = analyzer.analyze_rhetorical_mode(
            text,
            morph_coords_real,
            syntax_coords_real, 
            morph_meta,
            synt_meta
        )

        # Jeśli wykryto tryb retoryczny, zaktualizuj współrzędne
        if rhetorical_mode != 'literal':
            final_coords = rhetorical_coords
            result['coordinates'] = {
                'determination': float(final_coords[0]),
                'stability': float(final_coords[1]),
                'entropy': float(final_coords[2])
            }
            result['rhetorical_mode'] = rhetorical_mode
            result['rhetorical_analysis'] = rhetorical_meta

            print(f"  RHETORICAL: Detected {rhetorical_mode.upper()}")
            if rhetorical_mode == 'irony':
                print(f"    Indicators: {rhetorical_meta['irony_analysis']['irony_indicators']}")

    except ImportError:
        pass  # Moduł retoryczny opcjonalny

    return result

def batch_analyze(texts: List[str]) -> List[Dict]:
    """Analyze multiple texts."""
    results = []
    for i, text in enumerate(texts, 1):
        print(f"\nBatch {i}/{len(texts)}")
        try:
            result = gtmo_analyze(text)
            results.append(result)
        except Exception as e:
            print(f"Failed: {e}")
            results.append({'text': text, 'error': str(e)})
    return results

# Test examples
if __name__ == "__main__":
    import sys
    import pathlib

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        with open(input_path, encoding="utf-8") as f:
            text = f.read()
        import spacy
        nlp = spacy.load("pl_core_news_sm")
        doc = nlp(text)
        saver = GTMOOptimizedSaver()
        # --- ANALIZA PO ZDANIACH ---
        for i, sent in enumerate(doc.sents, 1):
            print(f"\nAnalyzing sentence {i}: {sent.text}")
            result = gtmo_analyze(sent.text)
            saver.save_md_analysis(
                md_file_path=input_path,
                text_content=sent.text,
                coordinates=result['coordinates'],
                additional_metrics=result.get('morphology')
            )
            print(f"  D={result['coordinates']['determination']:.3f}, S={result['coordinates']['stability']:.3f}, E={result['coordinates']['entropy']:.3f}")
    else:
        test_texts = [
            "Skąd się biorą dzieci?",
            "Ona ma tunele w uszach.",
            "Wczoraj padało, a może nie będzie?",
            "Kocham cię bardzo mocno i tak samo nienawidzę!",
            "PraWO jest prawem."
        ]
        
        print("GTMØ Polish Morphosyntax Engine - Test Run")
        print("=" * 50)
        
        if not morfeusz or not nlp:
            print("Missing required components. Install:")
            print("pip install morfeusz2 spacy")
            print("python -m spacy download pl_core_news_lg")
        else:
            results = batch_analyze(test_texts)
            
            print("\nSummary:")
            for r in results:
                if 'coordinates' in r:
                    c = r['coordinates']
                    print(f"'{r['text']}' -> D={c['determination']:.3f}, S={c['stability']:.3f}, E={c['entropy']:.3f}")
                else:
                    print(f"'{r['text']}' -> ERROR: {r['error']}")
    
    import spacy
    nlp = spacy.load("pl_core_news_sm")
    from gtmo_morphosyntax import gtmo_analyze
    from gtmo_json_saver import GTMOOptimizedSaver

    saver = GTMOOptimizedSaver()

    with open("C:\\Users\\develop22\\Downloads\\gtmo-session-summary.md", encoding="utf-8") as f:
        text = f.read()

    doc = nlp(text)
    for i, token in enumerate(doc, 1):
        if not token.is_space:
            result = gtmo_analyze(token.text)
            saver.save_md_analysis(
                md_file_path="C:\\Users\\develop22\\Downloads\\gtmo-session-summary.md",
                text_content=token.text,
                coordinates=result['coordinates'],
                additional_metrics=result.get('morphology')
            )
            print(f"Word {i}: {token.text}")
            print(f"  D={result['coordinates']['determination']:.3f}, S={result['coordinates']['stability']:.3f}, E={result['coordinates']['entropy']:.3f}")
