#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTMØ GENERAL TEXT PROCESSOR
============================
Wrapper for processing general (non-legal) text with sentence-level HerBERT embeddings.
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from scipy.spatial.distance import cosine
import spacy
import re

# Import GTMØ components
from gtmo_morphosyntax import analyze_quantum_with_axioms
from gtmo_json_saver import GTMOOptimizedSaver
from gtmo_herbert_mapping_9d import HerBERTtoGTMO9DMapper

# Load spaCy
try:
    nlp = spacy.load('pl_core_news_lg')
except:
    try:
        nlp = spacy.load('pl_core_news_sm')
    except:
        print("❌ spaCy not found. Install with: pip install spacy && python -m spacy download pl_core_news_sm")
        sys.exit(1)


def load_text_file(file_path):
    """Load text from markdown or txt file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Remove markdown formatting
        content = re.sub(r'#+ ', '', content)  # Remove headers
        content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # Remove bold
        content = re.sub(r'\*(.*?)\*', r'\1', content)  # Remove italic
        content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', content)  # Remove links
        content = re.sub(r'`(.*?)`', r'\1', content)  # Remove code

        return content
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python gtmo_general_text.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    print(f"🔍 Loading file: {file_path}")
    print("📝 Mode: General text (sentence-level analysis with HerBERT)")
    print("=" * 70)

    # Load text
    full_text = load_text_file(file_path)
    if not full_text:
        sys.exit(1)

    # Split into sentences
    doc = nlp(full_text)
    all_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip() and len(sent.text.strip()) >= 10]

    print(f"\n📝 Found {len(all_sentences)} sentences")

    # Initialize HerBERT mapper
    print("\n🤖 Initializing HerBERT...")
    try:
        herbert_mapper = HerBERTtoGTMO9DMapper(
            herbert_model_name="allegro/herbert-base-cased",
            device="cpu"
        )
        print("✅ HerBERT initialized")
    except Exception as e:
        print(f"❌ HerBERT initialization failed: {e}")
        sys.exit(1)

    # Initialize saver
    saver = GTMOOptimizedSaver()
    analysis_folder = saver.create_analysis_folder(os.path.basename(file_path))
    print(f"📁 Analysis folder: {analysis_folder}")

    # Analyze each sentence
    print(f"\n🌟 Analyzing {len(all_sentences)} sentences...")
    sentence_analyses = []

    for s_idx, sentence in enumerate(all_sentences, 1):
        try:
            print(f"  [{s_idx}/{len(all_sentences)}] Analyzing...", end='\r')

            # GTMØ quantum analysis
            sent_result = analyze_quantum_with_axioms(sentence, os.path.basename(file_path))
            sent_result['sentence_number'] = s_idx
            sent_result['text'] = sentence

            # HerBERT embedding (use from gtmo_morphosyntax if available)
            if 'herbert_embedding' not in sent_result:
                try:
                    sent_embedding = herbert_mapper.get_herbert_embedding(sentence)
                    sent_result['herbert_embedding'] = sent_embedding.tolist()
                    sent_result['herbert_magnitude'] = float(np.linalg.norm(sent_embedding))
                except Exception as e:
                    print(f"\n  ⚠️ HerBERT embedding failed for sentence {s_idx}: {e}")
            else:
                # Calculate magnitude from existing embedding
                sent_embedding = np.array(sent_result['herbert_embedding'])
                sent_result['herbert_magnitude'] = float(np.linalg.norm(sent_embedding))

            sentence_analyses.append(sent_result)

            # Save individual sentence using saver (extracts embeddings automatically)
            saver.save_sentence_analysis(sent_result, sentence, s_idx)

        except Exception as e:
            print(f"\n  ⚠️ Error analyzing sentence {s_idx}: {e}")
            continue

    print(f"\n✅ Analyzed {len(sentence_analyses)} sentences")
    print(f"✅ Saved {len(sentence_analyses)} individual sentence files")

    # Document-level analysis
    print("\n📊 Computing document-level metrics...")
    result = {
        "file_name": os.path.basename(file_path),
        "analysis_type": "general_text",
        "sentence_count": len(sentence_analyses),
        "sentences": sentence_analyses,
        "timestamp": datetime.now().isoformat()
    }

    # Document embedding
    try:
        doc_embedding = herbert_mapper.get_herbert_embedding(full_text)
        result['herbert_embedding'] = doc_embedding.tolist()
        result['herbert_magnitude'] = float(np.linalg.norm(doc_embedding))
    except Exception as e:
        print(f"⚠️ Document embedding failed: {e}")

    # Semantic flow between consecutive sentences
    sent_embeddings = [
        np.array(s['herbert_embedding'])
        for s in sentence_analyses
        if 'herbert_embedding' in s
    ]

    if len(sent_embeddings) > 1:
        similarities = []
        for idx in range(len(sent_embeddings) - 1):
            sim = 1 - cosine(sent_embeddings[idx], sent_embeddings[idx + 1])
            similarities.append(float(sim))

        result['sentence_semantic_flow'] = {
            'mean_similarity': float(np.mean(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'std_similarity': float(np.std(similarities)),
            'all_similarities': similarities
        }

        print(f"  Mean sentence-to-sentence similarity: {np.mean(similarities):.3f}")

    # Save full document with all sentences (for reference)
    output_file = os.path.join(analysis_folder, "full_document.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Save HerBERT analysis summary separately
    herbert_analysis = {
        "sentence_count": len(sentence_analyses),
        "sentences_with_embeddings": len(sent_embeddings),
        "document_herbert_magnitude": result.get('herbert_magnitude'),
        "sentence_semantic_flow": result.get('sentence_semantic_flow', {}),
        "timestamp": result['timestamp']
    }

    # Add similarity matrix if we have embeddings
    if len(sent_embeddings) > 1:
        # Create full similarity matrix between all sentences
        n_sents = len(sent_embeddings)
        similarity_matrix = []
        for i in range(n_sents):
            row = []
            for j in range(n_sents):
                if i == j:
                    row.append(1.0)
                else:
                    sim = 1 - cosine(sent_embeddings[i], sent_embeddings[j])
                    row.append(float(sim))
            similarity_matrix.append(row)

        herbert_analysis['similarity_matrix'] = similarity_matrix

        # Find most and least similar sentence pairs
        similar_pairs = []
        for i in range(n_sents):
            for j in range(i + 1, n_sents):
                similar_pairs.append({
                    'sentence_1': i + 1,
                    'sentence_2': j + 1,
                    'similarity': similarity_matrix[i][j]
                })

        similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        herbert_analysis['most_similar_pairs'] = similar_pairs[:10]  # Top 10
        herbert_analysis['least_similar_pairs'] = similar_pairs[-10:]  # Bottom 10
        herbert_analysis['average_similarity'] = float(np.mean([
            similarity_matrix[i][j]
            for i in range(n_sents)
            for j in range(i + 1, n_sents)
        ]))

    herbert_file = os.path.join(analysis_folder, "full_document_herbert_analysis.json")
    with open(herbert_file, 'w', encoding='utf-8') as f:
        json.dump(herbert_analysis, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved full document analysis to: {output_file}")
    print(f"✅ Saved HerBERT analysis to: {herbert_file}")
    print(f"📊 Total sentences: {len(sentence_analyses)}")
    print(f"📊 Sentences with HerBERT embeddings: {len(sent_embeddings)}")

    # Finalize and save HerBERT embeddings to .npz
    embeddings_file = saver.finalize_embeddings()
    if embeddings_file:
        print(f"🤖 Saved {len(saver.embedding_storage.embeddings_cache)} HerBERT embeddings to: {embeddings_file}")

    print("\n🎯 Analysis complete!")


if __name__ == "__main__":
    main()
