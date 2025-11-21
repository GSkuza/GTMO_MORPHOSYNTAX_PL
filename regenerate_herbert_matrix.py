#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerate HerBERT analysis with full similarity matrix for existing analysis folder.

Usage:
    python regenerate_herbert_matrix.py <analysis_folder_path>

Example:
    python regenerate_herbert_matrix.py gtmo_results/analysis_21112025_no1_NEW_ustawa_o_trzezwosci
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from gtmo_json_saver import GTMOOptimizedSaver
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    analysis_folder = Path(sys.argv[1])

    if not analysis_folder.exists():
        logger.error(f"Folder not found: {analysis_folder}")
        sys.exit(1)

    if not analysis_folder.is_dir():
        logger.error(f"Not a directory: {analysis_folder}")
        sys.exit(1)

    # Check if folder contains sentence embeddings
    embedding_files = list(analysis_folder.glob("sentence_*_embedding.npz"))
    if not embedding_files:
        logger.error(f"No sentence embeddings found in {analysis_folder}")
        logger.error("This script works only with analysis folders containing sentence_XXX_embedding.npz files")
        sys.exit(1)

    logger.info(f"Found {len(embedding_files)} sentence embeddings")
    logger.info(f"Regenerating HerBERT analysis with FULL similarity matrix...")

    # Create saver instance and set current folder
    saver = GTMOOptimizedSaver(output_dir=analysis_folder.parent)
    saver.current_analysis_folder = analysis_folder

    # Regenerate with full matrix
    try:
        herbert_file = saver.create_herbert_analysis(create_full_matrix=True)
        if herbert_file:
            logger.info(f"✅ Successfully regenerated: {herbert_file}")
            logger.info("File now contains 'similarity_matrix' and 'average_similarity' fields")
            logger.info("Compatible with herbert_gtmo_analysis.py")
        else:
            logger.error("Failed to create HerBERT analysis")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error during regeneration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
