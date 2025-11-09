#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTMØ File Loader - Production MD Analysis Module
================================================
Loads and analyzes all Markdown files from specified location.
No sampling, full analysis for production use.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import hashlib
import re
import spacy
from gtmo_json_saver import GTMOOptimizedSaver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the GTMØ morphosyntax engine
try:
    from gtmo_morphosyntax import analyze_quantum_with_axioms as gtmo_analyze
except ImportError:
    logger.warning("Could not import gtmo_morphosyntax, using fallback")
    gtmo_analyze = None

nlp = spacy.load("pl_core_news_sm")  # lub inny model polski


class GTMOFileLoader:
    """Production-ready file loader for GTMØ analysis."""
    
    def __init__(self, base_path: str, chunk_size: int = 500):
        """
        Initialize file loader.
        
        Args:
            base_path: Base directory path for MD files
            chunk_size: Size of text chunks for analysis (chars)
        """
        self.base_path = Path(base_path)
        self.chunk_size = chunk_size
        self.processed_files = []
        self.errors = []
        
        if not self.base_path.exists():
            raise ValueError(f"Path does not exist: {base_path}")
    
    def find_md_files(self, recursive: bool = True) -> List[Path]:
        """
        Find all MD files in the specified location.
        """
        if self.base_path.is_file() and self.base_path.suffix.lower() == ".md":
            logger.info(f"Single MD file provided: {self.base_path}")
            return [self.base_path]
        if recursive:
            pattern = "**/*.md"
        else:
            pattern = "*.md"
        md_files = list(self.base_path.glob(pattern))
        logger.info(f"Found {len(md_files)} MD files in {self.base_path}")
        return sorted(md_files)
    
    def read_file_content(self, file_path: Path) -> Optional[str]:
        """
        Read content from MD file with proper encoding handling.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content or None if error
        """
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1250']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                logger.debug(f"Successfully read {file_path} with {encoding}")
                return content
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                self.errors.append({
                    'file': str(file_path),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                return None
        
        logger.error(f"Could not decode {file_path} with any encoding")
        return None
    
    def extract_text_blocks(self, content: str) -> List[Dict]:
        """
        Extract meaningful text blocks from MD content.
        
        Args:
            content: MD file content
            
        Returns:
            List of text blocks with metadata
        """
        blocks = []
        
        # Split by headers
        header_pattern = r'^#+\s+(.+)$'
        sections = re.split(header_pattern, content, flags=re.MULTILINE)
        
        current_header = "Document Start"
        for i, section in enumerate(sections):
            if i % 2 == 1:  # Header
                current_header = section.strip()
            else:  # Content
                # Remove MD formatting
                text = self.clean_markdown(section)
                
                # Split into sentences
                sentences = self.split_sentences(text)
                
                # Group sentences into chunks
                for chunk in self.create_chunks(sentences):
                    if chunk.strip():
                        blocks.append({
                            'header': current_header,
                            'text': chunk,
                            'char_count': len(chunk),
                            'word_count': len(chunk.split())
                        })
        
        return blocks
    
    def clean_markdown(self, text: str) -> str:
        """Remove MD formatting while preserving text."""
        # Remove code blocks
        text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
        text = re.sub(r'`[^`]+`', '', text)
        
        # Remove links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove images
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)
        
        # Remove formatting
        text = re.sub(r'[*_]{1,3}([^*_]+)[*_]{1,3}', r'\1', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Clean up whitespace
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Polish sentence endings
        sentence_endings = r'[.!?]+[\s\n]+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_chunks(self, sentences: List[str]) -> List[str]:
        """
        Create chunks from sentences without losing data.
        
        Args:
            sentences: List of sentences
            
        Returns:
            List of text chunks
        """
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def analyze_file(self, file_path: Path) -> Dict:
        """
        Analyze single MD file completely.
        
        Args:
            file_path: Path to MD file
            
        Returns:
            Analysis results
        """
        logger.info(f"Analyzing file: {file_path}")
        
        # Read file
        content = self.read_file_content(file_path)
        if not content:
            return {
                'file': str(file_path),
                'error': 'Could not read file',
                'timestamp': datetime.now().isoformat()
            }
        
        # Extract blocks
        blocks = self.extract_text_blocks(content)
        
        # Analyze each block
        analyses = []
        for block in blocks:
            try:
                result = gtmo_analyze(block['text'])
                result['metadata'] = block
                analyses.append(result)
            except Exception as e:
                logger.warning(f"Error analyzing block: {e}")
                analyses.append({
                    'text': block['text'][:100] + '...',
                    'error': str(e),
                    'metadata': block
                })
        
        # Calculate file hash
        file_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Aggregate results
        file_result = {
            'file': str(file_path),
            'file_hash': file_hash,
            'file_size': file_path.stat().st_size,
            'timestamp': datetime.now().isoformat(),
            'total_blocks': len(blocks),
            'analyzed_blocks': len([a for a in analyses if 'coordinates' in a]),
            'failed_blocks': len([a for a in analyses if 'error' in a]),
            'analyses': analyses,
            'aggregate_metrics': self.calculate_aggregates(analyses)
        }
        
        self.processed_files.append(file_path)
        return file_result
    
    def calculate_aggregates(self, analyses: List[Dict]) -> Dict:
        """Calculate aggregate metrics from analyses."""
        valid_analyses = [a for a in analyses if 'coordinates' in a]
        
        if not valid_analyses:
            return {}
        
        d_values = [a['coordinates']['determination'] for a in valid_analyses]
        s_values = [a['coordinates']['stability'] for a in valid_analyses]
        e_values = [a['coordinates']['entropy'] for a in valid_analyses]
        
        import numpy as np
        
        return {
            'mean_determination': float(np.mean(d_values)),
            'mean_stability': float(np.mean(s_values)),
            'mean_entropy': float(np.mean(e_values)),
            'std_determination': float(np.std(d_values)),
            'std_stability': float(np.std(s_values)),
            'std_entropy': float(np.std(e_values)),
            'min_determination': float(np.min(d_values)),
            'max_determination': float(np.max(d_values)),
            'min_stability': float(np.min(s_values)),
            'max_stability': float(np.max(s_values)),
            'min_entropy': float(np.min(e_values)),
            'max_entropy': float(np.max(e_values))
        }
    
    def analyze_directory(self, recursive: bool = True,
                         file_filter: Optional[str] = None) -> List[Dict]:
        """
        Analyze all MD files in directory.
        
        Args:
            recursive: Search subdirectories
            file_filter: Optional regex pattern for file filtering
            
        Returns:
            List of analysis results
        """
        # Find files
        md_files = self.find_md_files(recursive)
        
        # Apply filter if provided
        if file_filter:
            pattern = re.compile(file_filter)
            md_files = [f for f in md_files if pattern.search(str(f))]
        
        logger.info(f"Processing {len(md_files)} files")
        
        # Analyze each file
        results = []
        for i, file_path in enumerate(md_files, 1):
            logger.info(f"Processing file {i}/{len(md_files)}: {file_path.name}")
            result = self.analyze_file(file_path)
            results.append(result)
        
        # Summary
        logger.info(f"Analysis complete. Processed: {len(self.processed_files)}, Errors: {len(self.errors)}")
        
        return results


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='GTMØ MD File Analyzer')
    parser.add_argument('path', help='Path to directory with MD files')
    parser.add_argument('--output', '-o', default='gtmo_analysis.json',
                       help='Output JSON file (default: gtmo_analysis.json)')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='Search subdirectories')
    parser.add_argument('--filter', '-f', help='Regex pattern for file filtering')
    parser.add_argument('--chunk-size', '-c', type=int, default=700,
                       help='Chunk size for text analysis (default: 700)')
    
    args = parser.parse_args()
    
    # Initialize loader
    loader = GTMOFileLoader(args.path, chunk_size=args.chunk_size)
    
    # Analyze directory
    results = loader.analyze_directory(
        recursive=args.recursive,
        file_filter=args.filter
    )
    
    # Create results directory if it doesn't exist
    results_dir = Path('gtmo_results')
    results_dir.mkdir(exist_ok=True)

    # Save each result as a separate file
    for i, result in enumerate(results):
        # Generate filename from timestamp and file name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        source_filename = Path(result['file']).stem
        output_filename = f"{timestamp}_{source_filename}_{i:03d}.json"
        output_path = results_dir / output_filename

        # Prepare output data for this result
        output_data = {
            'metadata': {
                'source_path': str(args.path),
                'source_file': result['file'],
                'analysis_timestamp': result['timestamp'],
                'chunk_size': args.chunk_size,
                'recursive': args.recursive,
                'filter': args.filter
            },
            'result': result,
            'errors': [e for e in loader.errors if e.get('file') == result['file']]
        }

        # Save to individual file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved analysis to: {output_path}")
    
    saver = GTMOOptimizedSaver()

    for result in results:
        for analysis in result['analyses']:
            text = analysis['text'] if 'text' in analysis else ""
            doc = nlp(text)
            for sent in doc.sents:
                result_sent = gtmo_analyze(sent.text)
                coordinates = result_sent['coordinates']
                metadata = result_sent.get('metadata', {})
                saver.save_md_analysis(
                    md_file_path=result['file'],
                    text_content=sent.text,
                    coordinates=coordinates,
                    additional_metrics=metadata
                )
    
    print(f"Analysis saved to: {results_dir}/ ({len(results)} files)")
    print(f"Files processed: {len(loader.processed_files)}")
    print(f"Errors encountered: {len(loader.errors)}")


def load_markdown_file(file_path: str) -> List[str]:
    """
    Load and parse a markdown file into legal articles (not sentences).
    For legal documents, each article with all its paragraphs is one unit.

    Args:
        file_path: Path to the markdown file

    Returns:
        List of articles (complete legal units)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Remove markdown formatting
        content = re.sub(r'#+ ', '', content)  # Remove headers
        content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # Remove bold
        content = re.sub(r'\*(.*?)\*', r'\1', content)  # Remove italic
        content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', content)  # Remove links
        content = re.sub(r'`(.*?)`', r'\1', content)  # Remove code

        # Find article boundaries using lookahead to capture full content
        # Matches from "Art. X" to the next "Art. Y" (or end of document)
        # This preserves all paragraphs (§1, §2, §3) within one article
        # Using DOTALL flag and .*? for lazy matching until next article
        article_pattern = r'(?:Art\.|Artykuł|ART\.)\s*\d+\.?\s*.*?(?=(?:Art\.|Artykuł|ART\.)\s*\d+\.|\Z)'

        article_matches = re.findall(article_pattern, content, flags=re.MULTILINE | re.IGNORECASE | re.DOTALL)

        articles = []

        if article_matches:
            for match in article_matches:
                # Clean up excessive whitespace but preserve paragraph structure
                article_text = re.sub(r'\n\s*\n', '\n', match)  # Remove blank lines
                article_text = re.sub(r'[ \t]+', ' ', article_text)  # Normalize spaces
                article_text = article_text.strip()

                # Skip metadata headers (©Kancelaria, Dz. U., etc.)
                if any(skip in article_text[:30] for skip in ['©Kancelaria', 'Dz. U.', 'U S T A W A', 'KSIĘGA', 'TYTUŁ', 'DZIAŁ', 'Rozdział']):
                    continue

                # Filter very short fragments (likely headers or metadata)
                if len(article_text) >= 50:  # Increased threshold for meaningful articles
                    articles.append(article_text)
        else:
            # No articles found - try splitting by paragraphs
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                para_clean = re.sub(r'\s+', ' ', para).strip()
                if len(para_clean) >= 20:
                    articles.append(para_clean)

        logger.info(f"Loaded {len(articles)} articles from {file_path}")
        return articles

    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        return []


if __name__ == "__main__":
    main()
