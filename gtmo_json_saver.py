#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTMØ JSON Saver - Optimized for Individual MD File Analysis
============================================================
Saves each GTMØ analysis result to a separate JSON file with
sequential naming: gtmoanalysisddmmyyyynoX.json
"""

import json
import gzip
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import hashlib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GTMOOptimizedSaver:
    """Optimized JSON saver for individual GTMØ MD file analyses."""
    
    def __init__(self, output_dir: str = "gtmo_results"):
        """
        Initialize optimized JSON saver.
        
        Args:
            output_dir: Directory for saving results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.daily_counter = self._initialize_daily_counter()
        self.current_date = datetime.now().strftime("%d%m%Y")
        
    def _initialize_daily_counter(self) -> int:
        """
        Initialize counter based on existing files for today.
        
        Returns:
            Next available counter number for today
        """
        today_str = datetime.now().strftime("%d%m%Y")
        pattern = f"gtmoanalysis{today_str}no*.json"
        
        existing_files = list(self.output_dir.glob(pattern))
        if not existing_files:
            return 1
        
        # Extract numbers from existing files
        numbers = []
        for file in existing_files:
            try:
                # Extract number from filename
                name_without_ext = file.stem
                no_part = name_without_ext.split('no')[-1]
                numbers.append(int(no_part))
            except (ValueError, IndexError):
                continue
        
        return max(numbers) + 1 if numbers else 1
    
    def _get_next_filename(self) -> str:
        """
        Generate next sequential filename.
        
        Returns:
            Next filename in format gtmoanalysisddmmyyyynoX.json
        """
        # Check if date changed (new day)
        current_date = datetime.now().strftime("%d%m%Y")
        if current_date != self.current_date:
            self.current_date = current_date
            self.daily_counter = 1
        
        filename = f"gtmoanalysis{self.current_date}no{self.daily_counter}.json"
        self.daily_counter += 1
        return filename
    
    def validate_coordinates(self, coords: Dict) -> bool:
        """
        Validate GTMØ coordinate values.
        
        Args:
            coords: Coordinates dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['determination', 'stability', 'entropy']
        
        for field in required_fields:
            if field not in coords:
                logger.error(f"Missing coordinate field: {field}")
                return False
            
            value = coords[field]
            if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                logger.error(f"Invalid {field}={value}, must be in [0,1]")
                return False
        
        return True
    
    def save_md_analysis(self, 
                        md_file_path: str,
                        text_content: str,
                        coordinates: Dict,
                        additional_metrics: Optional[Dict] = None,
                        compress: bool = False) -> str:
        """
        Save individual MD file analysis result.
        
        Args:
            md_file_path: Path to the analyzed MD file
            text_content: Text content that was analyzed
            coordinates: GTMØ coordinates (determination, stability, entropy)
            additional_metrics: Optional additional metrics
            compress: Whether to gzip the output
            
        Returns:
            Path to saved JSON file
        """
        # Validate coordinates
        if not self.validate_coordinates(coordinates):
            raise ValueError("Invalid coordinates structure")
        
        # Generate filename
        filename = self._get_next_filename()
        if compress:
            filename = filename.replace('.json', '.json.gz')
        
        filepath = self.output_dir / filename
        
        # Prepare analysis result
        result = {
            'version': '2.0',
            'analysis_type': 'GTMØ',
            'timestamp': datetime.now().isoformat(),
            'source_file': {
                'path': str(md_file_path),
                'name': Path(md_file_path).name,
                'extension': Path(md_file_path).suffix,
                'hash': self._calculate_file_hash(md_file_path) if Path(md_file_path).exists() else None
            },
            'content': {
                'text': text_content,
                'length': len(text_content),
                'word_count': len(text_content.split())
            },
            'coordinates': {
                'determination': round(coordinates['determination'], 6),
                'stability': round(coordinates['stability'], 6),
                'entropy': round(coordinates['entropy'], 6)
            },
            'analysis_metadata': {
                'analyzed_at': datetime.now().isoformat(),
                'sequence_number': self.daily_counter - 1,
                'daily_date': self.current_date
            }
        }
        
        # Add additional metrics if provided
        if additional_metrics:
            result['additional_metrics'] = additional_metrics
        
        # Add interpretation
        result['interpretation'] = self._generate_interpretation(coordinates)
        
        # Save file
        if compress:
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved MD analysis to: {filepath}")
        return str(filepath)
    
    def save_batch_md_analyses(self, 
                               analyses: List[Dict],
                               create_index: bool = True) -> List[str]:
        """
        Save multiple MD file analyses as separate JSON files.
        
        Args:
            analyses: List of analysis dictionaries, each containing:
                     - md_file_path: str
                     - text_content: str
                     - coordinates: Dict
                     - additional_metrics: Optional[Dict]
            create_index: Whether to create an index file
            
        Returns:
            List of paths to saved files
        """
        saved_files = []
        index_entries = []
        
        for analysis in analyses:
            try:
                filepath = self.save_md_analysis(
                    md_file_path=analysis['md_file_path'],
                    text_content=analysis['text_content'],
                    coordinates=analysis['coordinates'],
                    additional_metrics=analysis.get('additional_metrics')
                )
                saved_files.append(filepath)
                
                # Prepare index entry
                index_entries.append({
                    'file': filepath,
                    'source': analysis['md_file_path'],
                    'timestamp': datetime.now().isoformat(),
                    'coordinates': analysis['coordinates']
                })
                
            except Exception as e:
                logger.error(f"Failed to save analysis for {analysis.get('md_file_path')}: {e}")
                continue
        
        # Create index file if requested
        if create_index and index_entries:
            index_path = self._create_index_file(index_entries)
            logger.info(f"Created index file: {index_path}")
        
        return saved_files
    
    def _calculate_file_hash(self, filepath: str) -> str:
        """
        Calculate SHA256 hash of file.
        
        Args:
            filepath: Path to file
            
        Returns:
            Hex string of file hash
        """
        sha256_hash = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.warning(f"Could not calculate hash for {filepath}: {e}")
            return ""
    
    def _generate_interpretation(self, coordinates: Dict) -> Dict:
        """
        Generate interpretation of GTMØ coordinates.
        
        Args:
            coordinates: GTMØ coordinates
            
        Returns:
            Interpretation dictionary
        """
        d = coordinates['determination']
        s = coordinates['stability']
        e = coordinates['entropy']
        
        # Classify each dimension
        determination_class = (
            'low' if d < 0.33 else
            'medium' if d < 0.67 else
            'high'
        )
        
        stability_class = (
            'unstable' if s < 0.33 else
            'moderate' if s < 0.67 else
            'stable'
        )
        
        entropy_class = (
            'ordered' if e < 0.33 else
            'mixed' if e < 0.67 else
            'chaotic'
        )
        
        # Overall assessment
        overall_score = (d + s + (1 - e)) / 3
        overall_class = (
            'low_quality' if overall_score < 0.4 else
            'moderate_quality' if overall_score < 0.7 else
            'high_quality'
        )
        
        return {
            'determination': determination_class,
            'stability': stability_class,
            'entropy': entropy_class,
            'overall': overall_class,
            'overall_score': round(overall_score, 4)
        }
    
    def _create_index_file(self, entries: List[Dict]) -> str:
        """
        Create index file for batch processing.
        
        Args:
            entries: List of index entries
            
        Returns:
            Path to index file
        """
        index_filename = f"gtmoanalysis_index_{self.current_date}.json"
        index_path = self.output_dir / index_filename
        
        index_data = {
            'version': '2.0',
            'created_at': datetime.now().isoformat(),
            'total_files': len(entries),
            'date': self.current_date,
            'entries': entries,
            'statistics': self._calculate_batch_statistics(entries)
        }
        
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
        
        return str(index_path)
    
    def _calculate_batch_statistics(self, entries: List[Dict]) -> Dict:
        """Calculate statistics for batch of analyses."""
        if not entries:
            return {}
        
        coords_list = [e['coordinates'] for e in entries]
        
        # Calculate averages
        avg_d = sum(c['determination'] for c in coords_list) / len(coords_list)
        avg_s = sum(c['stability'] for c in coords_list) / len(coords_list)
        avg_e = sum(c['entropy'] for c in coords_list) / len(coords_list)
        
        return {
            'average_coordinates': {
                'determination': round(avg_d, 4),
                'stability': round(avg_s, 4),
                'entropy': round(avg_e, 4)
            },
            'total_analyses': len(entries)
        }
    
    def load_analysis(self, filename: str) -> Dict:
        """
        Load a previously saved analysis.
        
        Args:
            filename: Name or path of the JSON file
            
        Returns:
            Loaded analysis dictionary
        """
        # Handle both filename and full path
        if not Path(filename).is_absolute():
            filepath = self.output_dir / filename
        else:
            filepath = Path(filename)
        
        # Handle compressed files
        if filepath.suffix == '.gz' or str(filepath).endswith('.json.gz'):
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                return json.load(f)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)


def main():
    """Example usage and testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='GTMØ Optimized JSON Saver')
    parser.add_argument('--test', action='store_true', 
                       help='Run with test data')
    parser.add_argument('--output-dir', '-d', default='gtmo_results',
                       help='Output directory')
    parser.add_argument('--compress', '-c', action='store_true',
                       help='Compress output files')
    
    args = parser.parse_args()
    
    # Initialize saver
    saver = GTMOOptimizedSaver(args.output_dir)
    
    if args.test:
        # Test with sample data
        test_analyses = [
            {
                'md_file_path': 'test_file1.md',
                'text_content': 'This is test content for the first markdown file.',
                'coordinates': {
                    'determination': 0.75,
                    'stability': 0.82,
                    'entropy': 0.31
                },
                'additional_metrics': {'complexity': 0.45}
            },
            {
                'md_file_path': 'test_file2.md',
                'text_content': 'Another test with different characteristics.',
                'coordinates': {
                    'determination': 0.55,
                    'stability': 0.68,
                    'entropy': 0.52
                }
            }
        ]
        
        # Save individual analyses
        for analysis in test_analyses:
            filepath = saver.save_md_analysis(
                **analysis,
                compress=args.compress
            )
            print(f"Saved: {filepath}")
        
        # Or save as batch
        # saved_files = saver.save_batch_md_analyses(test_analyses)
        # print(f"Saved {len(saved_files)} files")
    
    else:
        print("Use --test flag to run with sample data")
        print(f"Output directory: {saver.output_dir}")
        print(f"Next file number: {saver.daily_counter}")


if __name__ == "__main__":
    main()