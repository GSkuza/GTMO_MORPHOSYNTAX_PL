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
import statistics
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
        self.current_analysis_folder = None  # Track current analysis folder
        
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
    
    def create_analysis_folder(self, source_file_name: str = None) -> Path:
        """
        Create a new subfolder for current analysis.

        Args:
            source_file_name: Optional source file name to include in folder name

        Returns:
            Path to created folder
        """
        # Check if date changed (new day)
        current_date = datetime.now().strftime("%d%m%Y")
        if current_date != self.current_date:
            self.current_date = current_date
            self.daily_counter = 1

        # Create folder name
        if source_file_name:
            # Remove extension and clean filename
            clean_name = Path(source_file_name).stem
            clean_name = clean_name.replace(' ', '_')
            folder_name = f"analysis_{self.current_date}_no{self.daily_counter}_{clean_name}"
        else:
            folder_name = f"analysis_{self.current_date}_no{self.daily_counter}"

        # Create folder
        analysis_folder = self.output_dir / folder_name
        analysis_folder.mkdir(parents=True, exist_ok=True)

        self.current_analysis_folder = analysis_folder
        self.daily_counter += 1

        logger.info(f"Created analysis folder: {analysis_folder}")
        return analysis_folder

    def save_full_document_analysis(self,
                                   source_file: str,
                                   sentences: List[str],
                                   sentence_analyses: List[Dict],
                                   compress: bool = False) -> str:
        """
        Save complete document analysis with all sentences.

        Args:
            source_file: Path to source .md file
            sentences: List of all sentences
            sentence_analyses: List of all sentence analysis results
            compress: Whether to gzip the output

        Returns:
            Path to saved JSON file
        """
        if not self.current_analysis_folder:
            raise ValueError("No analysis folder created. Call create_analysis_folder() first.")

        filename = "full_document.json"
        if compress:
            filename = "full_document.json.gz"

        filepath = self.current_analysis_folder / filename

        # Calculate aggregate metrics
        if sentence_analyses:
            d_values = [a['coordinates']['determination'] for a in sentence_analyses if 'coordinates' in a]
            s_values = [a['coordinates']['stability'] for a in sentence_analyses if 'coordinates' in a]
            e_values = [a['coordinates']['entropy'] for a in sentence_analyses if 'coordinates' in a]

            aggregate_coords = {
                'determination': sum(d_values) / len(d_values) if d_values else 0.5,
                'stability': sum(s_values) / len(s_values) if s_values else 0.5,
                'entropy': sum(e_values) / len(e_values) if e_values else 0.5
            }

            # Aggregate ambiguity and depth across sentences
            ambiguities = []
            depths = []
            for a in sentence_analyses:
                # Prefer top-level keys if present, fallback to nested metrics
                amb = a.get('ambiguity')
                if amb is None:
                    amb = a.get('additional_metrics', {}).get('ambiguity')
                if amb is not None:
                    ambiguities.append(float(amb))

                dep = a.get('depth')
                if dep is None:
                    dep = a.get('depth_metrics', {}).get('max_depth')
                if dep is not None:
                    depths.append(int(dep))

            # Aggregate geometric factors (with fallback to nested constitutional metrics)
            balances = []
            tensions = []
            for a in sentence_analyses:
                bal = a.get('geometric_balance')
                if bal is None:
                    bal = (
                        a.get('constitutional_metrics', {})
                         .get('definiteness', {})
                         .get('components', {})
                         .get('geometric_balance')
                    )
                if bal is not None:
                    try:
                        balances.append(float(bal))
                    except Exception:
                        pass

                ten = a.get('geometric_tension')
                if ten is None:
                    ten = (
                        a.get('constitutional_metrics', {})
                         .get('indefiniteness', {})
                         .get('components', {})
                         .get('geometric_tension')
                    )
                if ten is not None:
                    try:
                        tensions.append(float(ten))
                    except Exception:
                        pass

            # Coordinates (D,S,E) stats across sentences
            # Stddev for coordinates
            d_std = round(statistics.stdev(d_values), 6) if len(d_values) > 1 else None
            s_std = round(statistics.stdev(s_values), 6) if len(s_values) > 1 else None
            e_std = round(statistics.stdev(e_values), 6) if len(e_values) > 1 else None

            coord_stats = {
                'determination': {
                    'average': round(sum(d_values) / len(d_values), 6) if d_values else None,
                    'min': round(min(d_values), 6) if d_values else None,
                    'max': round(max(d_values), 6) if d_values else None,
                    'stddev': d_std,
                    'sample_count': len(d_values)
                },
                'stability': {
                    'average': round(sum(s_values) / len(s_values), 6) if s_values else None,
                    'min': round(min(s_values), 6) if s_values else None,
                    'max': round(max(s_values), 6) if s_values else None,
                    'stddev': s_std,
                    'sample_count': len(s_values)
                },
                'entropy': {
                    'average': round(sum(e_values) / len(e_values), 6) if e_values else None,
                    'min': round(min(e_values), 6) if e_values else None,
                    'max': round(max(e_values), 6) if e_values else None,
                    'stddev': e_std,
                    'sample_count': len(e_values)
                }
            }

            # Precompute standard deviations (None if <2 samples)
            amb_std = round(statistics.stdev(ambiguities), 4) if len(ambiguities) > 1 else None
            depth_std = round(statistics.stdev(depths), 4) if len(depths) > 1 else None
            bal_std = round(statistics.stdev(balances), 6) if len(balances) > 1 else None
            ten_std = round(statistics.stdev(tensions), 6) if len(tensions) > 1 else None

            aggregate_metrics = {
                # Ambiguity stats
                'average_ambiguity': round(sum(ambiguities) / len(ambiguities), 4) if ambiguities else None,
                'min_ambiguity': round(min(ambiguities), 4) if ambiguities else None,
                'max_ambiguity': round(max(ambiguities), 4) if ambiguities else None,
                'std_ambiguity': amb_std,
                'sample_count_ambiguity': len(ambiguities),

                # Depth stats (based on per-sentence max_depth)
                'average_max_depth': round(sum(depths) / len(depths), 4) if depths else None,
                'min_max_depth': int(min(depths)) if depths else None,
                'max_max_depth': int(max(depths)) if depths else None,
                'std_max_depth': depth_std,
                'sample_count_max_depth': len(depths),

                # Geometric balance/tension stats
                'average_geometric_balance': round(sum(balances) / len(balances), 6) if balances else None,
                'min_geometric_balance': round(min(balances), 6) if balances else None,
                'max_geometric_balance': round(max(balances), 6) if balances else None,
                'std_geometric_balance': bal_std,
                'sample_count_geometric_balance': len(balances),

                'average_geometric_tension': round(sum(tensions) / len(tensions), 6) if tensions else None,
                'min_geometric_tension': round(min(tensions), 6) if tensions else None,
                'max_geometric_tension': round(max(tensions), 6) if tensions else None,
                'std_geometric_tension': ten_std,
                'sample_count_geometric_tension': len(tensions),

                # Coordinates stats
                'coordinates_stats': coord_stats
            }
        else:
            aggregate_coords = {'determination': 0.5, 'stability': 0.5, 'entropy': 0.5}
            aggregate_metrics = {'average_ambiguity': None, 'average_max_depth': None, 'max_depth': None}

        # Prepare full document result
        timestamp = datetime.now()
        result = {
            'version': '2.0',
            'analysis_type': 'GTMØ_FULL_DOCUMENT',
            'timestamp': timestamp.isoformat(),
            'source_file': {
                'path': str(source_file),
                'name': Path(source_file).name,
                'extension': Path(source_file).suffix,
                'hash': self._calculate_file_hash(source_file) if Path(source_file).exists() else None
            },
            'document_metadata': {
                'total_sentences': len(sentences),
                'analyzed_sentences': len(sentence_analyses),
                'total_characters': sum(len(s) for s in sentences),
                'total_words': sum(len(s.split()) for s in sentences)
            },
            'aggregate_coordinates': {
                'determination': round(aggregate_coords['determination'], 6),
                'stability': round(aggregate_coords['stability'], 6),
                'entropy': round(aggregate_coords['entropy'], 6)
            },
            'aggregate_metrics': aggregate_metrics,
            'interpretation': self._generate_interpretation(aggregate_coords),
            'sentences': sentence_analyses,
            'analysis_metadata': {
                'analyzed_at': timestamp.isoformat(),
                'daily_date': self.current_date,
                'folder': str(self.current_analysis_folder)
            }
        }

        # Save file
        if compress:
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved full document analysis to: {filepath}")
        return str(filepath)

    def save_sentence_analysis(self,
                             result: Dict,
                             sentence: str,
                             sentence_number: int,
                             compress: bool = False) -> str:
        """
        Save individual sentence analysis result with optimized format.

        Args:
            result: Complete GTMØ analysis result
            sentence: Original sentence text
            sentence_number: Sentence number in document
            compress: Whether to gzip the output

        Returns:
            Path to saved JSON file
        """
        if not self.current_analysis_folder:
            raise ValueError("No analysis folder created. Call create_analysis_folder() first.")

        # Create filename with sentence number
        custom_filename = f"sentence_{sentence_number:03d}.json"
        if compress:
            custom_filename = custom_filename.replace('.json', '.json.gz')

        filepath = self.current_analysis_folder / custom_filename

        # Ensure result has proper structure and add sentence info
        if 'analysis_metadata' not in result:
            result['analysis_metadata'] = {}

        result['analysis_metadata']['sentence_number'] = sentence_number
        result['analysis_metadata']['saved_at'] = datetime.now().isoformat()

        # Save file
        if compress:
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved sentence {sentence_number} analysis to: {filepath}")
        return str(filepath)


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
