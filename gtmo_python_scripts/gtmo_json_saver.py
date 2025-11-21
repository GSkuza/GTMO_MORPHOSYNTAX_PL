#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTMØ JSON Saver - Optimized for Individual MD File Analysis
============================================================
Saves each GTMØ analysis result to a separate JSON file with
sequential naming: gtmoanalysisddmmyyyynoX.json

Version 1.1: Added HerBERT embedding storage in separate .npz files
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
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HerBERTEmbeddingStorage:
    """
    Efficient storage for HerBERT embeddings in compressed .npz format.

    Stores all embeddings for a document in a single compressed file,
    reducing disk usage by ~85% compared to JSON storage.
    """

    def __init__(self, analysis_folder: Path):
        """
        Initialize embedding storage for an analysis.

        Args:
            analysis_folder: Folder where embeddings will be saved
        """
        self.analysis_folder = Path(analysis_folder)
        self.embeddings_file = self.analysis_folder / "herbert_embeddings.npz"
        self.embeddings_cache = {}  # {key: embedding_array}

    def add_embedding(self, key: str, embedding: np.ndarray, use_float16: bool = True):
        """
        Add an embedding to the cache.

        Args:
            key: Unique identifier (e.g., "article_001", "sentence_1_2")
            embedding: HerBERT embedding (768D numpy array)
            use_float16: Use float16 for 50% size reduction (default: True)
        """
        if use_float16:
            embedding = embedding.astype(np.float16)
        self.embeddings_cache[key] = embedding

    def save_all(self, compress: bool = True):
        """
        Save all cached embeddings to a single .npz file.

        Args:
            compress: Use compression (default: True, ~50% smaller)

        Returns:
            Path to saved embeddings file
        """
        if not self.embeddings_cache:
            logger.info("No embeddings to save")
            return None

        if compress:
            np.savez_compressed(self.embeddings_file, **self.embeddings_cache)
        else:
            np.savez(self.embeddings_file, **self.embeddings_cache)

        size_kb = self.embeddings_file.stat().st_size / 1024
        logger.info(f"Saved {len(self.embeddings_cache)} embeddings to {self.embeddings_file.name} ({size_kb:.1f} KB)")
        return str(self.embeddings_file)

    def load_embedding(self, key: str) -> Optional[np.ndarray]:
        """
        Load a specific embedding from the .npz file.

        Args:
            key: Embedding identifier

        Returns:
            Embedding array or None if not found
        """
        if not self.embeddings_file.exists():
            return None

        with np.load(self.embeddings_file) as data:
            return data.get(key, None)

    def load_all(self) -> Dict[str, np.ndarray]:
        """
        Load all embeddings from the .npz file.

        Returns:
            Dictionary mapping keys to embedding arrays
        """
        if not self.embeddings_file.exists():
            return {}

        with np.load(self.embeddings_file) as data:
            return {key: data[key] for key in data.files}


class NumericMatrixStorage:
    """
    Efficient storage for large numeric matrices in compressed .npz format.

    Stores matrices like entanglement_matrix, density_matrix, etc.
    in a separate file to keep JSON files small and readable.
    """

    def __init__(self, analysis_folder: Path):
        """
        Initialize matrix storage for an analysis.

        Args:
            analysis_folder: Folder where matrices will be saved
        """
        self.analysis_folder = Path(analysis_folder)
        self.matrices_file = self.analysis_folder / "numeric_matrices.npz"
        self.matrices_cache = {}  # {key: matrix_array}

    def add_matrix(self, key: str, matrix: np.ndarray, use_float16: bool = True):
        """
        Add a matrix to the cache.

        Args:
            key: Unique identifier (e.g., "entanglement_sentence_001")
            matrix: Numeric matrix (numpy array)
            use_float16: Use float16 for 50% size reduction (default: True)
        """
        if use_float16 and matrix.dtype in [np.float32, np.float64]:
            matrix = matrix.astype(np.float16)
        self.matrices_cache[key] = matrix
        logger.debug(f"Added matrix '{key}' with shape {matrix.shape}")

    def save_all(self, compress: bool = True):
        """
        Save all cached matrices to a single .npz file.

        Args:
            compress: Use compression (default: True)

        Returns:
            Path to saved matrices file
        """
        if not self.matrices_cache:
            logger.info("No matrices to save")
            return None

        if compress:
            np.savez_compressed(self.matrices_file, **self.matrices_cache)
        else:
            np.savez(self.matrices_file, **self.matrices_cache)

        size_kb = self.matrices_file.stat().st_size / 1024
        logger.info(f"Saved {len(self.matrices_cache)} matrices to {self.matrices_file.name} ({size_kb:.1f} KB)")
        return str(self.matrices_file)

    def load_matrix(self, key: str) -> Optional[np.ndarray]:
        """
        Load a specific matrix from the .npz file.

        Args:
            key: Matrix identifier

        Returns:
            Matrix array or None if not found
        """
        if not self.matrices_file.exists():
            return None

        with np.load(self.matrices_file) as data:
            return data.get(key, None)

    def load_all(self) -> Dict[str, np.ndarray]:
        """
        Load all matrices from the .npz file.

        Returns:
            Dictionary mapping keys to matrix arrays
        """
        if not self.matrices_file.exists():
            return {}

        with np.load(self.matrices_file) as data:
            return {key: data[key] for key in data.files}


class GTMOOptimizedSaver:
    """Optimized JSON saver for individual GTMØ MD file analyses."""

    def __init__(self, output_dir: str = "gtmo_results", save_embeddings: bool = True, save_matrices: bool = True):
        """
        Initialize optimized JSON saver.

        Args:
            output_dir: Directory for saving results
            save_embeddings: Whether to save HerBERT embeddings separately (default: True)
            save_matrices: Whether to save large numeric matrices separately (default: True)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.daily_counter = self._initialize_daily_counter()
        self.current_date = datetime.now().strftime("%d%m%Y")
        self.current_analysis_folder = None  # Track current analysis folder
        self.save_embeddings = save_embeddings
        self.embedding_storage = None  # Will be initialized when analysis folder is created
        self.save_matrices = save_matrices
        self.matrix_storage = None  # Will be initialized when analysis folder is created

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

        # Initialize embedding storage if enabled
        if self.save_embeddings:
            self.embedding_storage = HerBERTEmbeddingStorage(analysis_folder)
            logger.info("HerBERT embedding storage initialized")

        # Initialize matrix storage if enabled
        if self.save_matrices:
            self.matrix_storage = NumericMatrixStorage(analysis_folder)
            logger.info("Numeric matrix storage initialized")

        self.daily_counter += 1

        logger.info(f"Created analysis folder: {analysis_folder}")
        return analysis_folder

    def save_full_document_analysis(self,
                                   source_file: str,
                                   articles: List[str] = None,
                                   article_analyses: List[Dict] = None,
                                   sentences: List[str] = None,
                                   sentence_analyses: List[Dict] = None,
                                   compress: bool = False) -> str:
        """
        Save complete document analysis with all articles or sentences.
        Supports both article-based (legal docs) and sentence-based (general text) analysis.

        Args:
            source_file: Path to source .md file
            articles: List of all articles (for legal documents)
            article_analyses: List of all article analysis results
            sentences: List of all sentences (for general documents)
            sentence_analyses: List of all sentence analysis results
            compress: Whether to gzip the output

        Returns:
            Path to saved JSON file
        """
        if not self.current_analysis_folder:
            raise ValueError("No analysis folder created. Call create_analysis_folder() first.")

        # Determine if we're processing articles or sentences
        if article_analyses is not None:
            analyses = article_analyses
            units = articles if articles else []
            unit_type = "articles"
        elif sentence_analyses is not None:
            analyses = sentence_analyses
            units = sentences if sentences else []
            unit_type = "sentences"
        else:
            raise ValueError("Either article_analyses or sentence_analyses must be provided")

        filename = "full_document.json"
        if compress:
            filename = "full_document.json.gz"

        filepath = self.current_analysis_folder / filename

        # Calculate aggregate metrics
        if analyses:
            d_values = [a['coordinates']['determination'] for a in analyses if 'coordinates' in a]
            s_values = [a['coordinates']['stability'] for a in analyses if 'coordinates' in a]
            e_values = [a['coordinates']['entropy'] for a in analyses if 'coordinates' in a]

            aggregate_coords = {
                'determination': sum(d_values) / len(d_values) if d_values else 0.5,
                'stability': sum(s_values) / len(s_values) if s_values else 0.5,
                'entropy': sum(e_values) / len(e_values) if e_values else 0.5
            }

            # Aggregate ambiguity and depth across units
            ambiguities = []
            depths = []
            for a in analyses:
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
            for a in analyses:
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
                'unit_type': unit_type,
                'total_units': len(units),
                'analyzed_units': len(analyses),
                'total_characters': sum(len(u) for u in units),
                'total_words': sum(len(u.split()) for u in units)
            },
            'aggregate_coordinates': {
                'determination': round(aggregate_coords['determination'], 6),
                'stability': round(aggregate_coords['stability'], 6),
                'entropy': round(aggregate_coords['entropy'], 6)
            },
            'aggregate_metrics': aggregate_metrics,
            'interpretation': self._generate_interpretation(aggregate_coords),
            unit_type: analyses,
            'analysis_metadata': {
                'analyzed_at': timestamp.isoformat(),
                'daily_date': self.current_date,
                'folder': str(self.current_analysis_folder)
            }
        }

        # Extract embeddings and matrices from full document if present
        if self.save_embeddings and self.embedding_storage:
            embedding_count = self._extract_embeddings_recursive(result, "full_document")
            if embedding_count > 0:
                logger.info(f"Extracted {embedding_count} embeddings from full document")

        if False and self.save_matrices and self.matrix_storage:  # TEMP DISABLED
            matrix_count = self._extract_matrices_recursive(result, "full_document")
            if matrix_count > 0:
                logger.info(f"Extracted {matrix_count} matrices from full document")

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
        Each sentence gets its own .npz file for embeddings.

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

        # Extract and save HerBERT embedding BEFORE saving JSON
        # Each sentence gets its OWN .npz file for faster loading
        if self.save_embeddings and 'herbert_embedding' in result:
            if isinstance(result['herbert_embedding'], list):
                embedding = np.array(result['herbert_embedding'])

                # Save to individual .npz file for this sentence
                embedding_filename = f"sentence_{sentence_number:03d}_embedding.npz"
                embedding_filepath = self.current_analysis_folder / embedding_filename
                np.savez_compressed(embedding_filepath, embedding=embedding.astype(np.float16))

                logger.info(f"Saved embedding for sentence {sentence_number} to {embedding_filename} ({embedding.shape})")

                # Replace full embedding with reference
                result['herbert_embedding'] = {
                    "_type": "reference",
                    "_file": embedding_filename,
                    "_key": "embedding",
                    "_shape": list(embedding.shape),
                    "_note": "Full embedding stored in separate .npz file for efficiency"
                }

        # Extract and save numeric matrices BEFORE saving JSON
        # TEMPORARILY DISABLED - causes hangs on some sentences
        if False and self.save_matrices and self.matrix_storage:
            matrix_count = self._extract_matrices_recursive(result, f"sentence_{sentence_number:03d}")
            if matrix_count > 0:
                logger.info(f"Extracted {matrix_count} matrices from sentence {sentence_number}")

        # Save file
        if compress:
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved sentence {sentence_number} analysis to: {filepath}")
        return str(filepath)

    def save_article_analysis(self,
                             result: Dict,
                             article: str,
                             article_number: int,
                             compress: bool = False) -> str:
        """
        Save individual article analysis result (article = complete legal unit with all paragraphs).

        Args:
            result: Complete GTMØ analysis result
            article: Original article text (including all paragraphs)
            article_number: Article number in document
            compress: Whether to gzip the output

        Returns:
            Path to saved JSON file
        """
        if not self.current_analysis_folder:
            raise ValueError("No analysis folder created. Call create_analysis_folder() first.")

        # Create filename with article number
        custom_filename = f"article_{article_number:03d}.json"
        if compress:
            custom_filename = custom_filename.replace('.json', '.json.gz')

        filepath = self.current_analysis_folder / custom_filename

        # Ensure result has proper structure and add article info
        if 'analysis_metadata' not in result:
            result['analysis_metadata'] = {}

        result['analysis_metadata']['article_number'] = article_number
        result['analysis_metadata']['saved_at'] = datetime.now().isoformat()
        result['analysis_metadata']['unit_type'] = 'legal_article'

        # Extract and save HerBERT embeddings recursively BEFORE saving JSON
        if self.save_embeddings and self.embedding_storage:
            embedding_count = self._extract_embeddings_recursive(result, f"article_{article_number:03d}")
            if embedding_count > 0:
                logger.info(f"Extracted {embedding_count} embeddings from article {article_number}")

        # Extract and save numeric matrices recursively BEFORE saving JSON
        if False and self.save_matrices and self.matrix_storage:  # TEMP DISABLED
            matrix_count = self._extract_matrices_recursive(result, f"article_{article_number:03d}")
            if matrix_count > 0:
                logger.info(f"Extracted {matrix_count} matrices from article {article_number}")

        # Save file (with references if data was extracted, or full data if not)
        if compress:
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved article {article_number} analysis to: {filepath}")
        return str(filepath)

    def finalize_embeddings(self):
        """
        Finalize and save all collected HerBERT embeddings to .npz file.

        Call this after all analyses are complete to write embeddings to disk.

        Returns:
            Path to saved embeddings file or None if no embeddings
        """
        if self.save_embeddings and self.embedding_storage:
            return self.embedding_storage.save_all(compress=True)
        return None

    def finalize_matrices(self):
        """
        Finalize and save all collected numeric matrices to .npz file.

        Call this after all analyses are complete to write matrices to disk.

        Returns:
            Path to saved matrices file or None if no matrices
        """
        if False and self.save_matrices and self.matrix_storage:  # TEMP DISABLED
            return self.matrix_storage.save_all(compress=True)
        return None

    def _extract_embeddings_recursive(self, data: Dict, base_key: str, counter: List[int] = None) -> int:
        """
        Recursively extract all HerBERT embeddings from nested data structure.

        Replaces embedding lists with references and stores embeddings in .npz file.

        Args:
            data: Dictionary that may contain herbert_embedding and nested structures
            base_key: Base key for embedding naming (e.g., "article_001")
            counter: List with single int to track total extractions (mutable counter)

        Returns:
            Number of embeddings extracted
        """
        if counter is None:
            counter = [0]

        # Extract embedding at this level if it exists and is a list
        if 'herbert_embedding' in data and isinstance(data['herbert_embedding'], list):
            embedding_array = np.array(data['herbert_embedding'])
            # Create unique key for this embedding
            embedding_key = f"{base_key}_emb{counter[0]}"
            counter[0] += 1

            # Add to storage
            self.embedding_storage.add_embedding(embedding_key, embedding_array)

            # Replace with reference
            data['herbert_embedding'] = {
                "_type": "reference",
                "_file": "herbert_embeddings.npz",
                "_key": embedding_key,
                "_shape": list(embedding_array.shape),
                "_note": "Full embedding stored in separate .npz file for efficiency"
            }

        # Recursively process paragraphs
        if 'paragraphs' in data and isinstance(data['paragraphs'], list):
            for para in data['paragraphs']:
                if isinstance(para, dict):
                    self._extract_embeddings_recursive(para, base_key, counter)

        # Recursively process sentences
        if 'sentences' in data and isinstance(data['sentences'], list):
            for sent in data['sentences']:
                if isinstance(sent, dict):
                    self._extract_embeddings_recursive(sent, base_key, counter)

        return counter[0]

    def _extract_matrices_recursive(self, data: Dict, base_key: str, counter: List[int] = None, depth: int = 0) -> int:
        """
        Recursively extract large numeric matrices from nested data structure.

        Replaces matrix lists with references and stores matrices in .npz file.

        Args:
            data: Dictionary that may contain matrices (e.g., entanglement_matrix)
            base_key: Base key for matrix naming (e.g., "sentence_001")
            counter: List with single int to track total extractions (mutable counter)
            depth: Current recursion depth (to prevent infinite loops)

        Returns:
            Number of matrices extracted
        """
        # Prevent infinite recursion
        MAX_RECURSION_DEPTH = 10
        if depth > MAX_RECURSION_DEPTH:
            logger.warning(f"Matrix extraction: max recursion depth {MAX_RECURSION_DEPTH} exceeded")
            return counter[0] if counter else 0

        if counter is None:
            counter = [0]

        # List of matrix fields to extract
        matrix_fields = ['entanglement_matrix']

        # Extract matrices at this level
        for field_name in matrix_fields:
            if field_name in data:
                # Check if it's inside a nested dict (e.g., data['entanglement']['entanglement_matrix'])
                if isinstance(data[field_name], dict) and 'entanglement_matrix' in data[field_name]:
                    matrix_data = data[field_name]['entanglement_matrix']
                    if isinstance(matrix_data, list) and len(matrix_data) > 0:
                        matrix_array = np.array(matrix_data)
                        # Create unique key for this matrix
                        matrix_key = f"{base_key}_{field_name}"
                        counter[0] += 1

                        # Add to storage
                        self.matrix_storage.add_matrix(matrix_key, matrix_array)

                        # Replace with reference
                        data[field_name]['entanglement_matrix'] = {
                            "_type": "reference",
                            "_file": "numeric_matrices.npz",
                            "_key": matrix_key,
                            "_shape": list(matrix_array.shape),
                            "_note": "Full matrix stored in separate .npz file for efficiency"
                        }
                # Or if it's a direct list
                elif isinstance(data[field_name], list) and len(data[field_name]) > 0:
                    matrix_array = np.array(data[field_name])
                    # Only extract if it's actually a matrix (2D with size > threshold)
                    if matrix_array.ndim >= 2 or (matrix_array.ndim == 1 and len(matrix_array) > 50):
                        matrix_key = f"{base_key}_{field_name}"
                        counter[0] += 1

                        # Add to storage
                        self.matrix_storage.add_matrix(matrix_key, matrix_array)

                        # Replace with reference
                        data[field_name] = {
                            "_type": "reference",
                            "_file": "numeric_matrices.npz",
                            "_key": matrix_key,
                            "_shape": list(matrix_array.shape),
                            "_note": "Full matrix stored in separate .npz file for efficiency"
                        }

        # Check for entanglement dict specifically
        if 'entanglement' in data and isinstance(data['entanglement'], dict):
            entanglement = data['entanglement']
            if 'entanglement_matrix' in entanglement and isinstance(entanglement['entanglement_matrix'], list):
                matrix_data = entanglement['entanglement_matrix']
                if len(matrix_data) > 0:
                    matrix_array = np.array(matrix_data)
                    matrix_key = f"{base_key}_entanglement_matrix"
                    counter[0] += 1

                    # Add to storage
                    self.matrix_storage.add_matrix(matrix_key, matrix_array)
                    logger.info(f"Added matrix '{matrix_key}' with shape {matrix_array.shape}")

                    # Replace with reference
                    entanglement['entanglement_matrix'] = {
                        "_type": "reference",
                        "_file": "numeric_matrices.npz",
                        "_key": matrix_key,
                        "_shape": list(matrix_array.shape),
                        "_note": "Full matrix stored in separate .npz file for efficiency"
                    }

        # Recursively process paragraphs
        if 'paragraphs' in data and isinstance(data['paragraphs'], list):
            for para in data['paragraphs']:
                if isinstance(para, dict):
                    self._extract_matrices_recursive(para, base_key, counter, depth + 1)

        # Recursively process sentences
        if 'sentences' in data and isinstance(data['sentences'], list):
            for sent in data['sentences']:
                if isinstance(sent, dict):
                    self._extract_matrices_recursive(sent, base_key, counter, depth + 1)

        return counter[0]


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
