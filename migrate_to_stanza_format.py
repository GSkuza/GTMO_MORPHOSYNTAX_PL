#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIGRATION SCRIPT: Convert existing GTMØ JSON to Stanza-enhanced format
========================================================================
Converts old JSON format to new enhanced format with Stanza integration
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import hashlib

def calculate_text_hash(text: str) -> str:
    """Calculate SHA-256 hash of text for integrity."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def migrate_json_format(old_json: Dict[str, Any], reanalyze: bool = False) -> Dict[str, Any]:
    """
    Migrate old JSON format to new Stanza-enhanced format.

    Args:
        old_json: Original JSON data
        reanalyze: If True, rerun full Stanza analysis. If False, create placeholder structure.

    Returns:
        New enhanced JSON format
    """

    # Extract text if available
    text = old_json.get('text', old_json.get('input_text', ''))

    # Create new structure
    new_json = {
        "metadata": {
            "analysis_version": "4.0-STANZA",
            "timestamp": datetime.now().isoformat(),
            "input_text": text,
            "text_hash": calculate_text_hash(text) if text else None,
            "processing_time_ms": old_json.get('processing_time_ms', 0),
            "engines_used": {
                "stanza": False,  # Will be True if reanalyzed
                "morfeusz2": old_json.get('morfeusz_available', False),
                "gtmo_quantum": True
            },
            "migration_info": {
                "migrated_from_version": old_json.get('version', 'unknown'),
                "migration_timestamp": datetime.now().isoformat(),
                "reanalyzed": reanalyze
            }
        },

        "gtmo_coordinates": {
            "determination": old_json.get('determination', 0.5),
            "stability": old_json.get('stability', 0.5),
            "entropy": old_json.get('entropy', 0.5),
            "vector": [
                old_json.get('determination', 0.5),
                old_json.get('stability', 0.5),
                old_json.get('entropy', 0.5)
            ],
            "distance_to_singularity": old_json.get('singularity_distance', None),
            "cognitive_distance": old_json.get('cognitive_distance', None),
            "phase_classification": _classify_phase(
                old_json.get('determination', 0.5),
                old_json.get('stability', 0.5),
                old_json.get('entropy', 0.5)
            )
        },

        "stanza_analysis": {
            "dependency_structure": {
                "sentences": [],
                "logical_chains": [],
                "hierarchical_structure": []
            },
            "smoking_guns": [],
            "contradiction_score": 0.0,
            "_note": "Run with --reanalyze to populate with Stanza data"
        },

        "causality_analysis": {
            "causal_chains": [],
            "broken_chains": [],
            "circular_reasoning": [],
            "causal_strength": 0.0,
            "_note": "Run with --reanalyze to populate with Stanza data"
        },

        "temporal_analysis": {
            "timeline": [],
            "inconsistencies": [],
            "paradoxes": [],
            "temporal_graph": {},
            "_note": "Run with --reanalyze to populate with Stanza data"
        },

        "legal_assessment": {
            "quality": "unknown",
            "legal_coherence_score": old_json.get('coherence_score', 0.5),
            "issues": [],
            "recommendations": [],
            "strengths": [],
            "_note": "Run with --reanalyze for detailed assessment"
        },

        "singularity_warning": {
            "active": False,
            "distance": old_json.get('singularity_distance', None),
            "type": None,
            "severity": 0.0,
            "trigger_elements": []
        },

        "visualization_data": {
            "dependency_trees": [],
            "causal_graph": {},
            "temporal_timeline": {},
            "gtmo_phase_space": {
                "current_position": [
                    old_json.get('determination', 0.5),
                    old_json.get('stability', 0.5),
                    old_json.get('entropy', 0.5)
                ],
                "trajectory": old_json.get('trajectory', []),
                "singularity_location": [1.0, 1.0, 0.0],
                "cognitive_center": [0.5, 0.5, 0.5]
            }
        },

        "quantum_metrics": {
            "superposition_index": old_json.get('superposition_index', 0.0),
            "entanglement_score": old_json.get('entanglement', 0.0),
            "collapse_probability": old_json.get('collapse_prob', 0.0),
            "wavefunction": old_json.get('wavefunction', [])
        },

        "polish_linguistic_features": {
            "case_distribution": old_json.get('case_distribution', {
                "nominative": 0, "genitive": 0, "dative": 0, "accusative": 0,
                "instrumental": 0, "locative": 0, "vocative": 0
            }),
            "aspect_analysis": {
                "perfective_count": 0,
                "imperfective_count": 0,
                "aspect_consistency": 0.0
            },
            "formality_level": old_json.get('formality', 'neutral')
        },

        # Preserve any custom fields
        "_original_data": {
            k: v for k, v in old_json.items()
            if k not in ['determination', 'stability', 'entropy', 'text', 'input_text']
        }
    }

    # If reanalyze requested and text available
    if reanalyze and text:
        new_json = _reanalyze_with_stanza(new_json, text)

    return new_json

def _classify_phase(D: float, S: float, E: float) -> str:
    """Classify phase based on coordinates."""
    import numpy as np

    singularity = np.array([1.0, 1.0, 0.0])
    current = np.array([D, S, E])
    distance = np.linalg.norm(current - singularity)

    if distance < 0.2:
        return "singularity_approach"
    elif S < 0.3:
        return "unstable"
    elif S < 0.6:
        return "metastable"
    else:
        return "stable"

def _reanalyze_with_stanza(new_json: Dict, text: str) -> Dict:
    """
    Reanalyze text with Stanza and populate enhanced fields.

    Args:
        new_json: Partially filled new format
        text: Text to analyze

    Returns:
        Fully populated new format
    """
    try:
        from gtmo_morphosyntax import EnhancedGTMOProcessor

        processor = EnhancedGTMOProcessor()
        result = processor.analyze_legal_text(text)

        # Merge results
        new_json['stanza_analysis'] = result.get('modules', {}).get('dependency', {})
        new_json['causality_analysis'] = result.get('modules', {}).get('causality', {})
        new_json['temporal_analysis'] = result.get('modules', {}).get('temporal', {})
        new_json['legal_assessment'] = result.get('overall_assessment', {})
        new_json['singularity_warning'] = result.get('singularity_warning', {})

        # Update metadata
        new_json['metadata']['engines_used']['stanza'] = True
        new_json['metadata']['migration_info']['reanalyzed'] = True

        # Update GTMØ coordinates with new analysis
        if 'gtmo_coordinates' in result:
            new_json['gtmo_coordinates'] = result['gtmo_coordinates']

        print("✓ Stanza reanalysis completed successfully")

    except ImportError:
        print("⚠ Stanza not available - creating placeholder structure only")
        print("  Install Stanza: pip install stanza")
        print("  Then download Polish model: import stanza; stanza.download('pl')")
    except Exception as e:
        print(f"⚠ Error during reanalysis: {e}")
        print("  Creating placeholder structure")

    return new_json

def migrate_file(input_path: Path, output_path: Path = None, reanalyze: bool = False):
    """
    Migrate a JSON file from old to new format.

    Args:
        input_path: Path to old JSON file
        output_path: Path for new JSON file (defaults to input_path with .migrated.json suffix)
        reanalyze: Whether to rerun Stanza analysis
    """
    print(f"\n{'='*60}")
    print(f"Migrating: {input_path.name}")
    print(f"{'='*60}")

    # Read old JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        old_json = json.load(f)

    print(f"✓ Loaded old format ({len(old_json)} fields)")

    # Migrate
    new_json = migrate_json_format(old_json, reanalyze=reanalyze)

    print(f"✓ Migrated to new format ({len(new_json)} top-level fields)")

    # Determine output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}.migrated.json"

    # Write new JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_json, f, ensure_ascii=False, indent=2)

    print(f"✓ Saved to: {output_path}")
    print(f"\nMigration complete!")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Reanalyzed: {'Yes' if reanalyze else 'No (placeholder only)'}")

    if new_json['stanza_analysis'].get('smoking_guns'):
        print(f"\n⚠ Smoking guns detected: {len(new_json['stanza_analysis']['smoking_guns'])}")

    if new_json['singularity_warning'].get('active'):
        print(f"⚠ Singularity warning active!")

    print(f"\nGTMØ Coordinates:")
    coords = new_json['gtmo_coordinates']
    print(f"  D={coords['determination']:.3f}, S={coords['stability']:.3f}, E={coords['entropy']:.3f}")
    print(f"  Phase: {coords['phase_classification']}")

def migrate_directory(dir_path: Path, pattern: str = "*.json", reanalyze: bool = False):
    """
    Migrate all JSON files in a directory.

    Args:
        dir_path: Directory containing JSON files
        pattern: Glob pattern for JSON files
        reanalyze: Whether to rerun Stanza analysis
    """
    json_files = list(dir_path.glob(pattern))

    # Exclude already migrated files
    json_files = [f for f in json_files if '.migrated' not in f.stem]

    if not json_files:
        print(f"No JSON files found in {dir_path} matching pattern '{pattern}'")
        return

    print(f"\nFound {len(json_files)} files to migrate")

    for i, json_file in enumerate(json_files, 1):
        print(f"\n[{i}/{len(json_files)}]")
        try:
            migrate_file(json_file, reanalyze=reanalyze)
        except Exception as e:
            print(f"✗ Error migrating {json_file.name}: {e}")
            continue

def main():
    """Main CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate GTMØ JSON files to Stanza-enhanced format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate single file (placeholder only)
  python migrate_to_stanza_format.py old_result.json

  # Migrate with full Stanza reanalysis
  python migrate_to_stanza_format.py old_result.json --reanalyze

  # Migrate all JSON files in directory
  python migrate_to_stanza_format.py --dir ./results --reanalyze

  # Specify output file
  python migrate_to_stanza_format.py old.json -o new.json
        """
    )

    parser.add_argument('input', nargs='?', type=str,
                       help='Input JSON file to migrate')
    parser.add_argument('-o', '--output', type=str,
                       help='Output JSON file path')
    parser.add_argument('--dir', type=str,
                       help='Migrate all JSON files in directory')
    parser.add_argument('--pattern', type=str, default='*.json',
                       help='Glob pattern for files in directory (default: *.json)')
    parser.add_argument('--reanalyze', action='store_true',
                       help='Rerun full Stanza analysis (slower but complete)')

    args = parser.parse_args()

    # Validate arguments
    if not args.input and not args.dir:
        parser.print_help()
        sys.exit(1)

    # Directory mode
    if args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            print(f"Error: {args.dir} is not a directory")
            sys.exit(1)

        migrate_directory(dir_path, args.pattern, args.reanalyze)

    # Single file mode
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: {args.input} does not exist")
            sys.exit(1)

        output_path = Path(args.output) if args.output else None
        migrate_file(input_path, output_path, args.reanalyze)

if __name__ == "__main__":
    main()
