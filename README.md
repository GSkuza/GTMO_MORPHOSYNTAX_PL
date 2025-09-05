# GTMO_MORPHOSYNTAX_PL

## Overview

**GTMØ Morphosyntax Engine** is an advanced Polish language analysis toolkit for morphosyntactic and rhetorical evaluation. It supports both classical morphosyntactic analysis and pure rhetorical mode detection based on structural anomalies, without relying on pattern matching. The engine is designed for research, linguistic analysis, and advanced NLP pipelines.

## Features

- **Morphosyntactic Analysis**: Calculates determination, stability, and entropy coordinates for Polish texts.
- **Rhetorical Mode Detection**: Identifies irony, paradox, and literal modes using structural (not keyword-based) criteria.
- **Sentence-Level Processing**: Processes each sentence or text block individually, saving results to separate JSON files.
- **Batch File Loader**: Loads and analyzes `.md` and text files from directories, supporting recursive search.
- **Optimized JSON Saver**: Stores each analysis result in a uniquely named JSON file for easy downstream processing.
- **Integration with spaCy and Morfeusz2**: Uses spaCy for sentence segmentation and Morfeusz2 for Polish morphological analysis.
- **Configurable and Extensible**: Modular codebase allows easy adaptation to new analysis types or languages.

## Project Structure

- `gtmo_morphosyntax.py` – Core morphosyntactic analysis functions.
- `gtmo_pure_rhetoric.py` – Pure rhetorical mode detection (irony, paradox, literal).
- `gtmo_extended.py` – Extended analysis (quantum, temporal, etc.).
- `gtmo_json_saver.py` – Saves analysis results to individual JSON files.
- `gtmo_file_loader.py` – Loads files, splits into sentences/blocks, runs analysis, saves results.
- `gtmo_results/` – Output directory for JSON analysis results.
- `requirements.txt` – Python dependencies.
- `test_basic.py` – Example/test script.

## Usage

1. **Install dependencies**  
   Activate your virtual environment and run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run analysis**  
   To analyze a file, use the following command:
   ```bash
   python gtmo_morphosyntax.py path/to/your/file.md
   ```
   This will generate a JSON file with the analysis results in the `gtmo_results/` directory.

3. **Explore results**  
   Each analysis result is saved as a separate JSON file. You can explore these files for detailed insights into the morphosyntactic and rhetorical features of your text.

## Examples

- To analyze a directory of Markdown files recursively:
  ```bash
  python gtmo_file_loader.py path/to/your/directory
  ```

- To test the installation and see example outputs:
  ```bash
  python test_basic.py
  ```

## Notes

- Ensure that you have the latest versions of spaCy and Morfeusz2 installed, as the engine relies on these for morphological analysis and sentence segmentation.
- The engine is optimized for Polish language texts. While it may work with other languages, results are not guaranteed.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by the need for advanced linguistic analysis tools for the Polish language.
- Developed as part of ongoing research in computational linguistics and natural language processing.