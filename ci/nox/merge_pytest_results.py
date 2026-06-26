#!/usr/bin/env python3

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Merge multiple pytest JUnit XML result files into a single file.

This script is used in CI to combine session-specific pytest results
(e.g., pytest-results-3.10.xml, pytest-results-3.11.xml) into a single
pytest-results.xml file.

Usage:
    python merge_pytest_results.py [--input-dir DIR] [--output FILE] [--pattern PATTERN]

Examples:
    python merge_pytest_results.py --input-dir results --output combined.xml
"""

import argparse
import logging
import sys
from pathlib import Path

from junitparser import JUnitXml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)


def merge_junit_xml_files(
    input_dir: str = "test-results",
    output_file: str = "test-results/pytest-results.xml",
    pattern: str = "pytest-results-*.xml",
) -> bool:
    """Merge multiple JUnit XML files into a single file.

    Args:
        input_dir: Directory containing the XML files to merge
        output_file: Path to the output merged XML file
        pattern: Glob pattern for matching input files

    Returns:
        True if merge was successful, False otherwise
    """
    input_path = Path(input_dir)
    xml_files = list(input_path.glob(pattern))

    if not xml_files:
        logging.info(f"No files matching '{pattern}' found in {input_dir}, nothing to merge")
        return True

    logging.info(f"Found {len(xml_files)} XML files to merge:")
    for f in sorted(xml_files):
        logging.info(f"  - {f}")

    merged = JUnitXml()
    for xml_file in sorted(xml_files):
        try:
            merged += JUnitXml.fromfile(str(xml_file))
        except Exception as e:
            logging.error(f"Error parsing {xml_file}: {e}")
            return False

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write(str(output_path))

    logging.info(f"Successfully merged into {output_file}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge multiple pytest JUnit XML result files into a single file."
    )
    parser.add_argument(
        "--input-dir",
        default="test-results",
        help="Directory containing XML files to merge (default: test-results)",
    )
    parser.add_argument(
        "--output",
        default="test-results/pytest-results.xml",
        help="Output file path (default: test-results/pytest-results.xml)",
    )
    parser.add_argument(
        "--pattern",
        default="pytest-results-*.xml",
        help="Glob pattern for input files (default: pytest-results-*.xml)",
    )

    args = parser.parse_args()

    success = merge_junit_xml_files(
        input_dir=args.input_dir,
        output_file=args.output,
        pattern=args.pattern,
    )

    return not success


if __name__ == "__main__":
    sys.exit(main())
