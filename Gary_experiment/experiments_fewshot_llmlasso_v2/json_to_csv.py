#!/usr/bin/env python3
"""
Extract scores from JSON file and fill into corresponding CSV columns.
CSV版本 - 更簡單、更容易調試
"""

import json
import csv
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import sys
import os

# ===========================
# Configuration
# ===========================

# CSV file paths (Target CSV and Output CSV)
CSV_FILE = "scored_by_ChatGPT_promptC_template.csv"
OUTPUT_FILE = "scored_by_qwen_8b.csv"

# CSV column configuration
FILENAME_COL = 0         # Column index for filenames (0 = Column A)
SCORE_START_COL = 5      # Start column for scores (5 = Column F)
SCORE_END_COL = 14       # End column for scores (14 = Column O)

# ===========================
# Core Functions
# ===========================

def select_json_file():
    """
    Opens a file dialog for the user to select the JSON source file.
    Returns:
        str: Path to the selected file, or None if cancelled.
    """
    # Create a root window and hide it (we only want the dialog)
    root = tk.Tk()
    root.withdraw()

    print("Opening file selection dialog...")
    file_path = filedialog.askopenfilename(
        title="Select JSON Source File",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )

    # Destroy the root window after selection
    root.destroy()

    return file_path if file_path else None

def normalize_filename(filename):
    """
    Normalize filename by removing extension variations.
    """
    if not filename:
        return ""

    # Remove .csv suffix
    name = str(filename).strip().replace('.csv', '')
    return name

def load_json_data(json_path):
    """
    Load data from the specified JSON file.
    """
    print(f"Loading JSON file: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def fill_scores_to_csv(json_data, csv_path, output_path):
    """
    Map scores from JSON data into the CSV file.
    """

    # Check if input CSV exists
    if not os.path.exists(csv_path):
        print(f"Error: Target CSV file not found: {csv_path}")
        return 0, []

    # Load CSV data
    print(f"Loading CSV: {csv_path}")
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) == 0:
        print("Error: CSV file is empty")
        return 0, []

    # Create a dictionary for mapping filenames to scores
    json_scores = {}
    for key, value in json_data.items():
        normalized_key = normalize_filename(key)
        scores = value.get('scores', [])
        # Take only the first 10 scores
        json_scores[normalized_key] = scores[:10] if len(scores) >= 10 else scores

    print(f"\nTotal records in JSON: {len(json_scores)}")
    print(f"Total rows in CSV: {len(rows)}")

    # Iterate through CSV rows (skip header row 0)
    matched_count = 0
    unmatched_files = []

    for row_idx in range(1, len(rows)):
        row = rows[row_idx]

        # Get filename from first column
        if len(row) <= FILENAME_COL:
            continue

        filename = row[FILENAME_COL]

        if not filename:
            continue

        # Normalize filename for matching
        normalized_filename = normalize_filename(filename)

        # Find corresponding scores
        if normalized_filename in json_scores:
            scores = json_scores[normalized_filename]

            # Ensure row has enough columns
            while len(row) < SCORE_END_COL + 1:
                row.append('')

            # Fill scores into columns (SCORE_START_COL to SCORE_END_COL)
            for i, score in enumerate(scores):
                col_idx = SCORE_START_COL + i
                if col_idx <= SCORE_END_COL:
                    row[col_idx] = score

            rows[row_idx] = row
            matched_count += 1
            print(f"✓ Filled: {filename} -> {scores}")
        else:
            unmatched_files.append(filename)
            # Debug: show what we're looking for
            if len(unmatched_files) <= 5:
                print(f"✗ No match: '{filename}' -> normalized: '{normalized_filename}'")

    # Save the result
    print(f"\nSaving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"\n{'='*60}")
    print(f"Process Complete!")
    print(f"{'='*60}")
    print(f"Successfully matched and filled: {matched_count} records")
    print(f"Unmatched: {len(unmatched_files)} records")

    if unmatched_files:
        print(f"\nUnmatched files (First 10):")
        for fname in unmatched_files[:10]:
            normalized = normalize_filename(fname)
            print(f"  - {fname} (normalized: {normalized})")
        if len(unmatched_files) > 10:
            print(f"  ... and {len(unmatched_files) - 10} more")

        # Show sample JSON keys for comparison
        print(f"\nSample JSON keys (First 5):")
        for i, key in enumerate(list(json_scores.keys())[:5]):
            print(f"  - {key}")

    print(f"\nResult saved to: {output_path}")

    return matched_count, unmatched_files

# ===========================
# Execution
# ===========================

if __name__ == "__main__":
    print("="*60)
    print("JSON to CSV Score Filler Tool")
    print("="*60)

    try:
        # Step 1: Select JSON file via GUI
        json_file_path = select_json_file()

        if not json_file_path:
            print("\nOperation cancelled: No file selected.")
            sys.exit(0)

        # Step 2: Load JSON data
        json_data = load_json_data(json_file_path)

        # Step 3: Create output directory if needed
        Path("outputs").mkdir(exist_ok=True)

        # Step 4: Fill scores
        matched, unmatched = fill_scores_to_csv(json_data, CSV_FILE, OUTPUT_FILE)

    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()

    # Keep window open briefly if run via double-click (optional)
    input("\nPress Enter to exit...")
