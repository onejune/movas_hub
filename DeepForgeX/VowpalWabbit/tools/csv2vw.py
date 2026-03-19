#!/usr/bin/env python3
"""
CSV to VW format converter (fast version).
Uses \x02 as delimiter (ivr_sample_v7 format).

Usage:
    python csv2vw.py /path/to/csv/*.csv --schema conf/combine_schema --colnames conf/column_name
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Set
import glob


def parse_combine_schema(schema_path: str) -> Tuple[List[str], List[List[str]]]:
    """Parse combine_schema file."""
    single_features = []
    cross_features = []
    
    with open(schema_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '#' in line:
                parts = line.split('#')
                cross_features.append(parts)
            else:
                single_features.append(line)
    
    return single_features, cross_features


def parse_column_names(colnames_path: str) -> Dict[str, int]:
    """Parse column_name file to get column index mapping."""
    col_to_idx = {}
    with open(colnames_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                idx = int(parts[0])
                name = parts[1]
                col_to_idx[name] = idx
    return col_to_idx


def sanitize_value(val: str) -> str:
    """Sanitize feature value for VW format."""
    if not val or val == '-' or val == '-1.0' or val == '-1':
        return "NULL"
    val = val.replace(' ', '_').replace(':', '_').replace('|', '_').replace('\t', '_')
    return val if val else "NULL"


def process_csv_files(csv_pattern: str, schema_path: str, colnames_path: str,
                      label_col: str = 'label', key_col: str = 'business_type',
                      output: str = None, delimiter: str = '\x02'):
    """Process CSV files and convert to VW format."""
    
    # Parse schema
    single_features, cross_features = parse_combine_schema(schema_path)
    
    # Parse column names
    col_to_idx = parse_column_names(colnames_path)
    
    # Get all base features needed
    base_features = set(single_features)
    for cross in cross_features:
        base_features.update(cross)
    
    # Check which features are available
    available = base_features & set(col_to_idx.keys())
    missing = base_features - available
    if missing:
        print(f"Warning: {len(missing)} features not in column_name, skipping", file=sys.stderr)
    
    # Get indices for features we need
    label_idx = col_to_idx.get(label_col)
    key_idx = col_to_idx.get(key_col)
    
    if label_idx is None:
        print(f"Error: label column '{label_col}' not found", file=sys.stderr)
        return
    
    # Filter single and cross features to available ones
    single_features = [f for f in single_features if f in col_to_idx]
    cross_features = [[f for f in cross if f in col_to_idx] for cross in cross_features]
    cross_features = [c for c in cross_features if len(c) > 1]  # Need at least 2 features
    
    # Get indices
    single_indices = [(f, col_to_idx[f]) for f in single_features]
    cross_indices = [[(f, col_to_idx[f]) for f in cross] for cross in cross_features]
    
    # Output
    out = open(output, 'w') if output else sys.stdout
    
    try:
        csv_files = sorted(glob.glob(csv_pattern))
        total_rows = 0
        
        for csv_file in csv_files:
            with open(csv_file, 'r') as f:
                for line in f:
                    cols = line.rstrip('\n').split(delimiter)
                    
                    # Label
                    try:
                        label_val = float(cols[label_idx]) if label_idx < len(cols) else 0
                        label = 1 if int(label_val) == 1 else -1
                    except (ValueError, IndexError):
                        label = -1
                    
                    # Tag (business_type)
                    tag = ""
                    if key_idx is not None and key_idx < len(cols):
                        tag = f"'{cols[key_idx]}"
                    
                    # Single features
                    single_feats = []
                    for feat_name, idx in single_indices:
                        if idx < len(cols):
                            val = sanitize_value(cols[idx])
                            if val != "NULL":
                                single_feats.append(f"{feat_name}={val}")
                    
                    # Cross features
                    cross_feats = []
                    for cross in cross_indices:
                        vals = []
                        valid = True
                        names = []
                        for feat_name, idx in cross:
                            if idx < len(cols):
                                val = sanitize_value(cols[idx])
                                if val == "NULL":
                                    valid = False
                                    break
                                vals.append(val)
                                names.append(feat_name)
                            else:
                                valid = False
                                break
                        
                        if valid and vals:
                            cross_name = "_X_".join(names)
                            cross_val = "_X_".join(vals)
                            cross_feats.append(f"{cross_name}={cross_val}")
                    
                    # Build VW line
                    parts = [str(label)]
                    if tag:
                        parts.append(tag)
                    if single_feats:
                        parts.append("|s " + " ".join(single_feats))
                    if cross_feats:
                        parts.append("|c " + " ".join(cross_feats))
                    
                    out.write(" ".join(parts) + '\n')
                    total_rows += 1
                    
                    if total_rows % 500000 == 0:
                        print(f"Processed {total_rows:,} rows...", file=sys.stderr)
            
            print(f"Finished {csv_file}, total: {total_rows:,}", file=sys.stderr)
        
        print(f"Total: {total_rows:,} rows converted", file=sys.stderr)
    
    finally:
        if output:
            out.close()


def main():
    parser = argparse.ArgumentParser(description='Convert CSV to VW format')
    parser.add_argument('csv_pattern', help='CSV file pattern (e.g., /path/*.csv)')
    parser.add_argument('--schema', required=True, help='Path to combine_schema file')
    parser.add_argument('--colnames', required=True, help='Path to column_name file')
    parser.add_argument('--label', default='label', help='Label column name')
    parser.add_argument('--key', default='business_type', help='Key column for tracking')
    parser.add_argument('--output', '-o', default=None, help='Output file (default: stdout)')
    parser.add_argument('--delimiter', default='\x02', help='CSV delimiter')
    
    args = parser.parse_args()
    
    process_csv_files(
        csv_pattern=args.csv_pattern,
        schema_path=args.schema,
        colnames_path=args.colnames,
        label_col=args.label,
        key_col=args.key,
        output=args.output,
        delimiter=args.delimiter
    )


if __name__ == '__main__':
    main()
