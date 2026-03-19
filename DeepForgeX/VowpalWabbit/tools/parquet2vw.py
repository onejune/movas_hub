#!/usr/bin/env python3
"""
Parquet to VW format converter.
Streams parquet data to VW format for pipe-based training.

Usage:
    python parquet2vw.py /path/to/parquet/part=2026-02-18 --schema conf/combine_schema
    python parquet2vw.py /path/to/parquet/part=2026-02-18 --schema conf/combine_schema | vw --ftrl ...
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple
import pyarrow.parquet as pq


def parse_combine_schema(schema_path: str) -> Tuple[List[str], List[List[str]]]:
    """
    Parse combine_schema file.
    Returns:
        - single_features: list of single feature names
        - cross_features: list of [feat1, feat2, ...] for cross features
    """
    single_features = []
    cross_features = []
    
    with open(schema_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '#' in line:
                # Cross feature: feat1#feat2#feat3
                parts = line.split('#')
                cross_features.append(parts)
            else:
                single_features.append(line)
    
    return single_features, cross_features


def get_all_base_features(single_features: List[str], cross_features: List[List[str]]) -> Set[str]:
    """Get all base feature names needed from data."""
    features = set(single_features)
    for cross in cross_features:
        features.update(cross)
    return features


def sanitize_value(val) -> str:
    """Sanitize feature value for VW format."""
    if val is None:
        return "NULL"
    val_str = str(val)
    # VW doesn't like spaces, colons, pipes in feature names/values
    val_str = val_str.replace(' ', '_').replace(':', '_').replace('|', '_').replace('\t', '_')
    if not val_str or val_str == '-1.0' or val_str == '-1':
        return "NULL"
    return val_str


def row_to_vw(row: Dict, label_col: str, single_features: List[str], 
              cross_features: List[List[str]], key_col: str = None) -> str:
    """
    Convert a row to VW format.
    
    VW format: label [importance] [tag]|namespace feature1 feature2 ...|namespace2 ...
    
    For categorical features: |ns feat_name=feat_value
    For cross features: we concatenate values as a single feature
    """
    # Label: 1 or -1 for binary classification (VW uses -1 for negative)
    label = row.get(label_col, 0)
    label = 1 if int(float(label)) == 1 else -1
    
    # Optional: add key as tag for tracking
    tag = ""
    if key_col and key_col in row:
        tag = f"'{row[key_col]}"
    
    parts = [str(label)]
    if tag:
        parts.append(tag)
    
    # Single features namespace
    single_feats = []
    for feat in single_features:
        if feat in row:
            val = sanitize_value(row[feat])
            if val != "NULL":
                single_feats.append(f"{feat}={val}")
    
    if single_feats:
        parts.append("|s " + " ".join(single_feats))
    
    # Cross features namespace
    cross_feats = []
    for cross in cross_features:
        vals = []
        valid = True
        for feat in cross:
            if feat in row:
                val = sanitize_value(row[feat])
                if val == "NULL":
                    valid = False
                    break
                vals.append(val)
            else:
                valid = False
                break
        
        if valid and vals:
            cross_name = "_X_".join(cross)
            cross_val = "_X_".join(vals)
            cross_feats.append(f"{cross_name}={cross_val}")
    
    if cross_feats:
        parts.append("|c " + " ".join(cross_feats))
    
    return " ".join(parts)


def convert_parquet_to_vw(parquet_path: str, schema_path: str, 
                          label_col: str = 'label', key_col: str = None,
                          output: str = None, batch_size: int = 10000):
    """
    Convert parquet file(s) to VW format.
    
    Args:
        parquet_path: Path to parquet directory (e.g., part=2026-02-18)
        schema_path: Path to combine_schema file
        label_col: Label column name
        key_col: Optional key column for tracking (e.g., requestid)
        output: Output file path (default: stdout)
        batch_size: Batch size for reading parquet
    """
    import pyarrow.dataset as ds
    
    # Parse schema
    single_features, cross_features = parse_combine_schema(schema_path)
    base_features = get_all_base_features(single_features, cross_features)
    
    # Add label and key columns
    columns_to_read = list(base_features)
    if label_col not in columns_to_read:
        columns_to_read.append(label_col)
    if key_col and key_col not in columns_to_read:
        columns_to_read.append(key_col)
    
    # Output
    out = open(output, 'w') if output else sys.stdout
    
    try:
        # Read parquet using dataset API (handles directories better)
        parquet_path = Path(parquet_path)
        dataset = ds.dataset(parquet_path, format='parquet')
        
        # Filter columns that exist in the dataset
        available_cols = set(dataset.schema.names)
        columns_to_read = [c for c in columns_to_read if c in available_cols]
        
        missing_cols = base_features - available_cols
        if missing_cols:
            print(f"Warning: {len(missing_cols)} features not in data, skipping: {list(missing_cols)[:5]}...", file=sys.stderr)
        
        # Process in batches - vectorized for speed
        total_rows = 0
        for batch in dataset.to_batches(columns=columns_to_read, batch_size=batch_size):
            df = batch.to_pandas()
            
            # Vectorized label conversion
            labels = df[label_col].fillna(0).astype(float).astype(int)
            labels = labels.apply(lambda x: 1 if x == 1 else -1)
            
            # Build VW lines in bulk
            lines = []
            for idx in range(len(df)):
                row = df.iloc[idx]
                label = labels.iloc[idx]
                
                # Tag
                tag = f"'{row[key_col]}" if key_col and key_col in df.columns else ""
                
                # Single features
                single_feats = []
                for feat in single_features:
                    if feat in df.columns:
                        val = sanitize_value(row[feat])
                        if val != "NULL":
                            single_feats.append(f"{feat}={val}")
                
                # Cross features
                cross_feats = []
                for cross in cross_features:
                    vals = []
                    valid = True
                    for feat in cross:
                        if feat in df.columns:
                            val = sanitize_value(row[feat])
                            if val == "NULL":
                                valid = False
                                break
                            vals.append(val)
                        else:
                            valid = False
                            break
                    if valid and vals:
                        cross_name = "_X_".join(cross)
                        cross_val = "_X_".join(vals)
                        cross_feats.append(f"{cross_name}={cross_val}")
                
                # Build line
                parts = [str(label)]
                if tag:
                    parts.append(tag)
                if single_feats:
                    parts.append("|s " + " ".join(single_feats))
                if cross_feats:
                    parts.append("|c " + " ".join(cross_feats))
                
                lines.append(" ".join(parts))
            
            out.write('\n'.join(lines) + '\n')
            total_rows += len(df)
            
            # Progress
            if total_rows % 100000 == 0:
                print(f"Processed {total_rows} rows...", file=sys.stderr)
        
        if output:
            print(f"Converted {total_rows} rows to {output}", file=sys.stderr)
        else:
            print(f"Converted {total_rows} rows", file=sys.stderr)
    
    finally:
        if output:
            out.close()


def main():
    parser = argparse.ArgumentParser(description='Convert Parquet to VW format')
    parser.add_argument('parquet_path', help='Path to parquet directory')
    parser.add_argument('--schema', required=True, help='Path to combine_schema file')
    parser.add_argument('--label', default='label', help='Label column name')
    parser.add_argument('--key', default=None, help='Key column for tracking')
    parser.add_argument('--output', '-o', default=None, help='Output file (default: stdout)')
    parser.add_argument('--batch-size', type=int, default=10000, help='Batch size')
    
    args = parser.parse_args()
    
    convert_parquet_to_vw(
        parquet_path=args.parquet_path,
        schema_path=args.schema,
        label_col=args.label,
        key_col=args.key,
        output=args.output,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
