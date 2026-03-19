#!/usr/bin/env python3
"""
Evaluate VW predictions.
Computes AUC, PCOC, LogLoss overall and by business_type.

Usage:
    python evaluate.py --predictions pred.txt --data val.vw --output report.csv
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from tabulate import tabulate


def parse_vw_line(line: str):
    """Parse VW format line to extract label and tag (business_type)."""
    parts = line.strip().split('|')[0].split()
    label = int(parts[0])
    label = 1 if label == 1 else 0  # Convert -1 to 0
    
    tag = None
    if len(parts) > 1 and parts[1].startswith("'"):
        tag = parts[1][1:]  # Remove leading '
    
    return label, tag


def load_data(pred_file: str, data_file: str):
    """Load predictions and ground truth."""
    # Load predictions (VW format: "score tag" or just "score")
    predictions = []
    with open(pred_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                predictions.append(float(parts[0]))
    
    # Load labels and tags from VW data file
    labels = []
    tags = []
    with open(data_file, 'r') as f:
        for line in f:
            label, tag = parse_vw_line(line)
            labels.append(label)
            tags.append(tag)
    
    assert len(predictions) == len(labels), \
        f"Mismatch: {len(predictions)} predictions vs {len(labels)} labels"
    
    return np.array(predictions), np.array(labels), tags


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute AUC, PCOC, LogLoss."""
    # AUC
    try:
        if len(np.unique(y_true)) < 2:
            auc = 0.5  # Only one class
        else:
            auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = 0.5  # Only one class
    
    # PCOC: predicted / actual conversion rate
    pred_mean = y_pred.mean()
    actual_mean = y_true.mean()
    pcoc = pred_mean / actual_mean if actual_mean > 0 else 0
    
    # LogLoss
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    if len(np.unique(y_true)) < 2:
        logloss = 0.0  # Only one class
    else:
        logloss = log_loss(y_true, y_pred_clipped)
    
    # Counts
    pos = int(y_true.sum())
    neg = int(len(y_true) - pos)
    ivr = actual_mean
    
    return {
        'AUC': auc,
        'PCOC': pcoc,
        'LogLoss': logloss,
        'Pos': pos,
        'Neg': neg,
        'IVR': ivr
    }


def evaluate(pred_file: str, data_file: str, output_file: str = None, model_name: str = 'VW-FTRL'):
    """Main evaluation function."""
    predictions, labels, tags = load_data(pred_file, data_file)
    
    results = []
    
    # Overall metrics
    overall = compute_metrics(labels, predictions)
    results.append({
        'Model': model_name,
        'Key1': 'Overall',
        'Key2': 'Overall',
        **overall
    })
    
    # By business_type
    if tags[0] is not None:
        df = pd.DataFrame({
            'pred': predictions,
            'label': labels,
            'business_type': tags
        })
        
        for biz_type in df['business_type'].unique():
            mask = df['business_type'] == biz_type
            subset = df[mask]
            metrics = compute_metrics(subset['label'].values, subset['pred'].values)
            results.append({
                'Model': model_name,
                'Key1': 'business_type',
                'Key2': biz_type,
                **metrics
            })
    
    # Sort by Neg count (descending)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Neg', ascending=False)
    
    # Print table
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    
    display_df = results_df[['Model', 'Key1', 'Key2', 'AUC', 'PCOC', 'LogLoss', 'Pos', 'Neg', 'IVR']].copy()
    display_df['AUC'] = display_df['AUC'].round(4)
    display_df['PCOC'] = display_df['PCOC'].round(4)
    display_df['LogLoss'] = display_df['LogLoss'].round(4)
    display_df['IVR'] = display_df['IVR'].round(4)
    
    print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
    
    # Save to CSV
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"\nSaved to {output_file}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Evaluate VW predictions')
    parser.add_argument('--predictions', '-p', required=True, help='Predictions file')
    parser.add_argument('--data', '-d', required=True, help='VW data file with labels')
    parser.add_argument('--output', '-o', default=None, help='Output CSV file')
    parser.add_argument('--model_name', '-m', default='VW-FTRL', help='Model name for output')
    
    args = parser.parse_args()
    evaluate(args.predictions, args.data, args.output, args.model_name)


if __name__ == '__main__':
    main()
