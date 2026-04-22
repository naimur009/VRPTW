import sys
import os
import argparse
import glob
import torch
import pandas as pd
import numpy as np

# Add the directory containing dataCleaning.py to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.join(os.path.dirname(script_dir), "3.training_model")
if training_dir not in sys.path:
    sys.path.insert(0, training_dir)

# Import core functions from dataCleaning.py
try:
    from dataCleaning import (
        safe_read_table,
        clean_columns,
        prepare_node_dataframe,
        build_edge_data_vectorized,
        build_pt_data
    )
except ImportError:
    print(f"Error: Could not import dataCleaning.py from {training_dir}")
    raise

def main():
    parser = argparse.ArgumentParser(description="Preprocess test CSV instances for inference")
    parser.add_argument("--test_dir", type=str, default="dataset", help="Directory containing Solomon-style CSVs")
    parser.add_argument("--output_root", type=str, default="Prepared_data", help="Root for output .pt files")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Resolve test_dir relative to the script's directory if it is a relative path
    test_path = args.test_dir
    if not os.path.isabs(test_path):
        test_path = os.path.join(script_dir, test_path)
    test_path = os.path.abspath(test_path)
    
    output_root = args.output_root
    if not os.path.isabs(output_root):
        output_root = os.path.join(script_dir, output_root)
    output_root = os.path.abspath(output_root)
    
    os.makedirs(output_root, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(test_path, "*.csv")))
    
    if not csv_files:
        print(f"Error: No .csv files found in {test_path}")
        return

    print(f"Found {len(csv_files)} test instances in {test_path}. Processing...")
    
    for csv_file in csv_files:
        inst_name = os.path.splitext(os.path.basename(csv_file))[0]
        output_dir = os.path.join(output_root, inst_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Load and clean nodes
        df = safe_read_table(csv_file)
        df = clean_columns(df)
        node_df, stats = prepare_node_dataframe(df)
        
        # 2. Build edges (unlabeled for inference)
        edge_features_df, edge_index_df, y_label_df, edge_meta = build_edge_data_vectorized(
            node_df, 
            used_edges=set()
        )
        
        # 3. Save .pt data and node features
        pt_data = build_pt_data(node_df, edge_features_df, y_label_df)
        pt_data['y'] = torch.full((pt_data['edge_index'].size(1),), -1.0)
        
        torch.save(pt_data, os.path.join(output_dir, "graph_data.pt"))
        node_df.to_csv(os.path.join(output_dir, "node_features.csv"), index=False)
        print(f"  Processed: {inst_name} -> {output_dir}/graph_data.pt + node_features.csv")

    print(f"\nDone! Processed {len(csv_files)} instances into {output_root}")

if __name__ == "__main__":
    main()
