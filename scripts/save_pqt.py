#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: save_pqt.py
Description:
    This is a very simple script that I use to save dataframe data in parquet format
    which is a highly compressed data format useful for storing/sharing large datasets.

Usage:
    python save_pqt.py

Author: Fraser King (kingfr@umich.edu)
Date: 2025-01-29
"""

import pandas as pd

DATA_PATH = '../data/reg_colab.csv'
OUTPUT_PATH = '../data/data_reg.parquet'
MIN_CLASS_SIZE = 500
SAMPLE_FRAC = 1
RANDOM_STATE = 42

def main():
    # Currently saving the entire dataset, but I have the code below for saving different subsamples
    df = pd.read_csv(DATA_PATH)
    df.to_parquet(
            OUTPUT_PATH,
            compression='snappy', 
            index=False
        )
    
    # dfs_sampled = []
    # grouped = df.groupby('precip_class')
    # for class_value, group_df in grouped:
    #     if class_value == 0:
    #         continue
    #     print(len(group_df))
    #     if len(group_df) < MIN_CLASS_SIZE:
    #         dfs_sampled.append(group_df)
    #     else:
    #         dfs_sampled.append(group_df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE))

    # df_sampled = pd.concat(dfs_sampled, ignore_index=True)

    # df_sampled.to_parquet(
    #     OUTPUT_PATH,
    #     compression='snappy',
    #     index=False
    # )

    # print(f"Total original rows: {len(df):,}")
    # print(f"Sampled rows: {len(df_sampled):,}")
    # print(f"Parquet file written to: {OUTPUT_PATH}")
    
    # class_counts = (
    #     df_sampled.groupby('precip_class')
    #     .size()
    #     .reset_index(name='counts')
    #     .sort_values('counts', ascending=False)
    # )
    # print("\nCounts of rows per class in the subsampled DataFrame:")
    # print(class_counts.to_string(index=False))

if __name__ == "__main__":
    main()
    print("Data saved.")
