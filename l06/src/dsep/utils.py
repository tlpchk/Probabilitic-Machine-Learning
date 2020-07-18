"""Utility functions for D-Separation implementation."""
import itertools
import pickle

import numpy as np
import pandas as pd


def check_d_separability_for_all_node_pairs(G, C, d_sep_fn):
    records = []
    
    all_node_pairs = [
        (v1, v2) 
        for v1, v2 in sorted(itertools.permutations(G.nodes(), r=2))
    ]

    for v1, v2 in all_node_pairs:
        records.append({
            'v1': v1, 
            'v2': v2, 
            'd_sep': d_sep_fn(G, v1, v2, C),
        })
        
    return pd.DataFrame.from_records(records, columns=['v1', 'v2', 'd_sep'])


def check(graph, conditioning_set, d_sep_fn, expected_df_path):
    actual_df = check_d_separability_for_all_node_pairs(
        G=graph,
        C=conditioning_set,
        d_sep_fn=d_sep_fn,
    )
    
    with open(expected_df_path, 'rb') as fin:
        expected_df = pickle.load(fin)
    
    num_errors = 0
        
    for (_, exp_row), act_d_sep in zip(expected_df.iterrows(), actual_df.d_sep):
        if exp_row.d_sep != act_d_sep:
            print(
                f'* Error for ({exp_row.v1}, {exp_row.v2})\n'
                f'\tExpected: {exp_row.d_sep}\n'
                f'\tGot: {act_d_sep}'
            )
            num_errors += 1
            
    if num_errors > 0:
        total_rows = expected_df.shape[0]
        error_rate = np.round(num_errors / total_rows * 100, 2)
        print(f'Error rate: {num_errors}/{total_rows} -> {error_rate}%')
    else:
        print('No errors found! Passed')
