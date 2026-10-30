#!/usr/bin/env python3
"""Compare two downloaded update_geometry directories using NumPy on CPU."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from EasyLM.update_geometry import compare_bundles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('a', help='First local export bundle directory')
    parser.add_argument('c', help='Second local export bundle directory')
    parser.add_argument('--output-prefix', default='update_geometry_comparison')
    args = parser.parse_args()
    try:
        result, matrices = compare_bundles(args.a, args.c)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f'Comparison rejected: {error}\n')
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    report = dict(a=str(Path(args.a).resolve()), c=str(Path(args.c).resolve()), **result)
    with open(str(prefix) + '.json', 'w') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    with open(str(prefix) + '_per_matrix.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['path', 'shape', 'euclidean_cosine'])
        writer.writeheader()
        for matrix in matrices:
            writer.writerow({**matrix, 'shape': json.dumps(matrix['shape']),
                             'euclidean_cosine': ('undefined' if matrix['euclidean_cosine'] is None
                                                  else matrix['euclidean_cosine'])})
    for key in ('euclidean_cosine', 'gn_cosine', 'a_Gc', 'c_Ga',
                'cross_symmetry_absolute_discrepancy', 'cross_symmetry_relative_discrepancy',
                'rayleigh_a', 'rayleigh_c', 'difference_norm', 'difference_rayleigh'):
        value = result[key]
        print(f'{key}: {"undefined" if value is None else format(value, ".8g")}')
    print(f'Saved {prefix}.json and {prefix}_per_matrix.csv')


if __name__ == '__main__':
    main()
