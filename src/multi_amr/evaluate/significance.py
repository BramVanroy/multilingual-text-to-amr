import logging
from itertools import combinations
from multiprocessing import Pool
from os import PathLike
from typing import List

import pandas as pd
from smatchpp import eval_statistics, preprocess, solvers
from tqdm import tqdm, trange

from multi_amr.evaluate.backoff_smatch import BackOffSmatchpp

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

graph_standardizer = preprocess.AMRStandardizer(syntactic_standardization="dereify")
ilp = solvers.ILP()
printer = eval_statistics.ResultPrinter(score_type="micro", do_bootstrap=True, output_format="json")
smatch_metric = BackOffSmatchpp(alignmentsolver=ilp, graph_standardizer=graph_standardizer, printer=printer)


def calculate_smatch_f1(reference_penman_strs: List[str], penman_strs: List[str]):
    score, optimization_status, _ = smatch_metric.score_corpus(reference_penman_strs, penman_strs)
    score = score["main"]
    return score["F1"]["result"]


def calculate_p_value(differences, observed_difference):
    """
    Calculate p-value based on bootstrap differences and observed difference.
    """
    more_extreme_count = sum(1 for diff in differences if abs(diff) >= abs(observed_difference))
    return more_extreme_count / len(differences)


def _bootstrap_difference(df1, df2):
    # Sample data with replacement
    sample_df1 = df1.sample(n=len(df1), replace=True)
    sample_df2 = df2.sample(n=len(df2), replace=True)

    # Calculate F1 scores on the sample
    f1_sample_df1 = calculate_smatch_f1(sample_df1["reference_penman_str"], sample_df1["penman_str"])
    f1_sample_df2 = calculate_smatch_f1(sample_df2["reference_penman_str"], sample_df2["penman_str"])

    return f1_sample_df1 - f1_sample_df2


def bootstrap_differences(df1, df2, num_samples: int = 1000, num_workers: int = 16):
    """
    Calculate difference in F1 scores for two systems using bootstrap sampling.
    """
    differences = []
    if num_workers > 1:
        with Pool(num_workers) as pool:
            futures = [pool.apply_async(_bootstrap_difference, (df1, df2)) for _ in range(num_samples)]

            for future in tqdm(futures, total=num_samples, leave=False, desc="Bootstrapping differences"):
                differences.append(future.get())
    else:
        differences = [
            _bootstrap_difference(df1, df2) for _ in trange(num_samples, leave=False, desc="Bootstrapping differences")
        ]

    return differences


def calculate_significant_differences(
    files: List[str | PathLike], names: List[str], fout: str | PathLike, num_samples: int = 1000, num_workers: int = 16
):
    if len(files) != len(names):
        raise ValueError("The number of 'files' must be correspond with the number of 'names'")

    dfs = [pd.read_csv(f, sep="\t", encoding="utf-8") for f in files]
    real_f1s = [
        calculate_smatch_f1(df["reference_penman_str"], df["penman_str"])
        for df in tqdm(dfs, desc="Calculate real smatch")
    ]
    dfs_names_fs = list(zip(dfs, names, real_f1s))

    results = []
    for df_name_f1, df_name_f2 in tqdm(list(combinations(dfs_names_fs, 2)), desc="Compare systems"):
        df1, fname1, f1_full_sys1 = df_name_f1
        df2, fname2, f1_full_sys2 = df_name_f2
        observed_diff = f1_full_sys1 - f1_full_sys2

        diffs = bootstrap_differences(df1, df2, num_samples=num_samples, num_workers=num_workers)
        p_val = calculate_p_value(diffs, observed_diff)
        comparison = {"system_1": fname1, "system_2": fname2, "pval": p_val, "significant": p_val < 0.05}
        results.append(comparison)
        print(comparison)

    resultsdf = pd.DataFrame(results)

    print(resultsdf)
    resultsdf.to_csv(fout, index=False, encoding="utf-8", sep="\t", float_format="%.6f")


def main():
    import argparse

    cparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cparser.add_argument(
        "--files",
        nargs="+",
        help="Files with system predictions. Must be tab-separated with columns 'penman_str' and 'reference_penman_str'",
    )
    cparser.add_argument("--names", nargs="+", help="System names corresponding with the given files (in order)")
    cparser.add_argument("-o", "--fout", help="Output file to write results to as tab-separated file")
    cparser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples to take to calculate significance",
    )
    cparser.add_argument(
        "-j", "--num_workers", type=int, default=16, help="How many workers to use to calculate resampling in parallel"
    )
    cargs = vars(cparser.parse_args())
    calculate_significant_differences(**cargs)


if __name__ == "__main__":
    main()
