from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_test_information(
    test_file: Path,
) -> pd.DataFrame:
    """
    Load patient-level clinical information from test.parquet.

    Queue-entry SOFA is taken from the first observed transition
    of each ICU stay.

    Outcome is taken from the terminal transition:
        reward = +100 -> survived
        reward = -100 -> died
    """

    df = pd.read_parquet(test_file)

    required = {
        "stay_id",
        "bin",
        "sofa_total",
        "done",
        "reward",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"test.parquet is missing columns: {sorted(missing)}"
        )

    df["stay_id"] = df["stay_id"].astype(int)

    first_state = (
        df.sort_values(
            [
                "stay_id",
                "bin",
            ]
        )
        .groupby(
            "stay_id",
            as_index=False,
        )
        .first()
    )

    first_state = first_state[
        [
            "stay_id",
            "sofa_total",
        ]
    ].copy()

    first_state.rename(
        columns={
            "sofa_total": "entry_sofa",
        },
        inplace=True,
    )

    terminal = df[
        df["done"] == 1
    ][
        [
            "stay_id",
            "reward",
        ]
    ].copy()

    terminal = (
        terminal.sort_values(
            "stay_id"
        )
        .drop_duplicates(
            subset="stay_id",
            keep="last",
        )
    )

    terminal.rename(
        columns={
            "reward": "terminal_reward",
        },
        inplace=True,
    )

    patient_info = first_state.merge(
        terminal,
        on="stay_id",
        how="left",
        validate="one_to_one",
    )

    patient_info["died"] = np.where(
        patient_info["terminal_reward"] == -100,
        1,
        np.where(
            patient_info["terminal_reward"] == 100,
            0,
            np.nan,
        ),
    )

    patient_info["outcome"] = np.where(
        patient_info["terminal_reward"] == -100,
        "Died",
        np.where(
            patient_info["terminal_reward"] == 100,
            "Survived",
            "Unknown",
        ),
    )

    patient_info["sofa_group"] = pd.cut(
        patient_info["entry_sofa"],
        bins=[
            -np.inf,
            3,
            5,
            np.inf,
        ],
        labels=[
            "SOFA 2-3",
            "SOFA 4-5",
            "SOFA 6+",
        ],
        right=True,
    )

    return patient_info


def load_queue_results(
    result_file: Path,
    policy_name: str,
) -> pd.DataFrame:
    """
    Load patient-level queue results for one policy.
    """

    if not result_file.exists():
        raise FileNotFoundError(
            f"Could not find {policy_name} results: "
            f"{result_file}"
        )

    df = pd.read_parquet(
        result_file
    )

    required = {
        "stay_id",
        "waiting_time_minutes",
        "treatment_start_hours",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{policy_name} queue results are missing "
            f"columns: {sorted(missing)}"
        )

    df = df[
        [
            "stay_id",
            "waiting_time_minutes",
            "treatment_start_hours",
        ]
    ].copy()

    df["stay_id"] = df["stay_id"].astype(int)

    df.rename(
        columns={
            "waiting_time_minutes":
                f"wait_{policy_name}",
            "treatment_start_hours":
                f"start_{policy_name}",
        },
        inplace=True,
    )

    return df


def build_patient_comparison(
    patient_info: pd.DataFrame,
    fifo: pd.DataFrame,
    policies: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Construct one row per patient containing waiting times
    under FIFO and every learned policy.

    Delta waiting time is:

        model waiting time - FIFO waiting time

    Therefore:
        delta < 0 -> treated earlier than FIFO
        delta > 0 -> treated later than FIFO
    """

    df = patient_info.merge(
        fifo,
        on="stay_id",
        how="inner",
        validate="one_to_one",
    )

    for policy_name, policy_df in policies.items():

        df = df.merge(
            policy_df,
            on="stay_id",
            how="inner",
            validate="one_to_one",
        )

        df[
            f"delta_wait_{policy_name}"
        ] = (
            df[f"wait_{policy_name}"]
            - df["wait_FIFO"]
        )

        df[
            f"earlier_than_fifo_{policy_name}"
        ] = (
            df[f"delta_wait_{policy_name}"] < 0
        )

    return df


def calculate_severity_summary(
    df: pd.DataFrame,
    policy_names: list[str],
) -> pd.DataFrame:
    """
    Summarize waiting-time changes relative to FIFO
    within each SOFA severity group.
    """

    rows = []

    sofa_order = [
        "SOFA 2-3",
        "SOFA 4-5",
        "SOFA 6+",
    ]

    for sofa_group in sofa_order:

        group_df = df[
            df["sofa_group"].astype(str)
            == sofa_group
        ]

        if group_df.empty:
            continue

        fifo_wait = group_df[
            "wait_FIFO"
        ]

        rows.append(
            {
                "policy": "FIFO",
                "sofa_group": sofa_group,
                "patients": len(group_df),
                "mean_wait_minutes":
                    fifo_wait.mean(),
                "median_wait_minutes":
                    fifo_wait.median(),
                "p90_wait_minutes":
                    fifo_wait.quantile(0.90),
                "mean_delta_vs_fifo_minutes":
                    0.0,
                "median_delta_vs_fifo_minutes":
                    0.0,
                "percent_treated_earlier_than_fifo":
                    0.0,
            }
        )

        for policy in policy_names:

            waiting = group_df[
                f"wait_{policy}"
            ]

            delta = group_df[
                f"delta_wait_{policy}"
            ]

            earlier = group_df[
                f"earlier_than_fifo_{policy}"
            ]

            rows.append(
                {
                    "policy": policy,
                    "sofa_group": sofa_group,
                    "patients": len(group_df),
                    "mean_wait_minutes":
                        waiting.mean(),
                    "median_wait_minutes":
                        waiting.median(),
                    "p90_wait_minutes":
                        waiting.quantile(0.90),
                    "mean_delta_vs_fifo_minutes":
                        delta.mean(),
                    "median_delta_vs_fifo_minutes":
                        delta.median(),
                    "percent_treated_earlier_than_fifo":
                        100.0 * earlier.mean(),
                }
            )

    return pd.DataFrame(
        rows
    )


def calculate_outcome_summary(
    df: pd.DataFrame,
    policy_names: list[str],
) -> pd.DataFrame:
    """
    Compare waiting times for eventual survivors
    and non-survivors.
    """

    rows = []

    for outcome in [
        "Survived",
        "Died",
    ]:

        group_df = df[
            df["outcome"] == outcome
        ]

        if group_df.empty:
            continue

        fifo_wait = group_df[
            "wait_FIFO"
        ]

        rows.append(
            {
                "policy": "FIFO",
                "outcome": outcome,
                "patients": len(group_df),
                "mean_wait_minutes":
                    fifo_wait.mean(),
                "median_wait_minutes":
                    fifo_wait.median(),
                "p90_wait_minutes":
                    fifo_wait.quantile(0.90),
                "mean_delta_vs_fifo_minutes":
                    0.0,
                "percent_treated_earlier_than_fifo":
                    0.0,
            }
        )

        for policy in policy_names:

            waiting = group_df[
                f"wait_{policy}"
            ]

            delta = group_df[
                f"delta_wait_{policy}"
            ]

            earlier = group_df[
                f"earlier_than_fifo_{policy}"
            ]

            rows.append(
                {
                    "policy": policy,
                    "outcome": outcome,
                    "patients": len(group_df),
                    "mean_wait_minutes":
                        waiting.mean(),
                    "median_wait_minutes":
                        waiting.median(),
                    "p90_wait_minutes":
                        waiting.quantile(0.90),
                    "mean_delta_vs_fifo_minutes":
                        delta.mean(),
                    "percent_treated_earlier_than_fifo":
                        100.0 * earlier.mean(),
                }
            )

    return pd.DataFrame(
        rows
    )


def spearman_from_ranks(
    x: pd.Series,
    y: pd.Series,
) -> float:
    """
    Calculate Spearman correlation without scipy.
    """

    valid = (
        x.notna()
        & y.notna()
    )

    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return float("nan")

    x_rank = x.rank(
        method="average"
    )

    y_rank = y.rank(
        method="average"
    )

    return float(
        x_rank.corr(
            y_rank,
            method="pearson",
        )
    )


def calculate_policy_summary(
    df: pd.DataFrame,
    policy_names: list[str],
) -> pd.DataFrame:
    """
    Calculate overall prioritization metrics.

    Treatment rank is determined from treatment start time.

    A negative SOFA-rank correlation means higher-SOFA
    patients tend to be treated earlier.
    """

    rows = []

    all_policies = [
        "FIFO",
        *policy_names,
    ]

    for policy in all_policies:

        wait_col = (
            f"wait_{policy}"
        )

        start_col = (
            f"start_{policy}"
        )

        treatment_rank = (
            df[start_col]
            .rank(
                method="first",
                ascending=True,
            )
        )

        sofa_rank_correlation = (
            spearman_from_ranks(
                df["entry_sofa"],
                treatment_rank,
            )
        )

        row = {
            "policy": policy,
            "patients": len(df),
            "mean_wait_minutes":
                df[wait_col].mean(),
            "median_wait_minutes":
                df[wait_col].median(),
            "p90_wait_minutes":
                df[wait_col].quantile(0.90),
            "sofa_treatment_rank_spearman":
                sofa_rank_correlation,
        }

        if policy == "FIFO":

            row[
                "mean_delta_vs_fifo_minutes"
            ] = 0.0

            row[
                "percent_treated_earlier_than_fifo"
            ] = 0.0

        else:

            delta = df[
                f"delta_wait_{policy}"
            ]

            earlier = df[
                f"earlier_than_fifo_{policy}"
            ]

            row[
                "mean_delta_vs_fifo_minutes"
            ] = delta.mean()

            row[
                "percent_treated_earlier_than_fifo"
            ] = (
                100.0 * earlier.mean()
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def calculate_high_severity_summary(
    df: pd.DataFrame,
    policy_names: list[str],
    threshold: float,
) -> pd.DataFrame:
    """
    Compact summary for high-severity patients.
    """

    severe = df[
        df["entry_sofa"] >= threshold
    ]

    rows = []

    fifo_wait = severe[
        "wait_FIFO"
    ]

    rows.append(
        {
            "policy": "FIFO",
            "sofa_threshold": threshold,
            "patients": len(severe),
            "mean_wait_minutes":
                fifo_wait.mean(),
            "median_wait_minutes":
                fifo_wait.median(),
            "p90_wait_minutes":
                fifo_wait.quantile(0.90),
            "mean_delta_vs_fifo_minutes":
                0.0,
            "percent_treated_earlier_than_fifo":
                0.0,
        }
    )

    for policy in policy_names:

        waiting = severe[
            f"wait_{policy}"
        ]

        delta = severe[
            f"delta_wait_{policy}"
        ]

        earlier = severe[
            f"earlier_than_fifo_{policy}"
        ]

        rows.append(
            {
                "policy": policy,
                "sofa_threshold": threshold,
                "patients": len(severe),
                "mean_wait_minutes":
                    waiting.mean(),
                "median_wait_minutes":
                    waiting.median(),
                "p90_wait_minutes":
                    waiting.quantile(0.90),
                "mean_delta_vs_fifo_minutes":
                    delta.mean(),
                "percent_treated_earlier_than_fifo":
                    100.0 * earlier.mean(),
            }
        )

    return pd.DataFrame(
        rows
    )


def print_summary(
    severity_summary: pd.DataFrame,
    outcome_summary: pd.DataFrame,
    high_severity_summary: pd.DataFrame,
    policy_summary: pd.DataFrame,
):
    """
    Print summaries to the terminal.
    """

    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        160,
    )

    print()
    print("PATIENT PRIORITIZATION ANALYSIS")

    print()
    print("Overall policy summary:")
    print(
        policy_summary.round(3).to_string(
            index=False
        )
    )

    print()
    print("Waiting time by SOFA group:")
    print(
        severity_summary.round(3).to_string(
            index=False
        )
    )

    print()
    print("High-severity patients:")
    print(
        high_severity_summary.round(3).to_string(
            index=False
        )
    )

    print()
    print("Waiting time by patient outcome:")
    print(
        outcome_summary.round(3).to_string(
            index=False
        )
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze patient prioritization relative to FIFO."
        )
    )

    parser.add_argument(
        "--test_file",
        type=str,
        default="test.parquet",
        help="Common test.parquet file.",
    )

    parser.add_argument(
        "--fifo",
        type=str,
        default="queue_fifo/queue_results.parquet",
        help="FIFO queue results.",
    )

    parser.add_argument(
    "--bcq",
    type=str,
    default="queue_bcq_stable/queue_results.parquet",
    help="BCQ queue results.",
)

    parser.add_argument(
        "--iql",
        type=str,
        default="queue_iql_stable/queue_results.parquet",
        help="IQL queue results.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="patient_prioritization",
        help="Directory for analysis outputs.",
    )

    parser.add_argument(
        "--high_sofa_threshold",
        type=float,
        default=6,
        help=(
            "SOFA threshold used to define "
            "high-severity patients."
        ),
    )

    args = parser.parse_args()

    test_file = Path(
        args.test_file
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        f"Test file: {test_file}"
    )

    patient_info = load_test_information(
        test_file
    )

    print(
        f"Clinical patients: {len(patient_info):,}"
    )

    fifo = load_queue_results(
        Path(args.fifo).resolve(),
        "FIFO",
    )

    bcq = load_queue_results(
        Path(args.bcq).resolve(),
        "BCQ",
    )

    iql = load_queue_results(
        Path(args.iql).resolve(),
        "IQL",
    )

    policies = {
        "BCQ": bcq,
        "IQL": iql,
    }

    policy_names = list(
        policies.keys()
    )

    patient_comparison = build_patient_comparison(
        patient_info=patient_info,
        fifo=fifo,
        policies=policies,
    )

    severity_summary = calculate_severity_summary(
        df=patient_comparison,
        policy_names=policy_names,
    )

    outcome_summary = calculate_outcome_summary(
        df=patient_comparison,
        policy_names=policy_names,
    )

    policy_summary = calculate_policy_summary(
        df=patient_comparison,
        policy_names=policy_names,
    )

    high_severity_summary = (
        calculate_high_severity_summary(
            df=patient_comparison,
            policy_names=policy_names,
            threshold=args.high_sofa_threshold,
        )
    )

    patient_comparison.to_csv(
        output_dir
        / "patient_level_comparison.csv",
        index=False,
    )

    severity_summary.to_csv(
        output_dir
        / "severity_summary.csv",
        index=False,
    )

    outcome_summary.to_csv(
        output_dir
        / "outcome_summary.csv",
        index=False,
    )

    policy_summary.to_csv(
        output_dir
        / "policy_summary.csv",
        index=False,
    )

    high_severity_summary.to_csv(
        output_dir
        / "high_severity_summary.csv",
        index=False,
    )

    print_summary(
        severity_summary=severity_summary,
        outcome_summary=outcome_summary,
        high_severity_summary=high_severity_summary,
        policy_summary=policy_summary,
    )

    print()
    print(
        f"Outputs saved to: {output_dir}"
    )

    print(
        "  patient_level_comparison.csv"
    )

    print(
        "  severity_summary.csv"
    )

    print(
        "  outcome_summary.csv"
    )

    print(
        "  policy_summary.csv"
    )

    print(
        "  high_severity_summary.csv"
    )


if __name__ == "__main__":
    main()
