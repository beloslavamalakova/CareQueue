"""
A patient queue simulation file that is set up to be Model-Agnostic.
Meaning that this simulator does not assume DDQN, IQL, BCQ, BC, etc.

The workflow is:

    1. Run the model-specific script:
        Input:  test.parquet
        Output: [model_specific]_scores.parquet

       The script loads the trained model and generates a
       priority score for each patient (Q-Value for RL models).

    2. Run this queue simulation using the generated score file.
       The queue uses these scores to prioritize patients.
       Note that this file also generates metrics. Please adapt them 
       as seems logical.

This separation keeps the queue simulation model-agnostic, allowing
the same simulation to be used with DDQN, IQL, BCQ, BC, etc.

For the current proof of concept, a random scoring mode is also
included so the queue can be tested without a trained model.

Example using random scores:

    python queue_simulation.py \
        --test_file test.parquet \
        --score_mode random \
        --output_dir queue_random

Example using DDQN scores:

    python queue_simulation.py \
        --test_file test.parquet \
        --score_mode file \
        --score_file ddqn_scores.parquet \
        --output_dir queue_ddqn
"""

from __future__ import annotations

import argparse
import heapq
import math
import random
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

@dataclass
class Patient:

    patient_id: int
    stay_id: int

    arrival_time: float

    priority_score: float

    service_time: float

    sofa_total: float = 0.0

    reward: float = 0.0


@dataclass
class QueueResult:
    """
    Queuing experience results for a patient
    """

    patient_id: int
    stay_id: int

    arrival_time: float
    treatment_start: float
    treatment_end: float

    waiting_time: float
    time_in_system: float

    priority_score: float

    sofa_total: float
    reward: float


# Score Generation Functions
def generate_random_scores(
    patient_ids: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    """
    Note, current setup: Higher score = higher priority
    """

    rng = np.random.default_rng(seed)

    scores = rng.random(len(patient_ids))

    return pd.DataFrame(
        {
            "stay_id": patient_ids,
            "priority_score": scores,
        }
    )


def load_score_file(
    score_file: Path,
) -> pd.DataFrame:
    """
    Load externally generated model scores.

    Required columns:

        stay_id
        priority_score
    """

    scores = pd.read_parquet(score_file)

    required = {
        "stay_id",
        "priority_score",
    }

    missing = required - set(scores.columns)

    if missing:
        raise ValueError(
            f"Score file is missing columns: {sorted(missing)}"
        )

    scores = scores[
        [
            "stay_id",
            "priority_score",
        ]
    ].copy()

    scores["stay_id"] = scores["stay_id"].astype(int)
    scores["priority_score"] = scores["priority_score"].astype(float)

    if scores["priority_score"].isna().any():
        raise ValueError(
            "Score file contains NaN priority scores."
        )

    return scores


# Load the test set of patients
def load_test_patients(
    test_file: Path,
) -> pd.DataFrame:
    """
    Convert the transition-level test parquet into a
    stay-level population for the queue. This is done 
    by ensuring each stay is represented only once. The 
    first state of each ICU stay is used as the 
    patient's queue-entry state.
    """

    df = pd.read_parquet(test_file)

    required = {
        "stay_id",
        "bin",
        "sofa_total",
        "reward",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Test parquet is missing columns: {sorted(missing)}"
        )

    # First observed state of each ICU stay
    patients = (
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

    # Keep only required columns
    patients = patients[
        [
            "stay_id",
            "bin",
            "sofa_total",
            "reward",
        ]
    ].copy()

    patients.rename(
        columns={
            "bin": "arrival_bin",
        },
        inplace=True,
    )

    return patients


# Simulating arrival process
def generate_poisson_arrivals(
    n_patients: int,
    arrival_rate: float,
    seed: int,
) -> np.ndarray:
    """
    Generate patient arrival times using an exponential
    inter-arrival distribution, i.e. a homogeneous Poisson 
    process. Note the total no. of patients arriving in
    the day is simulated elsewhere.

    Input: arrival_rate - Expected arrivals per hour.
    Returns: Arrival times in hours of all patients in the day
    """

    if arrival_rate <= 0:
        raise ValueError(
            "arrival_rate must be > 0."
        )

    rng = np.random.default_rng(seed)

    inter_arrivals = rng.exponential(
        scale=1.0 / arrival_rate,
        size=n_patients,
    )

    arrival_times = np.cumsum(
        inter_arrivals
    )

    return arrival_times


# Simulating doctor service / queue leaving times
def generate_service_times(
    n_patients: int,
    mean_service_hours: float,
    seed: int,
) -> np.ndarray:
    """
    Same logic as arrivals.

    Input: mean_service_hours - Mean treatment time in hours, 
    i.e. how often a doctor frees up.

    """

    if mean_service_hours <= 0:
        raise ValueError(
            "mean_service_hours must be > 0."
        )

    rng = np.random.default_rng(seed)

    return rng.exponential(
        scale=mean_service_hours,
        size=n_patients,
    )

class PatientQueue:
    """
    Priority queue.

    Higher priority_score = treated first.

    Tie-breaking: Earlier arrival wins.
    """

    def __init__(self):
        self._heap = []
        self._counter = 0

    def add(
        self,
        patient: Patient,
    ):
        """
        Add patient to queue.
        """

        heapq.heappush(
            self._heap,
            (
                -patient.priority_score,
                patient.arrival_time,
                self._counter,
                patient,
            ),
        )

        self._counter += 1

    def pop(
    self,
    current_time: float,
    alpha: float = 0.001,
    ) -> Patient:
        """
        Remove the patient with the highest effective priority,
        here we are assigning a weight on waiting time to our decision making
        
            Effective priority: model score + alpha * waiting time

        Note: Waiting time is measured in hours

        """

        if not self._heap:
            raise IndexError(
                "Cannot pop from empty queue."
            )

        best_index = 0

        first_patient = self._heap[0][3]

        first_waiting_time = max(
            0.0,
            current_time
            - first_patient.arrival_time,
        )

        best_priority = (
            first_patient.priority_score
            + alpha * first_waiting_time
        )

        for i in range(1, len(self._heap)):

            patient = self._heap[i][3]

            waiting_time = max(
                0.0,
                current_time
                - patient.arrival_time,
            )

            effective_priority = (
                patient.priority_score
                + alpha * waiting_time
            )

            if effective_priority > best_priority:

                best_priority = effective_priority
                best_index = i

            elif (
                effective_priority == best_priority
                and patient.arrival_time
                < self._heap[best_index][3].arrival_time
            ):

                best_index = i

        selected_item = self._heap.pop(
            best_index
        )

        heapq.heapify(
            self._heap
        )

        return selected_item[3]

    def empty(self) -> bool:
        return len(self._heap) == 0

    def __len__(self):
        return len(self._heap)

    # Function to provide us a view of the queue while running
    def snapshot(
        self,
        top_k: int = 10,
        current_time: float = 0.0,
        alpha: float = 0.001,
    ) -> list[Patient]:
        """
        Return the current top-k patients in priority order.
        """

        if not self._heap:
            return []

        patients = [
            item[3]
            for item in self._heap
        ]

        patients.sort(
            key=lambda p: (
                -(
                    p.priority_score
                    + alpha * max(
                        0.0,
                        current_time - p.arrival_time,
                    )
                ),
                p.arrival_time,
            )
        )

        return patients[:top_k]

# Visualization of the queue snapshot
def print_queue_snapshot(
    queue: PatientQueue,
    current_time: float,
    top_k: int = 10,
    event: str = "",
    event_patient: Patient | None = None,
    alpha: float = 0.001,
):
    """
    Print the current top-k patients in the queue.
    Note we print the queue for every trigerring event,
    i.e. a patient entering or leaving the queue

    current_time:
        Simulation time in hours.

    event:
        Event that triggered the snapshot.

    event_patient:
        Patient involved in the event.
    """

    print()
    print("=" * 75)

    event_description = event

    if event_patient is not None:
        event_description += (
            f" | Patient ID: {event_patient.patient_id}"
        )

    print(
        f"QUEUE SNAPSHOT | "
        f"Simulation time: "
        f"{current_time * 3600:.1f} seconds"
    )

    if event_description:
        print(
            f"Event: {event_description}"
        )

    print("=" * 75)

    if queue.empty():
        print("Queue is empty.")
        print("=" * 75)
        return

    snapshot = queue.snapshot(
        top_k=top_k,
        current_time=current_time,
        alpha=alpha,
    )

    print(
        f"{'Rank':<6}"
        f"{'Patient ID':<15}"
        f"{'Priority':<12}"
        f"{'Effective Priority':<20}"
        f"{'Arrival (s)':<15}"
        f"{'Waiting (s)':<15}"
    )

    print("-" * 75)

    for rank, patient in enumerate(
        snapshot,
        start=1,
    ):

        waiting_seconds = max(
            0.0,
            (
                current_time
                - patient.arrival_time
            ) * 3600.0,
        )

        waiting_hours = max(
            0.0,
            current_time
            - patient.arrival_time,
        )

        effective_priority = (
            patient.priority_score
            + alpha * waiting_hours
        )

        arrival_seconds = (
            patient.arrival_time * 3600.0
        )

        print(
            f"{rank:<6}"
            f"{patient.patient_id:<15}"
            f"{patient.priority_score:<15.4f}"
            f"{effective_priority:<20.4f}"
            f"{arrival_seconds:<15.1f}"
            f"{waiting_seconds:<15.1f}"
        )

    print("=" * 75)

# Queue Simulation

def simulate_queue(
    patients: list[Patient],
    visualize: bool = False,
    top_k: int = 10,
    alpha: float = 0.001,
) -> list[QueueResult]:

    patients = sorted(
        patients,
        key=lambda p: p.arrival_time,
    )

    queue = PatientQueue()

    results = []

    patient_index = 0

    current_time = 0.0

    current_patient = None

    treatment_start = None

    treatment_end = math.inf

    while (
        patient_index < len(patients)
        or not queue.empty()
        or current_patient is not None
    ):

        # Determine next arrival time.

        if patient_index < len(patients):

            next_arrival_time = patients[
                patient_index
            ].arrival_time

        else:

            next_arrival_time = math.inf

        # Determine which event happens next

        if (
            next_arrival_time
            <= treatment_end
        ):

            # Arrival event

            current_time = next_arrival_time

            patient = patients[
                patient_index
            ]

            patient_index += 1

            queue.add(
                patient
            )

            if visualize:

                print_queue_snapshot(
                    queue=queue,
                    current_time=current_time,
                    top_k=top_k,
                    event="ARRIVAL",
                    event_patient=patient,
                    alpha=alpha,
                )

            # If doctor is free, immediately start treatment

            if current_patient is None:

                current_patient = queue.pop(
                    current_time=current_time,
                    alpha=alpha,
                )

                treatment_start = (
                    current_time
                )

                treatment_end = (
                    treatment_start
                    + current_patient.service_time
                )

        else:

            # Departure Event

            current_time = treatment_end

            patient = current_patient

            if patient is None:
                raise RuntimeError(
                    "Departure event occurred without a patient in treatment."
                )

            if treatment_start is None:
                raise RuntimeError(
                    "Departure event occurred without a patient in treatment."
                )

            waiting_time = (
                treatment_start
                - patient.arrival_time
            )

            time_in_system = (
                current_time
                - patient.arrival_time
            )

            results.append(
                QueueResult(
                    patient_id=patient.patient_id,
                    stay_id=patient.stay_id,

                    arrival_time=patient.arrival_time,

                    treatment_start=treatment_start,

                    treatment_end=current_time,

                    waiting_time=waiting_time,

                    time_in_system=time_in_system,

                    priority_score=patient.priority_score,

                    sofa_total=patient.sofa_total,

                    reward=patient.reward,
                )
            )

            # Patient has now left the treatment system.

            current_patient = None

            treatment_start = None

            treatment_end = math.inf

            if visualize:

                print_queue_snapshot(
                    queue=queue,
                    current_time=current_time,
                    top_k=top_k,
                    event="ARRIVAL",
                    event_patient=patient,
                    alpha=alpha,
                )

            # If patients are waiting, start treating the next one

            if not queue.empty():

                current_patient = queue.pop(
                    current_time=current_time,
                    alpha=alpha,
                )

                treatment_start = (
                    current_time
                )

                treatment_end = (
                    treatment_start
                    + current_patient.service_time
                )

    return results


# Build patient objects

def create_patients(
    patient_df: pd.DataFrame,
    score_df: pd.DataFrame,
    mean_service_hours: float,
    arrival_rate: float,
    seed: int,
) -> list[Patient]:

    df = patient_df.merge(
        score_df,
        on="stay_id",
        how="inner",
        validate="one_to_one",
    )

    if len(df) != len(patient_df):

        missing = set(
            patient_df["stay_id"]
        ) - set(
            df["stay_id"]
        )

        raise ValueError(
            "Some test patients do not have scores. "
            f"Missing {len(missing)} scores."
        )

    # Generate arrivals

    arrival_times = np.asarray(
        generate_poisson_arrivals(
            n_patients=len(df),
            arrival_rate=arrival_rate,
            seed=seed,
        ),
        dtype=float,
    )

    # Generate treatment times

    service_times = np.asarray(
        generate_service_times(
            n_patients=len(df),
            mean_service_hours=mean_service_hours,
            seed=seed,
        ),
        dtype=float,
    )

    # Build objects

    patients = []

    for i, (_, row) in enumerate(df.iterrows()):

        patients.append(
            Patient(
                patient_id=int(
                    row["stay_id"]
                ),

                stay_id=int(
                    row["stay_id"]
                ),

                arrival_time=float(
                    arrival_times[i]
                ),

                priority_score=float(
                    row["priority_score"]
                ),

                service_time=float(
                    service_times[i]
                ),

                sofa_total=float(
                    row["sofa_total"]
                ),

                reward=float(
                    row["reward"]
                ),
            )
        )

    return patients


# Metrics for model comparisons

def calculate_metrics(
    results: list[QueueResult],
) -> dict:

    if not results:
        return {}

    waiting = np.array(
        [
            r.waiting_time
            for r in results
        ]
    )

    system = np.array(
        [
            r.time_in_system
            for r in results
        ]
    )

    scores = np.array(
        [
            r.priority_score
            for r in results
        ]
    )

    sofa = np.array(
        [
            r.sofa_total
            for r in results
        ]
    )

    return {
        "patients": len(results),

        "mean_waiting_hours": float(
            np.mean(waiting)
        ),

        "median_waiting_hours": float(
            np.median(waiting)
        ),

        "p90_waiting_hours": float(
            np.percentile(waiting, 90)
        ),

        "max_waiting_hours": float(
            np.max(waiting)
        ),

        "mean_waiting_minutes": float(
            np.mean(waiting) * 60
        ),

        "median_waiting_minutes": float(
            np.median(waiting) * 60
        ),

        "p90_waiting_minutes": float(
            np.percentile(waiting, 90) * 60
        ),

        "mean_time_in_system_hours": float(
            np.mean(system)
        ),

        "mean_priority_score": float(
            np.mean(scores)
        ),

        "mean_sofa": float(
            np.mean(sofa)
        ),
    }

# Output

def save_results(
    results: list[QueueResult],
    metrics: dict,
    output_dir: Path,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df = pd.DataFrame(
        [
            {
                "patient_id": r.patient_id,
                "stay_id": r.stay_id,

                "arrival_time_hours":
                    r.arrival_time,

                "treatment_start_hours":
                    r.treatment_start,

                "treatment_end_hours":
                    r.treatment_end,

                "waiting_time_hours":
                    r.waiting_time,

                "waiting_time_minutes":
                    r.waiting_time * 60,

                "time_in_system_hours":
                    r.time_in_system,

                "priority_score":
                    r.priority_score,

                "sofa_total":
                    r.sofa_total,

                "reward":
                    r.reward,
            }
            for r in results
        ]
    )

    result_df.to_parquet(
        output_dir / "queue_results.parquet",
        index=False,
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        output_dir / "queue_metrics.csv",
        index=False,
    )

# Main

def main():

    parser = argparse.ArgumentParser(
        description=(
            "General model-independent patient "
            "queue simulation."
        )
    )

    parser.add_argument(
        "--test_file",
        default=r"...\test.parquet",
        type=str,
        help="General test.parquet",
    )

    parser.add_argument(
        "--score_mode",
        choices=[
            "random",
            "file",
        ],
        default="random",
        help=(
            "How patient priority scores are obtained."
        ),
    )

    parser.add_argument(
        "--score_file",
        type=str,
        default=None,
        help=(
            "Parquet file containing stay_id and "
            "priority_score."
        ),
    )

    parser.add_argument(
        "--output_dir",
        default=r".",
        type=str,
    )

    parser.add_argument(
        "--arrival_rate",
        type=float,
        default=2,
        help=(
            "Mean patient arrivals per hour."
        ),
    )

    parser.add_argument(
        "--service_hours",
        type=float,
        default=0.75,
        help=(
            "Mean treatment time in hours. "
            "Default = 45 minutes."
        ),
    )

    parser.add_argument(
        "--visualize_queue",
        action="store_true",
        default=True,
        help=(
            "Print queue snapshots after "
            "arrival and departure events."
        ),
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help=(
            "Number of highest-priority patients "
            "shown in each queue snapshot."
        ),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    test_file = Path(
        args.test_file
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    # Load patients

    print()
    print("=" * 60)
    print("PATIENT QUEUE SIMULATION")
    print("=" * 60)

    print(
        f"Test file       : {test_file}"
    )

    patient_df = load_test_patients(
        test_file
    )

    print(
        f"Patients        : {len(patient_df):,}"
    )

    # Generate / load scores

    if args.score_mode == "random":

        print(
            "Score mode      : RANDOM"
        )

        score_df = generate_random_scores(
            patient_ids=
                patient_df[
                    "stay_id"
                ].to_numpy(),

            seed=args.seed,
        )

    else:

        if args.score_file is None:
            raise ValueError(
                "--score_file is required "
                "when --score_mode=file."
            )

        print(
            "Score mode      : MODEL FILE"
        )

        print(
            f"Score file      : {args.score_file}"
        )

        score_df = load_score_file(
            Path(
                args.score_file
            )
        )

    # Create patients

    patients = create_patients(
        patient_df=patient_df,
        score_df=score_df,
        mean_service_hours=
            args.service_hours,
        arrival_rate=
            args.arrival_rate,
        seed=args.seed,
    )

    # Run simulation

    print(
        "Running simulation..."
    )

    results = simulate_queue(
        patients=patients,
        visualize=args.visualize_queue,
        top_k=args.top_k,
        alpha=args.alpha,
    )


    # Calculating metrics

    metrics = calculate_metrics(
        results
    )

    # Save

    save_results(
        results=results,
        metrics=metrics,
        output_dir=output_dir,
    )

    # Print summary of metrics

    print()
    print("=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)

    for key, value in metrics.items():

        if isinstance(value, float):

            print(
                f"{key:35s}: "
                f"{value:.4f}"
            )

        else:

            print(
                f"{key:35s}: "
                f"{value:,}"
            )

    print("=" * 60)

    print(
        f"Results: "
        f"{output_dir / 'queue_results.parquet'}"
    )

    print(
        f"Metrics: "
        f"{output_dir / 'queue_metrics.csv'}"
    )


if __name__ == "__main__":
    main()