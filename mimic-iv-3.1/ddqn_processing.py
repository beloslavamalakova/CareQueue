'''
Docstring for mimic-iv-3.1.ddqn_processing
Idea is to make the dataset in a transition table format, specifically in the (s, a, r, s') format
Note that the only rewards at this time will be +1 or -1, indicating whether the patient has been 
discharged or has passed away, respectively. 
Author: Anusha Asthana

'''

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path("/home/20243009/mimic-iv-3.1")
ICU_DIR = BASE_DIR / "icu"
HOSP_DIR = BASE_DIR / "hosp"

# Picking the relevant csv files from the dataset

icustays = pd.read_csv(
    ICU_DIR / "icustays.csv.gz",
    parse_dates=["intime", "outtime"]
)

chartevents = pd.read_csv(
    ICU_DIR / "chartevents.csv.gz",
    parse_dates=["charttime"],
    low_memory=False
)

procedureevents = pd.read_csv(
    ICU_DIR / "procedureevents.csv.gz",
    parse_dates=["starttime"],
    low_memory=False
)

admissions = pd.read_csv(
    HOSP_DIR / "admissions.csv.gz"
)

diagnoses = pd.read_csv(
    HOSP_DIR / "diagnoses_icd.csv.gz",
    low_memory=False
)

items = pd.read_csv(
    ICU_DIR / "d_items.csv.gz"
)

# Sepsis cohort selection

SEPSIS_PREFIXES = ("A40", "A41", "R65")

sepsis_diagnoses = diagnoses[
    (diagnoses.icd_version == 10) &
    (diagnoses.icd_code.str.startswith(SEPSIS_PREFIXES))
]

sepsis_hadm_ids = set(sepsis_diagnoses.hadm_id.unique())

icustays = icustays[icustays.hadm_id.isin(sepsis_hadm_ids)]

# Making 4-hour bins

BIN_SIZE = 4

def make_bins(intime, outtime):
    return pd.date_range(
        start=intime,
        end=outtime,
        freq=f"{BIN_SIZE}H"
    )

# Making the state vector things

STATE_V_ITEMIDS = [
    220045,  # Heart Rate
    223762,  # Temperature Celsius
    220277,  # SpO2
    220050,  # Systolic Blood Pressure
    220051,  # Diastolic Blood Pressure
    220052,  # Mean Blood Pressure
]


