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

def get_state_vector(ce, start, end):
    window = ce[(ce.charttime >= start) & (ce.charttime < end)]
    state = []

    for iid in STATE_V_ITEMIDS:
        vals = window.loc[window.itemid == iid, "valuenum"]
        state.append(vals.mean() if not vals.empty else np.nan)

    return np.array(state, dtype=np.float32)

# Getting the relevant actions done on patient

VENTILATION_IDS = {225794} # Action label 1
INVASIVE_LINES_IDS = {224263, 224268, 225752} # Action label 2
U_CATHETER_IDS = {229351} # Action label 3
IN_EX_TUBATION_IDS = {227194} # Action label 4
DIALYSIS_IDS = {225802} # Action label 5

def get_action(pe, start, end):
    window = pe[(pe.starttime >= start) & (pe.starttime < end)]

    if window.empty:
        return 0
    if window.itemid.isin(VENTILATION_IDS).any():
        return 1
    if window.itemid.isin(INVASIVE_LINES_IDS).any():
        return 2
    if window.itemid.isin(U_CATHETER_IDS).any():
        return 3
    if window.itemid.isin(IN_EX_TUBATION_IDS).any():
        return 4
    if window.itemid.isin(DIALYSIS_IDS).any():
        return 5
    return 0    

# Now making it into the transition table format

admissions_idx = admissions.set_index("hadm_id")["hospital_expire_flag"]

def build_transitions_for_stay(stay):
    stay_id = stay.stay_id
    hadm_id = stay.hadm_id
    
    try:
        died = admissions_idx.loc[hadm_id] == 1
    except KeyError:
        died = False

    ce = chartevents[chartevents.stay_id == stay_id]
    pe = procedureevents[procedureevents.stay_id == stay_id]

    bins = make_bins(stay.intime, stay.outtime)
    
    if len(bins) < 2:
        return []

    states = []
    actions = []

    for i in range(len(bins) - 1):
        s = get_state_vector(ce, bins[i], bins[i+1])
        a = get_action(pe, bins[i], bins[i+1])
        states.append(s)
        actions.append(a)

    transitions = []

    for t in range(len(states) - 1):
        transitions.append({
            "state": states[t],
            "action": actions[t],
            "reward": 0,
            "next_state": states[t+1],
            "done": 0
        })

    transitions.append({
        "state": states[-1],
        "action": actions[-1],
        "reward": -1 if died else 1,
        "next_state": np.zeros_like(states[-1]),
        "done": 1
    })

    return transitions

# Making dataframe

all_transitions = []

for _, stay in icustays.iterrows():
    transitions = build_transitions_for_stay(stay)
    all_transitions.extend(transitions)

STATE_COLS = ["HR", "TEMP", "SPO2", "SBP", "DBP", "MBP"]

rows = []

for transition in all_transitions:
    row = {
        "action": transition["action"],
        "reward": transition["reward"],
        "done": transition["done"],
    }
    
    for i, col in enumerate(STATE_COLS):
        row[f"s_{col}"] = transition["state"][i]
        
    for i, col in enumerate(STATE_COLS):
        row[f"s_next_{col}"] = transition["next_state"][i]
    
    rows.append(row)
    
transitions_df = pd.DataFrame(rows)

print(transitions_df.head(20))

