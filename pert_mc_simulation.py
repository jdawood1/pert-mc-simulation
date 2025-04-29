"""
pert_mc_simulation.py
Author: John Dawood
Assignment: Monte Carlo Simulation and PERT Analysis
Date: April 29, 2025

This script analyzes project scheduling uncertainty using PERT and Monte Carlo 
simulation. It reads project task data from a CSV file, calculates expected durations, 
simulates random outcomes using triangular distributions, and generates summary outputs, 
plots, and managerial insights.
"""

# --- Import necessary libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import triang  # Used for optional theoretical distribution verification

# --- Configuration ---
VERIFY = False  # Set to True to overlay theoretical PDF for Task 1

# --- File Constants ---
INPUT_FILE = 'Critical Path Data.csv'
CLEAN_FILE = 'critical_path_clean.csv'
PERT_SUMMARY_FILE = 'pert_summary.csv'
MC_RAW_FILE = 'monte_carlo_raw.csv'
CONFIDENCE_CURVE_FILE = 'confidence_curve.csv'
CONFIDENCE_ANSWERS_FILE = 'confidence_answers.txt'
TASK1_HISTOGRAM_FILE = 'task1_histogram.png'
CONFIDENCE_PLOT_FILE = 'confidence_plot.png'

REQUIRED_OUTPUT_FILES = [
    CLEAN_FILE, PERT_SUMMARY_FILE, MC_RAW_FILE, CONFIDENCE_CURVE_FILE,
    CONFIDENCE_ANSWERS_FILE, TASK1_HISTOGRAM_FILE, CONFIDENCE_PLOT_FILE
]

# --- Functions ---

def validate_and_clean_input(file_path):
    """
    Reads and validates the input CSV file. Handles standard and transposed formats.
    Saves cleaned data to critical_path_clean.csv.
    """
    try:
        raw_df = pd.read_csv(file_path)

        if {'Task', 'O', 'ML', 'P'}.issubset(raw_df.columns):
            debug_msg = "Detected standard row-wise format with columns: Task, O, ML, P."
            df = raw_df[['Task', 'O', 'ML', 'P']].copy()
        elif raw_df.iloc[0, 0].strip().lower() in ['pess', 'ml', 'opt']:
            debug_msg = "Detected transposed format. Transposing."
            raw_df = pd.read_csv(file_path, index_col=0)
            df = raw_df.T
            if {'Pess', 'ML', 'Opt'}.issubset(df.columns):
                df.rename(columns={'Opt': 'O', 'Pess': 'P'}, inplace=True)
                df['Task'] = df.index
                df = df[['Task', 'O', 'ML', 'P']].copy()
            else:
                raise ValueError("Transposed format missing expected columns.")
        else:
            raise ValueError("Unsupported CSV format. Required structure not detected.")

        # Ensure all numeric columns are floats
        df[['O', 'ML', 'P']] = df[['O', 'ML', 'P']].astype(float)
        df.to_csv(CLEAN_FILE, index=False)
        print(f"[✔] Cleaned data saved to: {CLEAN_FILE}")
        print(f"[ℹ] {debug_msg}")
        return df

    except Exception as e:
        print(f"[✖] Error during validation: {e}")
        raise

def calculate_pert(df):
    """
    Calculates PERT expected durations and min/max ranges.
    Saves summary to pert_summary.csv.
    """
    df['PERT'] = (df['O'] + 4 * df['ML'] + df['P']) / 6
    df['Min'] = df['O']
    df['Max'] = df['P']
    df['FixedSimulation'] = np.where(df['O'] == df['P'], 'Yes', 'No')

    total_row = pd.DataFrame({
        'Task': ['TOTAL'],
        'O': [df['O'].sum()],
        'ML': [df['ML'].sum()],
        'P': [df['P'].sum()],
        'PERT': [df['PERT'].sum()],
        'Min': [df['Min'].sum()],
        'Max': [df['Max'].sum()],
        'FixedSimulation': ['-']
    })

    summary_df = pd.concat([df, total_row], ignore_index=True)
    summary_df.to_csv(PERT_SUMMARY_FILE, index=False)
    print(f"[✔] PERT summary saved to: {PERT_SUMMARY_FILE}")
    return df

def monte_carlo_simulation(df, n=1000):
    """
    Runs Monte Carlo simulation for all tasks.
    Each task is sampled from a triangular distribution.
    """
    task_names = df['Task'].values
    samples = []
    for idx, row in df.iterrows():
        o, ml, p = row['O'], row['ML'], row['P']
        if o == p:
            print(f"[⚠] Task '{row['Task']}' has O == P == {o}. Using constant ML value.")
            s = np.full(n, ml)
        else:
            s = np.random.triangular(left=o, mode=ml, right=p, size=n)
        samples.append(s)

    samples = np.array(samples)
    total_durations = samples.sum(axis=0)

    mc_df = pd.DataFrame(samples.T, columns=task_names)
    mc_df['Total'] = total_durations
    mc_df.to_csv(MC_RAW_FILE, index=False)
    print(f"[✔] Monte Carlo simulation data saved to: {MC_RAW_FILE}")
    return samples, total_durations

def plot_task1_histogram(samples):
    """
    Plots and saves a histogram of Task 1 simulated duration.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(samples[0], bins=30, color='skyblue', edgecolor='black')
    plt.title('Task 1 Duration Histogram')
    plt.xlabel('Duration')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(TASK1_HISTOGRAM_FILE)
    plt.close()
    print(f"[✔] Task 1 histogram saved to: {TASK1_HISTOGRAM_FILE}")

def calculate_confidence_curve(totals):
    """
    Calculates project duration confidence levels from 60% to 99.9%.
    """
    percentiles = np.arange(60.0, 100.0, 0.1)
    durations = np.percentile(totals, percentiles)
    conf_df = pd.DataFrame({'Percentile': percentiles, 'Duration': durations})
    conf_df.to_csv(CONFIDENCE_CURVE_FILE, index=False)
    print(f"[✔] Confidence curve saved to: {CONFIDENCE_CURVE_FILE}")
    return conf_df

def plot_confidence_curve(conf_df):
    """
    Plots and saves the project confidence curve.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(conf_df['Percentile'], conf_df['Duration'], color='darkgreen')
    plt.title('Confidence Curve')
    plt.xlabel('Confidence Level (%)')
    plt.ylabel('Project Duration')
    plt.grid(True)
    plt.savefig(CONFIDENCE_PLOT_FILE)
    plt.close()
    print(f"[✔] Confidence plot saved to: {CONFIDENCE_PLOT_FILE}")

def save_confidence_answers(conf_df):
    """
    Extracts and saves minimum project durations for 70%, 80%, and 90% confidence.
    """
    answers = {}
    for conf in [70.0, 80.0, 90.0]:
        row = conf_df[conf_df['Percentile'] >= conf].iloc[0]
        answers[int(conf)] = row['Duration']

    with open(CONFIDENCE_ANSWERS_FILE, 'w') as f:
        for k, v in answers.items():
            f.write(f"{k}% Confidence: Project will finish in {v:.2f} units or less\n")

    print(f"[✔] Confidence answers saved to: {CONFIDENCE_ANSWERS_FILE}")

def verify_task1_distribution(task_row, samples_task1):
    """
    (Optional) Overlays Task 1's histogram with the theoretical PDF using scipy.stats.triang.
    """
    O, ML, P = task_row['O'], task_row['ML'], task_row['P']
    if O == P:
        print("[ℹ] Task 1 is constant — skipping PDF overlay.")
        return

    c = (ML - O) / (P - O)
    loc = O
    scale = P - O
    x = np.linspace(O, P, 500)
    pdf = triang.pdf(x, c, loc=loc, scale=scale)

    plt.figure(figsize=(10, 6))
    plt.hist(samples_task1, bins=30, density=True, alpha=0.4, edgecolor='black', label="Simulated Histogram")
    plt.plot(x, pdf, 'r--', label="Theoretical PDF", linewidth=2)
    plt.title("Task 1: Simulated Histogram vs. Theoretical PDF")
    plt.xlabel("Duration")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()

def verify_outputs():
    """
    Verifies that all required output files have been generated.
    """
    print("\nVerifying output files:")
    all_exist = True
    for f in REQUIRED_OUTPUT_FILES:
        if not os.path.exists(f):
            print(f"[⚠] Missing: {f}")
            all_exist = False
        else:
            print(f"[✔] Found: {f}")
    if all_exist:
        print("\nAll required files are ready for submission.")
    else:
        print("\nOne or more required files are missing.")

def main():
    """
    Main execution function for the entire simulation and output generation process.
    """
    if not os.path.exists(INPUT_FILE):
        print(f"[✖] Missing input file: {INPUT_FILE}")
        return

    try:
        df_clean = validate_and_clean_input(INPUT_FILE)
        pert_df = calculate_pert(df_clean)
        samples, totals = monte_carlo_simulation(pert_df)
        plot_task1_histogram(samples)
        conf_df = calculate_confidence_curve(totals)
        plot_confidence_curve(conf_df)
        save_confidence_answers(conf_df)

        if VERIFY:
            verify_task1_distribution(pert_df.iloc[0], samples[0])

        verify_outputs()
    except Exception as e:
        print(f"[❌] Script failed with error: {e}")

if __name__ == '__main__':
    main()