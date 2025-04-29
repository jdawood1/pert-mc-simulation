# PERT and Monte Carlo Simulation

This project analyzes project scheduling uncertainty using the Program Evaluation and Review Technique (PERT) and Monte Carlo simulation. It calculates expected task durations, simulates project timelines, and analyzes risk and confidence levels.

## ✨ Features
- Cleans and validates critical path input data.
- Calculates PERT expected durations.
- Runs 1,000 Monte Carlo simulations using triangular distributions.
- Generates project duration confidence curves.
- Produces labeled plots for easy analysis.
- Optional verification of distribution shape for Task 1.

## 📂 Files Generated
- `critical_path_clean.csv` — Cleaned task input data.
- `pert_summary.csv` — PERT calculations for each task.
- `monte_carlo_raw.csv` — Raw Monte Carlo simulation results.
- `confidence_curve.csv` — Confidence interval table.
- `confidence_answers.txt` — Managerial insight answers.
- `task1_histogram.png` — Task 1 duration histogram.
- `confidence_plot.png` — Project confidence curve plot.

## 🛠️ How to Run
1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib scipy
   ```

2. Place `Critical Path Data.csv` in the project directory.

3. Run the script:
   ```bash
   python pert_mc_simulation.py
   ```

## ⚙️ Configuration
- Set `VERIFY = True` at the top of `pert_mc_simulation.py` to plot theoretical vs. simulated distribution for Task 1.

## 📊 Requirements
- Python 3.7+
- pandas
- numpy
- matplotlib
- scipy

## License
MIT License