# Portfolio-Optimisation
Mean Variance Optimisation

- Log return distribution of the different assets
![image](https://github.com/MOMOJordan/Portfolio-Optimisation/assets/86100448/a7de9e85-7725-4a6f-9020-8aedaab38f55)

- The portfolio is highly correlated
  
![image](https://github.com/MOMOJordan/Portfolio-Optimisation/assets/86100448/c719131f-d5e5-47d2-8441-2990844b6b1a)

- According to the results on the test set, if we fear about any loose, it is not a good idea to do a long short strategy but in the other case, if we are a risk taker, long short is the best option. This result is quite different from what we could expect base on the training curve. There is kind of "overfitting" or "stability" issue. We have to correct it using a regularisation

![image](https://github.com/MOMOJordan/Portfolio-Optimisation/assets/86100448/9f558d77-56bb-4611-898b-9d3b3e6afcad)

- Stabilisation of the portfolio by ridge regression

![image](https://github.com/MOMOJordan/Portfolio-Optimisation/assets/86100448/876b7fd3-143b-47de-8517-a68d3a537798)

![image](https://github.com/MOMOJordan/Portfolio-Optimisation/assets/86100448/95949c43-8a77-4678-bfd6-4d81ff534ee4)


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

timesteps = list(range(6))
assets = ['Asset 1', 'Asset 2', 'Asset 3']
np.random.seed(0)
weights = np.random.rand(6, 3)
returns = np.random.randn(6, 3) * 0.01
spreads = np.random.rand(6, 3) * 0.005

def to_df(data, value_name):
    df = pd.DataFrame(data, columns=assets)
    df['Time'] = timesteps
    return df.melt(id_vars='Time', var_name='Asset', value_name=value_name)

df_weights = to_df(weights, 'Weight')
df_returns = to_df(returns, 'Return')
df_spreads = to_df(spreads, 'Spread')

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                    subplot_titles=["Weights", "Returns", "Spreads"])

# Add fake legend header for Weights
fig.add_trace(go.Scatter(
    x=[None], y=[None], mode='lines',
    name='— Weights —',  # bold-style separator
    hoverinfo='skip', showlegend=True,
    line=dict(color='rgba(0,0,0,0)'),  # invisible line
), row=1, col=1)

for asset in assets:
    fig.add_trace(go.Scatter(
        x=df_weights[df_weights.Asset == asset]['Time'],
        y=df_weights[df_weights.Asset == asset]['Weight'],
        mode='lines+markers',
        name=f"{asset} (Weight)",
        showlegend=True
    ), row=1, col=1)

# Add fake legend header for Returns
fig.add_trace(go.Scatter(
    x=[None], y=[None], mode='lines',
    name='— Returns —',
    hoverinfo='skip', showlegend=True,
    line=dict(color='rgba(0,0,0,0)'),
), row=2, col=1)

for asset in assets:
    fig.add_trace(go.Scatter(
        x=df_returns[df_returns.Asset == asset]['Time'],
        y=df_returns[df_returns.Asset == asset]['Return'],
        mode='lines+markers',
        name=f"{asset} (Return)",
        showlegend=True
    ), row=2, col=1)

# Add fake legend header for Spreads
fig.add_trace(go.Scatter(
    x=[None], y=[None], mode='lines',
    name='— Spreads —',
    hoverinfo='skip', showlegend=True,
    line=dict(color='rgba(0,0,0,0)'),
), row=3, col=1)

for asset in assets:
    fig.add_trace(go.Scatter(
        x=df_spreads[df_spreads.Asset == asset]['Time'],
        y=df_spreads[df_spreads.Asset == asset]['Spread'],
        mode='lines+markers',
        name=f"{asset} (Spread)",
        showlegend=True
    ), row=3, col=1)

fig.update_layout(
    height=1000,
    width=1200,
    title="Weights, Returns, and Spreads Over Time",
    font=dict(size=14),
    margin=dict(t=80, r=300),
    legend=dict(
        x=1.02,
        y=1,
        xanchor="left",
        yanchor="top",
        font=dict(size=12),
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="gray",
        borderwidth=1,
        itemsizing="constant"
    )
)

fig.show()

import numpy as np
import plotly.graph_objects as go

# Sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)
color_vals = np.cos(x)  # This will determine the color of each segment

# Normalize color values for mapping to colorscale
color_norm = (color_vals - color_vals.min()) / (color_vals.max() - color_vals.min())

problem.solve(
    solver=cp.MOSEK,
    mosek_params={
        # Tighter tolerances for more accurate MIQP solutions
        "MSK_DPAR_MIO_TOL_REL_GAP": 1e-6,      # Relative MIP gap
        "MSK_DPAR_MIO_TOL_ABS_GAP": 1e-8,      # Absolute MIP gap
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-8,     # Primal feasibility
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-8,     # Dual feasibility
        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-8,   # Interior-point relative gap
    }
)
Key MOSEK parameters to set when using Big-M with potentially large values:
Parameter name	Purpose	Suggested setting for Big-M problems (~1e4 scale)
MSK_IPAR_INTPNT_SCALING	Enable automatic scaling	1 (enabled) — always keep on for stability
MSK_DPAR_INTPNT_CO_TOL_DFEAS	Dual feasibility tolerance	1e-6 to 1e-4 (loosen if needed for speed)
MSK_DPAR_INTPNT_CO_TOL_PFEAS	Primal feasibility tolerance	1e-6 to 1e-4
MSK_DPAR_INTPNT_CO_TOL_MU_RED	Complementarity reduction tolerance	1e-6 to 1e-4
MSK_IPAR_MIO_TOL_REL_GAP	Relative MIP gap tolerance	1e-2 (1%) for faster but still decent MIP solutions
MSK_DPAR_MIO_MAX_TIME	Time limit (seconds)	Set as needed (e.g., 60 or 300)
cp.norm_inf(w[:, t+1] - w[:, t]) <= epsilon + M * (1 - z[t, 0])
for t in range(T - 1):
    for i in range(n_assets):
        constraints += [
            w[i, t+1] - w[i, t] <= epsilon + M[i] * (1 - z[t,0]),
            w[i, t] - w[i, t+1] <= epsilon + M[i] * (1 - z[t,0])
        ]
import os
import cvxpy as cp
import optuna
import multiprocessing
import psutil

# ✅ 1. Limit MOSEK to 1 thread globally (must be set before solving)
os.environ["MOSEK_NUM_THREADS"] = "1"

# ✅ 2. Function to solve a CVXPY problem using MOSEK with limited threads
def objective(trial):
    # Example portfolio data
    n_assets = 3
    expected_returns = [0.05, 0.07, 0.06]
    cov_matrix = [[0.1, 0.01, 0.02],
                  [0.01, 0.12, 0.03],
                  [0.02, 0.03, 0.15]]

    risk_aversion = trial.suggest_float("risk_aversion", 0.1, 2.0, step=0.1)

    w = cp.Variable(n_assets)
    ret = cp.sum(cp.multiply(expected_returns, w))
    risk = cp.quad_form(w, cov_matrix)
    objective_fn = cp.Maximize(ret - risk_aversion * risk)
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(objective_fn, constraints)

    # MOSEK with 1 thread
    prob.solve(solver=cp.MOSEK, mosek_params={"MSK_IPAR_NUM_THREADS": 1})

    return -prob.value  # minimize negative return

# ✅ 3. Limit Optuna parallelism
max_cpus = multiprocessing.cpu_count()
mosek_threads = 1
optuna_jobs = max(1, max_cpus // mosek_threads)  # Safe number of parallel jobs

# Print diagnostics
print(f"Total logical CPUs: {max_cpus}")
print(f"MOSEK threads per trial: {mosek_threads}")
print(f"Optuna parallel jobs allowed: {optuna_jobs}")
print(f"Current CPU usage: {psutil.cpu_percent(interval=1)}%")

# ✅ 4. Create and run Optuna study (with limited CPU usage)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10, n_jobs=optuna_jobs)

# ✅ 5. Print best result
print("Best parameters:", study.best_params)
print("Best value:", study.best_value)

# Generate line segments and colors
segments = []
for i in range(len(x) - 1):
    segments.append(go.Scatter(
        x=[x[i], x[i+1]],
        y=[y[i], y[i+1]],
        mode='lines',
        line=dict(
            color=color_norm[i],  # normalized value
            colorscale='Viridis',
            cmin=0,
            cmax=1,
            width=3
        ),
        showlegend=False,
        hoverinfo='none'
    ))

# Create the figure
fig = go.Figure(segments)
fig.update_layout(title='Colored Line Based on Separate Array')
fig.show()
