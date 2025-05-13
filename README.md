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

