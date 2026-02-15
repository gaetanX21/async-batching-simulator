# Async Batching Simulator

**Simulate batch composition over time when batching is asynchronous i.e. we fill up the batch with whatever generations finish first.**

The goal is to visualize and understand the transient skew towards short (resp. long) sequences in early (resp. late) batches, alongside the steady-state behavior.

We plot both
- batch composition over time (short vs long sequences in each batch)
- inference engine timeline (short vs long sequences running inside engine and lane occupancy)
to illustrate the transient vs steady state regimes in asynchronous batching.

We use a simple threading-based simulation to model sequence generation and asynchronous batching, and Matplotlib for plotting. We assume two types of sequences (SHORT and LONG) with different generation times, and a fixed number of lanes in the inference engine.

A Streamlit interface allows for interactive parameter tuning and visualization.

## Files

- `simulator.py` - Core simulation logic (threading-based batching and lane management)
- `plotting.py` - Matplotlib plotting functions
- `app.py` - Streamlit web interface
- `plots/` - Saved plots directory

## Running the Simulation

### Command Line (saves plots to `plots/`)

```bash
python simulator.py
```

### Web Interface (interactive)

```bash
streamlit run app.py
```

Then open your browser to the URL shown (typically `http://localhost:8501`)

## Web Interface Features

- **Interactive sliders** for all simulation parameters
- **Real-time visualization** of results
- **Configuration summary** showing expected values (Little's law predictions)
- **Metrics dashboard** showing simulation statistics

## Parameters

### Data Distribution
- **Long sequence multiplier**: How many times longer LONG sequences are vs SHORT
- **Probability of short**: Fraction of sequences that are SHORT
- **Duration variance (sigma)**: Random jitter in generation times (smoothens plots)
- **Base time per token**: Mock processing time (increase if your computer is too slow, decrease for faster simulation)

### Infrastructure & Batching
- **Max concurrent requests**: Inference engine capacity
- **Sequences per batch**: Batch size for downstream processing

### Simulation Scale
- **Total sequences**: Number of sequences to generate (more = smoother plots)

## Output Plots

1. **Batch Composition**: Stacked bar chart showing SHORT vs LONG sequences in each batch
2. **Inference Engine Timeline**: Combined plot showing:
   - Top: Number of sequences in the engine over time
   - Bottom: Heatmap of lane occupancy over time
