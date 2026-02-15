# Async Batching Simulator

**Simulate batch composition over time when batching is asynchronous i.e. we fill up the batch with whatever generations finish first.**

## Files

- `simulator.py` - Core simulation logic (threading-based batching and lane management)
- `plotting.py` - Matplotlib plotting functions
- `app.py` - Streamlit web interface
- `plots/` - Saved plots directory

## Running the Simulation

### Command Line (saves plots to `plots/`)

```bash
uv run python simulator.py
```

### Web Interface (interactive)

```bash
uv run streamlit run app.py
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
