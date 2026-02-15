import streamlit as st
from plotting import plot_batcher_history, plot_inference_engine_history
from simulator import DataDistribution, SimulationConfig, simulate

st.set_page_config(page_title="Continuous Batching Simulation", layout="wide")

st.title("Batch Composition in Async Pipeline")

st.markdown(
    """
**Simulate batch composition over time when batching is asynchronous i.e. we fill up the batch with whatever generations finish first.**

### Transient Regimes
The key insight is that because longer sequences take longer to generate than shorter ones, we observe a transient regime at the beginning of the simulation where batches are mostly made up of short sequences.

Likewise, we observe a transient regime at the end of the simulation where batches are dominated by long sequences.


### Steady-State Regime
Crucially, **in the long run (i.e. outside beginning and end of simulation), batch composition only depends on the data distribution.**

That is, if we have a fraction $f_\\text{SHORT}$ of SHORT sequences (and thus $1-f_\\text{SHORT}$ of LONG sequences), then batches must respect these proportions in the steady-state regime.

The logical explanation behind this somewhat counterintuitive result is that regardless of what goes inside the batches in the initial and final transient regimes, every single sequence created throughout the simulation is sampled from the same static data distribution, so you cannot "cheat" the proportions in the long run.

*Note that in practice, sequence length (and thus generation duration) is better modeled by a continuous distribution, e.g. Laplace to account for fat tails. However, for the sake of simplicity, here we're considering a binary distribution (SHORT vs LONG sequences).*


### Severity
The severity of the transient regimes depends on:
- length of SHORT vs LONG sequences (sequence length asymmetry)
- batch size vs the number of concurrent requests (concurrency aggravator)

We can thus approximate:

$$\\text{severity} \\sim \\text{LONG}_\\text{multiplier} \\times \\frac{\\text{max concurrent requests}}{\\text{seq per batch}}$$


# Try it yourself!
Adjust the parameters below and click **Run Simulation** to see the results.
"""
)

# Sidebar with configuration
st.sidebar.header("⚙️ Simulation Configuration")

st.sidebar.subheader("Data Distribution")
long_multiplier = st.sidebar.slider(
    "Long sequence multiplier",
    min_value=2,
    max_value=10,
    value=4,
    help="LONG sequences are this many times longer than SHORT sequences",
)
prob_short = st.sidebar.slider(
    "Fraction of SHORT sequences",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.1,
    help="Fraction of sequences that are SHORT (vs LONG)",
)
sigma = st.sidebar.slider(
    "Duration variance (sigma)",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.1,
    help="Standard deviation for generation duration jitter (smoothens plots)",
)
base_dt = st.sidebar.slider(
    "Mock generation speed (s/token)",
    min_value=0.1,
    max_value=1.0,
    value=0.1,
    step=0.1,
    help="Mock seconds per token (increase only if buggy outputs)",
)

st.sidebar.subheader("Infrastructure & Batching")
max_concurrent_requests = st.sidebar.select_slider(
    "Max concurrent requests",
    options=[10, 50, 100, 250, 500, 1000],
    value=100,
    help="Maximum number of requests handled concurrently",
)
seq_per_batch = st.sidebar.select_slider(
    "Sequences per batch",
    options=[16, 32, 64, 128, 256, 512, 1024],
    value=128,
    help="Number of sequences to collect before creating a batch",
)

st.sidebar.subheader("Simulation Scale")
total_sequences = st.sidebar.slider(
    "Total sequences to generate",
    min_value=1000,
    max_value=20_000,
    value=10_000,
    step=1000,
    help="Total number of sequences to generate (more = smoother plots but slower)",
)

# Display configuration summary
# Create a preview config to show computed properties
preview_config = SimulationConfig(
    dist=DataDistribution(
        long=long_multiplier, prob_short=prob_short, sigma=sigma, base_dt=base_dt
    ),
    max_concurrent_requests=max_concurrent_requests,
    seq_per_batch=seq_per_batch,
    total_sequences=total_sequences,
)

with st.sidebar.expander("📊 Configuration Computed Properties", expanded=False):
    st.write(
        f"• Expected SHORT/batch: {preview_config.num_short_per_batch:.1f} seq/batch"
    )
    st.write(
        f"• Little's law (SHORT load): {preview_config.short_load:.1f} seq in flight"
    )
    st.write(f"• Long-term intensity: {preview_config.long_term_intensity:.2f} req/s")
    st.write(f"• Transient severity: {preview_config.transient_regime_severity:.2f}")
    st.write(f"• Batch duration: {preview_config.batch_duration:.1f} s")
    st.write(f"• Num batches: {preview_config.num_batches}")
    st.write(f"• Simulation duration: {preview_config.simulation_duration:.1f} s")

# Run button
run_simulation = st.sidebar.button(
    f"🚀 Run Simulation (⏱️ ~{preview_config.simulation_duration:.1f}s)",
    type="primary",
    use_container_width=True,
)

# Main content area
if run_simulation:
    # Create configuration
    config = SimulationConfig(
        dist=DataDistribution(
            long=long_multiplier, prob_short=prob_short, sigma=sigma, base_dt=base_dt
        ),
        max_concurrent_requests=max_concurrent_requests,
        seq_per_batch=seq_per_batch,
        total_sequences=total_sequences,
    )

    # Run simulation with progress indicator
    with st.spinner("Running simulation..."):
        batcher_history, inference_engine_history = simulate(config=config)

    st.success("✅ Simulation completed!")

    # Plot 1: Batcher History
    st.header("📦 Batch Composition")
    fig1 = plot_batcher_history(batcher_history, config, output="return")
    if fig1:
        st.pyplot(fig1)

    # Plot 2: Inference Engine
    st.header("🔧 Inference Engine Timeline")
    fig2 = plot_inference_engine_history(
        inference_engine_history, config, output="return"
    )
    if fig2:
        st.pyplot(fig2)

else:
    st.info(
        "👈 Configure the simulation parameters in the sidebar and click **Run Simulation** to start."
    )
