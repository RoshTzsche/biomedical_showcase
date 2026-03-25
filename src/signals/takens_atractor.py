import os
import numpy as np
import scipy.signal as signal
import pywt
import mne
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numpy.lib.stride_tricks import as_strided

# =============================================================================
# 1. INGESTION & PREPROCESSING PIPELINE
# =============================================================================

def load_and_preprocess_eeg(filepath: str, target_channel: str, tmin: float = 0, tmax: float = 10):
    """
    Ingests clinical EEG data (EDF/BDF), resolves spatial measurement basis mismatches, 
    and applies zero-phase FIR filtering to prevent phase distortion.
    """
    raw = mne.io.read_raw_edf(filepath, preload=True)
    available_channels = raw.ch_names
    
    # Topological Basis Verification
    if target_channel not in available_channels:
        fallback_candidates = ['C3-P3', 'FP1-F7', 'FZ-CZ', 'C4-P4']
        valid_fallback = next((ch for ch in fallback_candidates if ch in available_channels), available_channels[0])
        
        print(f"\n[!] Spatial node '{target_channel}' not found in the measurement matrix.")
        print(f"[!] Re-projecting manifold onto available spatial gradient: '{valid_fallback}'")
        target_channel = valid_fallback

    # Mathematical Filtering (Zero-phase FIR to preserve topological geometry)
    raw.filter(l_freq=1.0, h_freq=45.0, fir_design='firwin', phase='zero', verbose=False)
    
    # Extract the spatial vector
    raw.pick(picks=[target_channel])
    raw.crop(tmin=tmin, tmax=tmax)
    
    data, times = raw[:]
    sampling_freq = raw.info['sfreq']
    
    return data[0, :], sampling_freq, times, target_channel

# =============================================================================
# 2. NON-LINEAR DYNAMICS & TOPOLOGY (TAKENS' THEOREM)
# =============================================================================

def compute_optimal_tau(x: np.ndarray, fs: float, max_lag_sec: float = 0.5) -> int:
    """
    Computes the optimal embedding lag (tau) by finding the first zero-crossing
    of the continuous autocorrelation function via the Wiener-Khinchin theorem.
    """
    x_centered = x - np.mean(x)
    max_lag_samples = int(max_lag_sec * fs)
    
    # O(N log N) Autocorrelation via FFT
    acf = signal.correlate(x_centered, x_centered, mode='full', method='fft')
    acf = acf[len(acf)//2:] 
    acf = acf[:max_lag_samples]
    
    acf_normalized = acf / acf[0]
    
    # Detect first root (zero-crossing)
    zero_crossings = np.where(np.diff(np.sign(acf_normalized)))[0]
    
    if len(zero_crossings) > 0:
        optimal_tau = zero_crossings[0]
        print(f"      -> Optimal Tau via Autocorrelation root: {optimal_tau} samples ({(optimal_tau/fs)*1000:.1f} ms).")
        return optimal_tau
    else:
        fallback_tau = int(fs * 0.05)
        print(f"      -> [Warning] No zero-crossing found. Defaulting to empirical lag: {fallback_tau} samples.")
        return fallback_tau

def compute_phase_space_vectorized(x: np.ndarray, tau: int, dim: int = 3) -> np.ndarray:
    """
    Computes the Takens' delay embedding using advanced NumPy stride tricks for O(1) memory mapping.
    """
    N = len(x)
    M = N - (dim - 1) * tau
    
    if M <= 0:
        raise ValueError("Time series is too short for the selected dimension and temporal delay.")
    
    itemsize = x.itemsize
    shape = (M, dim)
    strides = (itemsize, itemsize * tau)
    
    return as_strided(x, shape=shape, strides=strides)

# =============================================================================
# 3. NON-STATIONARY SPECTRAL ANALYSIS
# =============================================================================

def compute_morlet_energy(x: np.ndarray, fs: float, freqs: np.ndarray, num_points: int) -> np.ndarray:
    """
    Calculates the integrated instantaneous spectral energy using the Complex Morlet wavelet.
    """
    scales = pywt.frequency2scale('morl', freqs / fs)
    cwt_matrix, _ = pywt.cwt(x, scales, 'morl', sampling_period=1/fs)
    
    # L2 Norm Equivalent (Instantaneous Power)
    energy = np.sum(np.abs(cwt_matrix)**2, axis=0)
    
    # Min-Max Normalization
    energy_normalized = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
    
    return energy_normalized[:num_points]

# =============================================================================
# 4. CLINICAL HCI & WEBGL RENDERING
# =============================================================================

def render_synchronized_dashboard(t_vector: np.ndarray, x: np.ndarray, E: np.ndarray, 
                                  color_grad: np.ndarray, channel_name: str, tau: int):
    """
    Renders a synchronized, dual-pane clinical visualization proving the 
    temporal isomorphism between the 1D signal and the 3D topological manifold.
    """
    M = E.shape[0]
    t_sync = t_vector[:M]
    x_sync = x[:M]
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=(
            f"1D Observation: {channel_name} (Color = Transient Spectral Power)", 
            f"3D State-Space Reconstruction (Delay = {tau} samples)"
        ),
        horizontal_spacing=0.05
    )

    # PANE 1: 1D Time Series
    fig.add_trace(
        go.Scatter(
            x=t_sync, 
            y=x_sync,
            mode='lines+markers',
            marker=dict(size=3, color=color_grad, colorscale='Turbo', showscale=False),
            line=dict(color='rgba(255,255,255,0.2)', width=1),
            name="EEG Signal"
        ),
        row=1, col=1
    )

    # PANE 2: 3D Topological Manifold
    fig.add_trace(
        go.Scatter3d(
            x=E[:, 0], y=E[:, 1], z=E[:, 2],
            mode='lines',
            line=dict(
                width=3, color=color_grad, colorscale='Turbo',
                cmin=0, cmax=1, colorbar=dict(title="L2 Spectral Energy", thickness=15, x=1.05)
            ),
            name="Phase Space"
        ),
        row=1, col=2
    )

    fig.update_layout(
        template="plotly_dark",
        title_text=f"Clinical Dynamics Dashboard - Node: {channel_name}",
        title_font_size=20,
        height=800,
        showlegend=False,
        margin=dict(l=20, r=20, b=20, t=80)
    )

    fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_yaxes(title_text="Potential (μV)", row=1, col=1)
    
    fig.update_scenes(
        xaxis_title='x(t)', yaxis_title='x(t + τ)', zaxis_title='x(t + 2τ)',
        bgcolor='rgb(15, 15, 15)'
    )

    fig.show()

# =============================================================================
# 5. ENGINE EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    # Configuration Matrix
    FILEPATH_EEG = "./chb01_01.edf" 
    TARGET_CHANNEL = "Cz"
    EPOCH_TMIN = 0.0      
    EPOCH_TMAX = 15.0     
    MAX_TAU_SEARCH = 0.5  

    print("="*60)
    print("BIOMEDICAL SIGNAL PROCESSING & TOPOLOGICAL ANALYSIS ENGINE")
    print("="*60)

    try:
        if not os.path.exists(FILEPATH_EEG):
            raise FileNotFoundError(f"Clinical file '{FILEPATH_EEG}' missing from directory.")

        print(f"\n[1/5] Ingesting clinical data and verifying spatial basis...")
        signal_array, fs, time_vector, actual_channel = load_and_preprocess_eeg(
            filepath=FILEPATH_EEG, target_channel=TARGET_CHANNEL, 
            tmin=EPOCH_TMIN, tmax=EPOCH_TMAX
        )
        print(f"      -> Success: {len(signal_array)} samples extracted at {fs} Hz.")
        print(f"      -> Spatial Node established at: {actual_channel}")

        print(f"\n[2/5] Computing optimal delay coordinate metric (R_xx(τ) = 0)...")
        optimal_tau = compute_optimal_tau(x=signal_array, fs=fs, max_lag_sec=MAX_TAU_SEARCH)

        print(f"\n[3/5] Executing state-space reconstruction (dim=3, τ={optimal_tau})...")
        manifold_matrix = compute_phase_space_vectorized(x=signal_array, tau=optimal_tau, dim=3)
        print(f"      -> Manifold successfully embedded into R3. Shape: {manifold_matrix.shape}")

        print(f"\n[4/5] Computing transient spectral power via Complex Morlet Transform...")
        frequency_bands = np.linspace(1, 45, 40) 
        manifold_length = manifold_matrix.shape[0]
        
        energy_gradient = compute_morlet_energy(
            x=signal_array, fs=fs, freqs=frequency_bands, num_points=manifold_length
        )

        print(f"\n[5/5] Launching synchronized topological dashboard...")
        render_synchronized_dashboard(
            t_vector=time_vector, x=signal_array, E=manifold_matrix, 
            color_grad=energy_gradient, channel_name=actual_channel, tau=optimal_tau
        )
        
        print("\nPipeline execution completed successfully.")

    except Exception as e:
        print(f"\n[!] SYSTEM EXCEPTION: A mathematical or pipeline error occurred.")
        print(f"    Trace: {e}")
