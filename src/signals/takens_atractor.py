import numpy as np
import plotly.graph_objects as go
import mne
import pywt
from numpy.lib.stride_tricks import as_strided

def load_and_preprocess_eeg(filepath: str, target_channel: str, tmin: float = 0, tmax: float = 10):
    """
    Ingests clinical EEG data (EDF/BDF), resolves spatial measurement basis mismatches, 
    and applies zero-phase FIR filtering to prevent phase distortion.
    
    Parameters:
        filepath (str): Path to the clinical data file.
        target_channel (str): Desired unipolar or bipolar spatial node.
        tmin (float): Start time of the epoch in seconds.
        tmax (float): End time of the epoch in seconds.
        
    Returns:
        tuple: (1D voltage array, sampling frequency, time vector, actual channel used)
    """
    # 1. Load the raw data into memory
    raw = mne.io.read_raw_edf(filepath, preload=True)
    
    # 2. Topological Basis Verification
    # Clinical datasets (like CHB-MIT) often use bipolar longitudinal montages (e.g., FP1-F7) 
    # rather than unipolar referential ones (e.g., Cz). We must dynamically check the basis.
    available_channels = raw.ch_names
    
    if target_channel not in available_channels:
        # Define a hierarchy of robust cortical gradients if the target is missing
        fallback_candidates = ['C3-P3', 'FP1-F7', 'FZ-CZ']
        
        # Find the first available fallback, or default to the first channel in the matrix
        valid_fallback = next((ch for ch in fallback_candidates if ch in available_channels), available_channels[0])
        
        print(f"\n[!] Spatial node '{target_channel}' not found in the measurement matrix.")
        print(f"[!] Re-projecting manifold onto available spatial gradient: '{valid_fallback}'")
        target_channel = valid_fallback

    # 3. Mathematical Filtering
    # Apply a zero-phase bandpass filter. 
    # Lower bound (1.0 Hz) removes slow baseline drift (e.g., sweat artifact).
    # Upper bound (45.0 Hz) removes high-frequency muscle noise (EMG) and 50/60Hz powerline noise.
    # Phase='zero' is critical: causal filters shift the phase, which geometrically deforms the attractor.
    raw.filter(l_freq=1.0, h_freq=45.0, fir_design='firwin', phase='zero', verbose=False)
    
    # 4. Extract the target spatial vector using the modern MNE API
    raw.pick(picks=[target_channel])
    
    # Isolate the temporal epoch of interest
    raw.crop(tmin=tmin, tmax=tmax)
    
    # Extract the underlying NumPy array, sampling frequency, and temporal vector
    data, times = raw[:]
    sampling_freq = raw.info['sfreq']
    
    return data[0, :], sampling_freq, times, target_channel


def compute_phase_space_vectorized(x: np.ndarray, tau: int, dim: int = 3) -> np.ndarray:
    """
    Computes the Takens' delay embedding using advanced NumPy stride tricks.
    
    Instead of iterating and copying memory, this creates an O(1) sliding window 
    view directly into the memory buffer of the original 1D time series.
    
    Parameters:
        x (np.ndarray): 1D continuous scalar observation function.
        tau (int): Delay parameter in discrete samples.
        dim (int): Embedding dimension (m >= 2d + 1).
        
    Returns:
        np.ndarray: Matrix of shape (M, dim) representing the embedded manifold.
    """
    N = len(x)
    M = N - (dim - 1) * tau
    
    if M <= 0:
        raise ValueError("Time series is too short for the selected dimension and temporal delay.")
    
    # Calculate bytes per element to safely manipulate memory pointers
    itemsize = x.itemsize
    
    # Define the shape (rows, columns) and memory strides (bytes to next row, bytes to next column)
    shape = (M, dim)
    strides = (itemsize, itemsize * tau)
    
    # Create the strided view (computationally instantaneous regardless of signal length)
    embedded_manifold = as_strided(x, shape=shape, strides=strides)
    
    return embedded_manifold


def compute_morlet_energy(x: np.ndarray, fs: float, freqs: np.ndarray, num_points: int) -> np.ndarray:
    """
    Calculates the integrated instantaneous spectral energy using the Complex Morlet wavelet.
    
    Parameters:
        x (np.ndarray): 1D continuous scalar observation function.
        fs (float): Sampling frequency.
        freqs (np.ndarray): Array of target frequencies to scan.
        num_points (int): The number of points in the reconstructed manifold (M) to align the vectors.
        
    Returns:
        np.ndarray: 1D array of normalized gradient energy.
    """
    # Convert real frequencies to wavelet scales based on the Morlet mother wavelet signature
    scales = pywt.frequency2scale('morl', freqs / fs)
    
    # Compute the Continuous Wavelet Transform (CWT)
    cwt_matrix, _ = pywt.cwt(x, scales, 'morl', sampling_period=1/fs)
    
    # Integrate energy across all computed frequency bands (Equivalent to L2 Norm)
    # Taking the absolute square of the complex coefficients yields instantaneous power
    energy = np.sum(np.abs(cwt_matrix)**2, axis=0)
    
    # Normalize the energy vector to [0, 1] via Min-Max scaling for colormap projection
    energy_normalized = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
    
    # Slice the energy vector to match the exact length of the Takens manifold
    return energy_normalized[:num_points]

def compute_optimal_tau(x: np.ndarray, fs: float, max_lag_sec: float = 0.5) -> int:
    """
    Computes the optimal embedding lag (tau) by finding the first zero-crossing
    of the autocorrelation function.
    
    Parameters:
        x (np.ndarray): 1D continuous scalar observation function.
        fs (float): Sampling frequency in Hz.
        max_lag_sec (float): Maximum temporal horizon to search for a root.
        
    Returns:
        int: Optimal delay in discrete sample indices.
    """
    # Remove DC offset to ensure a zero-mean process for accurate autocorrelation
    x_centered = x - np.mean(x)
    max_lag_samples = int(max_lag_sec * fs)
    
    # Vectorized computation of autocorrelation using the Wiener-Khinchin theorem via FFT
    # This is O(N log N), significantly faster than time-domain O(N^2) correlation
    acf = signal.correlate(x_centered, x_centered, mode='full', method='fft')
    
    # Extract only the positive temporal lags
    acf = acf[len(acf)//2:] 
    acf = acf[:max_lag_samples]
    
    # Normalize the autocorrelation function [-1, 1]
    acf_normalized = acf / acf[0]
    
    # Detect the first zero crossing using differences in sign bits
    # np.sign returns -1, 0, or 1. np.diff finds where the sign changes.
    zero_crossings = np.where(np.diff(np.sign(acf_normalized)))[0]
    
    if len(zero_crossings) > 0:
        optimal_tau = zero_crossings[0]
        print(f"[Math] Optimal Tau calculated via Autocorrelation root-finding: {optimal_tau} samples ({(optimal_tau/fs)*1000:.1f} ms).")
        return optimal_tau
    else:
        # Fallback heuristic if no root is found within the horizon
        fallback_tau = int(fs * 0.05)
        print(f"[Warning] No zero-crossing found. Defaulting to: {fallback_tau} samples.")
        return fallback_tau
def render_topological_attractor(E: np.ndarray, color_grad: np.ndarray, channel_name: str):
    """
    Renders the state-space manifold using hardware-accelerated WebGL via Plotly.
    
    Parameters:
        E (np.ndarray): The embedded manifold matrix.
        color_grad (np.ndarray): The normalized 1D wavelet energy gradient.
        channel_name (str): The clinical label of the spatial node.
    """
    # Construct the 3D Scatter object
    fig = go.Figure(data=[go.Scatter3d(
        x=E[:, 0], 
        y=E[:, 1], 
        z=E[:, 2],
        mode='lines',
        line=dict(
            width=2.5,
            color=color_grad,
            colorscale='Turbo', # 'Turbo' provides optimal perceptual uniformity for clinical data
            showscale=True,
            colorbar=dict(title="L2 Wavelet Energy (Normalized)", thickness=15)
        )
    )])

    # Configure the clinical visualization layout (Dark mode for contrast)
    fig.update_layout(
        title=dict(
            text=f'Non-Linear Dynamics & Topological Energy - Node: {channel_name}', 
            font=dict(size=18)
        ),
        scene=dict(
            xaxis_title='x(t)',
            yaxis_title='x(t + τ)',
            zaxis_title='x(t + 2τ)',
            bgcolor='rgb(10, 10, 10)',
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            zaxis=dict(showgrid=False, zeroline=False)
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    # Execute WebGL rendering pipeline
    fig.show()
def render_synchronized_dashboard(t_vector: np.ndarray, x: np.ndarray, E: np.ndarray, 
                                  color_grad: np.ndarray, channel_name: str, tau: int):
    """
    Renders a synchronized, dual-pane clinical visualization.
    Proves the temporal isomorphism between the 1D signal and the 3D topological manifold.
    """
    # The manifold E has length M. We must truncate the 1D time and signal vectors 
    # to match the length of the embedded manifold to maintain strict bijection.
    M = E.shape[0]
    t_sync = t_vector[:M]
    x_sync = x[:M]
    
    # Initialize a 1x2 Subplot matrix with mixed WebGL projection types
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=(
            f"1D Observation: {channel_name} (Color = Transient Spectral Power)", 
            f"3D State-Space Reconstruction (Delay = {tau} samples)"
        ),
        horizontal_spacing=0.05
    )

    # --- PANE 1: 1D Time Series Colored by Wavelet Energy ---
    # We must plot the 1D line as a series of connected segments to apply a color gradient
    # Plotly Scatter (2D) only accepts color gradients on markers, not continuous lines directly,
    # so we utilize a high-density marker approach that visually fuses into a line.
    fig.add_trace(
        go.Scatter(
            x=t_sync, 
            y=x_sync,
            mode='lines+markers',
            marker=dict(
                size=3,
                color=color_grad,
                colorscale='Turbo',
                showscale=False # Hide scale here, show it on the 3D plot
            ),
            line=dict(color='rgba(255,255,255,0.2)', width=1),
            name="EEG Signal"
        ),
        row=1, col=1
    )

    # --- PANE 2: 3D Topological Manifold ---
    fig.add_trace(
        go.Scatter3d(
            x=E[:, 0], y=E[:, 1], z=E[:, 2],
            mode='lines',
            line=dict(
                width=3,
                color=color_grad,
                colorscale='Turbo',
                cmin=0, cmax=1,
                colorbar=dict(title="L2 Spectral Energy", thickness=15, x=1.05)
            ),
            name="Phase Space"
        ),
        row=1, col=2
    )

    # --- HCI & Layout Configuration ---
    fig.update_layout(
        template="plotly_dark",
        title_text=f"Clinical Dynamics Dashboard - Node: {channel_name}",
        title_font_size=20,
        height=800,
        showlegend=False,
        margin=dict(l=20, r=20, b=20, t=80)
    )

    # Standardize axes for clinical review
    fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_yaxes(title_text="Potential (μV)", row=1, col=1)
    
    fig.update_scenes(
        xaxis_title='x(t)',
        yaxis_title='x(t + τ)',
        zaxis_title='x(t + 2τ)',
        bgcolor='rgb(15, 15, 15)'
    )

    fig.show()

# --- System Execution ---
if __name__ == "__main__":
    
    # Hardcoded parameters for the execution pipeline
    # Update the path to match the local directory structure of your repository
    FILEPATH_EEG = "./chb01_01.edf" 
    TARGET_CHANNEL = "Cz"
    
    try:
        print("1. Ingesting clinical data and verifying spatial basis...")
        signal_array, fs, time_vector, actual_channel = load_and_preprocess_eeg(
            FILEPATH_EEG, TARGET_CHANNEL, tmin=0, tmax=10
        )
        
        print(f" -> Extraction complete: {len(signal_array)} samples at {fs} Hz from Node '{actual_channel}'.")
        
        # 2. Mathematical Parameterization
        # Tau estimation. In a rigorous clinical setup, use the first zero of the autocorrelation function.
        # Here we approximate ~50ms delay, which is highly effective for cortical phase-locking capture.
        tau_delay = int(fs * 0.05) 
        
        print(f"2. Computing topological embedding (tau = {tau_delay} samples)...")
        manifold_matrix = compute_phase_space_vectorized(signal_array, tau=tau_delay, dim=3)
        
        print("3. Computing non-stationary Wavelet energy gradient...")
        # Define a frequency resolution array (1 Hz to 45 Hz)
        frequency_bands = np.linspace(1, 45, 40)
        energy_gradient = compute_morlet_energy(
            signal_array, 
            fs, 
            frequency_bands, 
            num_points=manifold_matrix.shape[0]
        )
        
        print("4. Executing WebGL visualization pipeline...")
        render_topological_attractor(manifold_matrix, energy_gradient, actual_channel)
        
    except FileNotFoundError:
        print(f"\n[Error] The file '{FILEPATH_EEG}' was not found.")
    except Exception as e:
        print(f"\n[Algorithm Failure] A mathematical or pipeline error occurred: {e}")
