import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from scipy.stats import linregress

def parse_mca_file(filepath):
    """
    Parses a .mca file to extract metadata, calibration points, and spectral data.
    """
    metadata = {}
    calibration_points = []
    counts = []

    section = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue

            # Section Switching
            if line == "<<PMCA SPECTRUM>>": continue
            if line == "<<CALIBRATION>>":
                section = "CALIBRATION"
                continue
            if line == "<<DATA>>":
                section = "DATA"
                continue
            if line == "<<END>>":
                break

            # Data Parsing Logic
            if section == "DATA":
                counts.append(int(line))
            elif section == "CALIBRATION":
                if line.startswith("LABEL"): continue
                # Store (Channel, Energy) pairs
                parts = line.split()
                if len(parts) == 2:
                    calibration_points.append([float(parts[0]), float(parts[1])])
            else:
                # Standard Metadata (Key - Value)
                if " - " in line:
                    key, val = line.split(" - ", 1)
                    metadata[key.strip()] = val.strip()

    return {
        "metadata": metadata,
        "calibration": np.array(calibration_points),
        "counts": np.array(counts, dtype=np.int32)
    }


def get_energy_axis(calibration_pts, num_channels):
    # Perform linear regression on the calibration points provided in the file
    channels = calibration_pts[:, 0]
    energies = calibration_pts[:, 1]
    slope, intercept, _, _, _ = linregress(channels, energies)

    # Generate the full energy scale for the heatmap X-axis
    return slope * np.arange(num_channels) + intercept




def plot_single_mca(file_path):
    # 1. Parse the file (using the parser from the previous step)
    data_dict = parse_mca_file(file_path)
    counts = data_dict['counts']
    cal_pts = data_dict['calibration']
    real_time = float(data_dict['metadata'].get('REAL_TIME', 1.0))

    # 2. Convert to Counts Per Second (CPS)
    # This normalizes for different acquisition durations
    cps = counts / real_time

    # 3. Create Energy Axis
    # Using the points: (219, 6.4), (278, 8), etc.
    channels = cal_pts[:, 0]
    energies = cal_pts[:, 1]
    slope, intercept, r_value, _, _ = linregress(channels, energies)

    energy_axis = (np.arange(len(counts)) * slope) + intercept

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    plt.step(energy_axis, cps, where='mid', color='midnightblue', lw=1)

    # Use Log Scale for the Y-axis (Intensity)
    plt.yscale('log')

    plt.title(f"Spectrum: {data_dict['metadata'].get('TAG', 'Unknown')}")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Counts Per Second (Log Scale)")
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Annotate Calibration Fit Quality
    plt.annotate(f"$R^2 = {r_value ** 2:.5f}$", xy=(0.05, 0.95), xycoords='axes fraction')

    plt.show()


def aggregate_data(root_dir, energy_bins=4096, energy_max=30.0):
    all_spectra = []
    file_labels = []

    # Define our common X-axis
    master_energy = np.linspace(0, energy_max, energy_bins)

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".mca"):
                path = os.path.join(subdir, file)
                try:
                    data = parse_mca_file(path)

                    # 1. Get current file's energy calibration
                    cal = data['calibration']
                    slope, intercept, _, _, _ = linregress(cal[:, 0], cal[:, 1])
                    current_energy = (np.arange(len(data['counts'])) * slope) + intercept

                    # 2. Normalize to CPS
                    real_time = float(data['metadata'].get('REAL_TIME', 1.0))
                    cps = data['counts'] / (real_time if real_time > 0 else 1.0)

                    # 3. Interpolate onto Master Energy Axis
                    # This ensures all folders (10264, 19511) align perfectly
                    f_interp = interp1d(current_energy, cps, bounds_error=False, fill_value=0)
                    aligned_cps = f_interp(master_energy)

                    all_spectra.append(aligned_cps)
                    # Label format: "Folder/Filename"
                    file_labels.append(f"{os.path.basename(subdir)}/{file}")
                    print(f"Appended {file}")
                except Exception as e:
                    print(f"Skipping {file}: {e}")

    return np.array(all_spectra), master_energy, file_labels

def plot_heatmap_plt(matrix, energy_axis, labels):
    plt.figure(figsize=(12, 8))

    # LogNorm handles the logarithmic intensity scaling automatically
    img = plt.imshow(matrix,
                     aspect='auto',
                     extent=(energy_axis[0], energy_axis[-1], 0, len(labels)),
                     cmap='viridis',
                     norm=LogNorm(vmin=1, vmax=matrix.max()))

    plt.colorbar(img, label='Counts Per Second (Log Scale)')
    plt.xlabel('Energy (keV)')
    plt.ylabel('File Index / Sample')
    plt.title('Spectral Heatmap')

    # Optional: If you want to see specific file names on the Y-axis
    # (Only recommended if you have < 50 files, otherwise it's messy)
    # plt.yticks(range(len(labels)), labels, fontsize=6)

    plt.tight_layout()
    plt.show()

import plotly.express as px
import plotly.io as pio
from matplotlib.colors import LogNorm


def plot_heatmap_px(data_matrix, x_axis, y_labels):
    data_matrix_log = np.log10(np.where(data_matrix <= 0, 1e-10, data_matrix))

    fig = px.imshow(
        data_matrix_log,
        x=x_axis,  # Map Energy values to X-axis
        y=y_labels,  # Map File IDs to Y-axis
        aspect='auto',  # Stretch to fit
        color_continuous_scale='Magma',  # Use a scientific color map
        labels=dict(x="Energy (keV)", y="Measurement Sequence (ID)", color="Log10(CPS)"),
        title="Interactive Spectral Heatmap (Folded Data)"
    )

    fig.update_layout(
        xaxis_title_text="Energy (keV)",
        yaxis_title_text="Measurement Sequence (ID)",
        coloraxis_colorbar=dict(title="Log10(CPS)")
    )
    fig.show()


PATH = 'resources/19511/'

data_matrix, x_axis, y_labels = aggregate_data(PATH, energy_bins=2048, energy_max=30.0)

plot_heatmap_plt(data_matrix, x_axis, y_labels)
