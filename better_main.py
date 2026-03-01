import math
import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

# Strictly White -> Color or Black -> Color
ELEMENT_MAP = {
    "S":  {"name": "Sulphur", "kev": 2.31, "cmap": "YlOrBr"}, # White-Yellow-Brown
    "Ca": {"name": "Calcium", "kev": 3.69, "cmap": "gray"},    # Black-White
    "Ti": {"name": "Titanium", "kev": 4.51, "cmap": "hot"},     # Black-Red-White
    "Fe": {"name": "Iron", "kev": 6.40, "cmap": "Reds"},      # White-Red
    "Ni": {"name": "Nickel", "kev": 7.47, "cmap": "RdPu"},    # White-Pink
    "Cu": {"name": "Copper", "kev": 8.04, "cmap": "Greens"},    # White-Green
    "As": {"name": "Arsenic", "kev": 10.54, "cmap": "OrRd"},    # White-Orange
    "Pb": {"name": "Lead Lb", "kev": 12.61, "cmap": "Purples"}, # White-Purple
    "Sn": {"name": "Tin", "kev": 25.27, "cmap": "Blues"}      # White-Blue
}

ELEMENT_DIFF_MAP = {
    key: {
        "name": f"Δ {val['name']}",
        "kev": val['kev'],
        "cmap": "seismic" # Diverging: Blue (negative), White (zero), Red (positive)
    } for key, val in ELEMENT_MAP.items()
}


def get_calibrated_axis(num_channels):
    cal_pts = np.array([[219, 6.4], [278, 8], [363, 10.5], [436, 12.6], [869, 25.3]])
    slope, intercept, _, _, _ = linregress(cal_pts[:, 0], cal_pts[:, 1])
    return (np.arange(num_channels) * slope) + intercept


def get_dynamic_area(energy_axis, cps, target_kev):
    idx = int(np.argmin(np.abs(energy_axis - target_kev)))
    #smoothed = gaussian_filter1d(cps, sigma=1.5)

    l = idx
    while l > 0 and cps[l - 1] < cps[l]:
        l -= 1
        if energy_axis[idx] - energy_axis[l] > 1.0:
            break

    r = idx
    while r < cps.shape[0] - 1 and cps[r + 1] < cps[r]:
        r += 1
        if energy_axis[r] - energy_axis[idx] > 1.0:
            break

    # Using the updated trapezoid function
    return np.trapezoid(cps[l:r + 1], x=energy_axis[l:r + 1])

def process_image_data(folder_path, width=120, height=60):
    total_points = width * height
    element_keys = list(ELEMENT_MAP.keys())

    # NEW: Initialize as a 3D matrix (Channels, Height, Width)
    matrix_3d = np.zeros((len(element_keys), height, width))

    cal_pts = np.array([[219, 6.4], [278, 8], [363, 10.5], [436, 12.6], [869, 25.3]])
    slope, intercept, _, _, _ = linregress(cal_pts[:, 0], cal_pts[:, 1])

    for i in range(1, total_points + 1):
        filename = f"None_{i}.mca"
        path = os.path.join(folder_path, filename)

        if not os.path.exists(path):
            # Useful for debugging missing steps in the scan
            print(f"Warning: Missing file {filename}")
            continue

        data = parse_mca_file(path)
        energy_axis = (np.arange(len(data['counts'])) * slope) + intercept
        cps = data['counts'] / data['time']
        row = (i - 1) // width
        col = (i - 1) % width

        for idx, key in enumerate(element_keys):
            area = get_dynamic_area(energy_axis, cps, ELEMENT_MAP[key]['kev'])
            matrix_3d[idx, row, col] = area


        print(f"Processed {i}...")

    return matrix_3d, element_keys, width, height


def parse_mca_file(filepath):
    """
    Parses a .mca file, extracting metadata and the raw counts.
    Stops reading data when the <<END>> tag is encountered.
    """
    metadata = {}
    counts = []
    in_data_section = False

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue

            if line == "<<DATA>>":
                in_data_section = True
                continue
            if line == "<<END>>":
                in_data_section = False
                break

            if in_data_section:
                try:
                    counts.append(int(line))
                except ValueError:
                    continue
            else:
                if " - " in line:
                    parts = line.split(" - ", 1)
                    metadata[parts[0].strip()] = parts[1].strip()

    return {
        "counts": np.array(counts, dtype=np.float64),
        "time": float(metadata.get("REAL_TIME", 1.0))
    }

def render_element_grid(matrix_3d: np.ndarray[any, np.dtype[np.float64]], element_keys ,element_map, width = 120, height = 60, savename ="", figname =""):
    num_elements = len(element_keys)
    cols = 3
    rows = math.ceil(num_elements / cols)

    # We set a wide figure size to accommodate the 120-pixel width
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows + 2), dpi=100)
    axes = axes.flatten()
    idx = 0
    for idx, key in enumerate(element_keys):
        config = element_map[key]
        img = matrix_3d[idx]

        # aspect='auto' prevents the 'skinny strip' syndrome
        # interpolation='nearest' keeps the 120x60 pixels crisp
        im = axes[idx].imshow(img, cmap=config['cmap'], aspect='auto', origin='upper', interpolation='nearest')

        axes[idx].set_title(f"{config['name']} ({config['kev']} keV)", fontsize=12, fontweight='bold')

        # fraction=0.03 makes the colorbar thin
        # pad=0.02 keeps it close to the image
        cbar = plt.colorbar(im, ax=axes[idx], fraction=0.03, pad=0.02)
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=9)

        axes[idx].set_xticks([0, width // 2, width])
        axes[idx].set_yticks([0, height // 2, height])
        axes[idx].tick_params(labelsize=8)

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout(rect=(0, 0.03, 1, 0.95), h_pad=3.0, w_pad=2.0)
    if figname:
        fig.suptitle(figname, fontsize=20, fontweight='bold', y=0.98)

    if savename != "":
        plt.savefig(savename)
        print(f"Figure saved to: {savename}")
    plt.show(block = False)



cube1, keys, w, h = process_image_data("Resources/aurora-antico1-prova1/10264", 120, 60)
cube2, _, _, _    = process_image_data("Resources/aurora-antico1-prova1/19511", 120, 60)

render_element_grid(cube1, keys, ELEMENT_MAP, width=w, height=h,
                    savename="rezultati/dynamic_windowed/prov1_10264.png", figname="Scan 10264")

render_element_grid(cube2, keys, ELEMENT_MAP, width=w, height=h,
                    savename="rezultati/dynamic_windowed/prov2_19511.png", figname="Scan 19511")

render_element_grid(cube1 - cube2, keys, ELEMENT_DIFF_MAP, width=w, height=h,
                    savename="rezultati/dynamic_windowed/diff_10265-19511", figname="Difference 10264 - 19511")

render_element_grid(cube1 + cube2 / 2, keys, ELEMENT_MAP, width=2, height=h, savename = "rezultati/dynamic_windowed/avg_prova1", figname = "AVG")