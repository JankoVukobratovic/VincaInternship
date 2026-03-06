import math
import os
from functools import lru_cache
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

# Strictly White -> Color or Black -> Color
ELEMENT_MAP = {
   #"S":  {"name": "Sulphur", "kev": 2.31, "cmap": "YlOrBr"}, # White-Yellow-Brown
    "Ca": {"name": "Kalcium Ca", "kev": 3.69, "cmap": "gray"},    # Black-White
    "Ti": {"name": "Titanium Ti", "kev": 4.51, "cmap": "hot"},     # Black-Red-White
    "Fe": {"name": "Gvozdje Fe", "kev": 6.40, "cmap": "Reds"},      # White-Red
   #"Ni": {"name": "Nickel", "kev": 7.47, "cmap": "RdPu"},    # White-Pink
    "Cu": {"name": "Bakar Cu", "kev": 8.04, "cmap": "Greens"},    # White-Green
   #"As": {"name": "Arsenic", "kev": 10.54, "cmap": "OrRd"},    # White-Orange
    "Pb": {"name": "Olovo Lb", "kev": 10.55, "cmap": "Purples"}, # White-Purple
   #"Sn": {"name": "Tin", "kev": 25.27, "cmap": "Blues"}      # White-Blue
   # "Zn": {"name": "Zinc", "kev" : 8.5, "cmap": "RdPu"}
}

ELEMENT_DIFF_MAP = {
    key: {
        "name": f"Δ {val['name']}",
        "kev": val['kev'],
        "cmap": "seismic" # Diverging: Blue (negative), White (zero), Red (positive)
    } for key, val in ELEMENT_MAP.items()
}

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

def process_image_data(folder_path, width=120, height=60, elemenent_map=None):
    if elemenent_map is None:
        elemenent_map = ELEMENT_MAP
    total_points = width * height
    element_keys = list(elemenent_map.keys())

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
            area = get_dynamic_area(energy_axis, cps, elemenent_map[key]['kev'])
            matrix_3d[idx, row, col] = area

        if i%100 == 0:
            print(f"Processed {i}...")

    return matrix_3d, element_keys, width, height

@lru_cache(maxsize=None)
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

def render_element_grid(matrix_3d: np.ndarray[Any, np.dtype[np.float64]], element_keys, element_map, width = 120, height = 60, savename ="", figname =""):
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
        cbar.outline.linewidth = 0.5
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

def full_thing(folder_path, save_path, w=120, h=60):
    cube, keys, w, h = process_image_data(folder_path, w, h)
    render_element_grid(cube, keys, ELEMENT_MAP, w, h, save_path)
    return cube, keys

    num_scans = len(strips)
    num_elements = len(element_keys)

    fig, axes = plt.subplots(num_elements, num_scans,
                             figsize=(4 * num_scans + 2, 3 * num_elements),
                             dpi=100, squeeze=False)


    for row_idx, key in enumerate(element_keys):
        all_data_for_this_element = [s[row_idx] for s in strips[:-1]]
        global_min = np.nanmin(all_data_for_this_element)
        global_max = np.nanmax(all_data_for_this_element)

        for col_idx in range(num_scans):
            ax = axes[row_idx, col_idx]

            # Extract current data and map
            data_cube = strips[col_idx]
            current_map = element_maps[col_idx]

            config = current_map[key]
            img_data = data_cube[row_idx]

            # Get height and width from the specific matrix slice
            h, w = img_data.shape


            #todo remove hardcoded 2
            if col_idx ==  2:
                vmin, vmax = -abs_max, abs_max
            else:
                vmin, vmax = global_min, global_max

            # Render the image
            im = ax.imshow(img_data, cmap=config['cmap'], aspect='auto',
                           origin='upper', interpolation='nearest',
                           vmin = vmin, vmax = vmax)

            # Titles only on the top row to identify the "Scan/Column"
            if row_idx == 0:
                ax.set_title(strip_names[col_idx], fontsize=14, fontweight='bold')

            # Label rows with the element name on the first column
            if col_idx == 0:
                ax.set_ylabel(f"{config['name']}\n({config['kev']} keV)",
                              fontsize=12, fontweight='bold', rotation=0, labelpad=40, ha='right')

            # Add colorbar for each subplot
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

            # Basic tick formatting
            ax.set_xticks([0, w // 2, w])
            ax.set_yticks([0, h // 2, h])
            ax.tick_params(labelsize=7)

    plt.tight_layout(rect=(0.05, 0.03, 1, 0.95))

    if figname:
        fig.suptitle(figname, fontsize=22, fontweight='bold', y=0.99)

    if savename:
        plt.savefig(savename)
        print(f"Comparison figure saved to: {savename}")

    plt.show(block=False)



if __name__ == "__main__":
    cube_prova1_10264, keys = full_thing("Resources/aurora-antico1-prova1/10264", "rezultati/prova1/10264.png")
    cube_prova2_10264, _ = full_thing("Resources/aurora-antico1-prova2/10264", "rezultati/prova2/10264.png")
    cube_prova1_19511, _ = full_thing("Resources/aurora-antico1-prova1/19511", "rezultati/prova1/19511.png")
    cube_prova2_19511, _ = full_thing("Resources/aurora-antico1-prova2/19511", "rezultati/prova2/19511.png")

    render_comparisons([cube_prova1_19511, cube_prova1_10264, cube_prova1_10264-cube_prova1_19511],
                       keys,
                       [ELEMENT_MAP, ELEMENT_MAP, ELEMENT_DIFF_MAP],
                       savename = "rezultati/best_j/prova1_10264-19511",
                       figname = "prova 1",
                       strip_names= ["19511", "10264", "10265 - 19511"]
                       )

    render_comparisons([cube_prova2_19511, cube_prova2_10264, cube_prova2_10264-cube_prova2_19511],
                       keys,
                       [ELEMENT_MAP, ELEMENT_MAP, ELEMENT_DIFF_MAP],
                       savename = "rezultati/best_j/prova2_10264-19511",
                       figname = "prova 2",
                       strip_names= ["19511", "10264", "10265 - 19511"]
                       )

    render_comparisons([cube_prova1_19511, cube_prova2_19511, cube_prova1_19511-cube_prova2_19511],
                       keys,
                       [ELEMENT_MAP, ELEMENT_MAP, ELEMENT_DIFF_MAP],
                       savename = "rezultati/best_j/prova1-prova2-19511",
                       figname = "19511",
                       strip_names= ["prova1", "prova2", "prova1-prova2"]
                       )

    render_comparisons([cube_prova1_10264, cube_prova2_10264, cube_prova1_10264-cube_prova2_10264],
                       keys,
                       [ELEMENT_MAP, ELEMENT_MAP, ELEMENT_DIFF_MAP],
                       savename = "rezultati/best_j/prova1-prova2-10264",
                       figname = "10264",
                       strip_names= ["prova1", "prova2", "prova1-prova2"]
                       )




