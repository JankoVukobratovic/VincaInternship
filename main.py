import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from scipy.stats import linregress


def parse_mca_file(filepath):
    """
    Parsira .mca fajl i izvlači metapodatke, kalibracione tačke i spektralne podatke.
    Vraća rečnik sa ključevima: 'metadata', 'calibration', 'counts'.
    """
    metadata = {}            # Rečnik za metapodatke (TAG, REAL_TIME, itd.)
    calibration_points = []  # Lista kalibracionih tačaka (kanal, energija)
    counts = []              # Lista broja odbroja po kanalu

    section = None           # Trenutna sekcija u fajlu (CALIBRATION / DATA)

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue

            # Prepoznavanje i prelaz između sekcija fajla
            if line == "<<PMCA SPECTRUM>>": continue
            if line == "<<CALIBRATION>>":
                section = "CALIBRATION"
                continue
            if line == "<<DATA>>":
                section = "DATA"
                continue
            if line == "<<END>>":
                break

            # Logika parsiranja podataka u zavisnosti od trenutne sekcije
            if section == "DATA":
                # Svaki red u DATA sekciji je broj odbroja jednog kanala
                counts.append(int(line))
            elif section == "CALIBRATION":
                if line.startswith("LABEL"): continue
                # Čuvamo parove (kanal, energija) za linearnu kalibraciju
                parts = line.split()
                if len(parts) == 2:
                    calibration_points.append([float(parts[0]), float(parts[1])])
            else:
                # Metapodaci u formatu "KLJUČ - VREDNOST"
                if " - " in line:
                    key, val = line.split(" - ", 1)
                    metadata[key.strip()] = val.strip()

    return {
        "metadata": metadata,
        "calibration": np.array(calibration_points),
        "counts": np.array(counts, dtype=np.int32)
    }


def get_energy_axis(calibration_pts, num_channels):
    # Linearna regresija na kalibracionim tačkama iz fajla
    channels = calibration_pts[:, 0]
    energies = calibration_pts[:, 1]
    slope, intercept, _, _, _ = linregress(channels, energies)

    # Generišemo kompletnu energetsku osu za X-osu toplotne mape
    return slope * np.arange(num_channels) + intercept


def plot_single_mca(file_path):
    # 1. Parsiranje fajla pomoću gore definisanog parsera
    data_dict = parse_mca_file(file_path)
    counts = data_dict['counts']
    cal_pts = data_dict['calibration']
    real_time = float(data_dict['metadata'].get('REAL_TIME', 1.0))

    # 2. Pretvaranje u Odbroje po Sekundi (CPS)
    # Ovo normalizuje rezultate za različita trajanja akvizicije
    cps = counts / real_time

    # 3. Kreiranje energetske ose
    # Koriste se kalibracione tačke npr. (219, 6.4), (278, 8), itd.
    channels = cal_pts[:, 0]
    energies = cal_pts[:, 1]
    slope, intercept, r_value, _, _ = linregress(channels, energies)

    energy_axis = (np.arange(len(counts)) * slope) + intercept

    # 4. Crtanje spektra
    plt.figure(figsize=(10, 6))
    plt.step(energy_axis, cps, where='mid', color='midnightblue', lw=1)

    # Logaritamska skala za Y-osu (intenzitet)
    plt.yscale('log')

    plt.title(f"Spektar: {data_dict['metadata'].get('TAG', 'Nepoznato')}")
    plt.xlabel("Energija (keV)")
    plt.ylabel("Odbroji po sekundi (logaritamska skala)")
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Anotacija kvaliteta kalibracione prave (R²)
    plt.annotate(f"$R^2 = {r_value ** 2:.5f}$", xy=(0.05, 0.95), xycoords='axes fraction')

    plt.show()


def aggregate_data(root_dir, energy_bins=4096, energy_max=30.0):
    all_spectra = []   # Lista svih spektara interpolovanih na zajedničku osu
    file_labels = []   # Oznake fajlova za Y-osu

    # Definišemo zajedničku referentnu X-osu (energetsku osu) za sve fajlove
    master_energy = np.linspace(0, energy_max, energy_bins)

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".mca"):
                path = os.path.join(subdir, file)
                try:
                    data = parse_mca_file(path)

                    # 1. Dohvatamo energetsku kalibraciju za trenutni fajl
                    cal = data['calibration']
                    slope, intercept, _, _, _ = linregress(cal[:, 0], cal[:, 1])
                    current_energy = (np.arange(len(data['counts'])) * slope) + intercept

                    # 2. Normalizacija na CPS (odbroji po sekundi)
                    real_time = float(data['metadata'].get('REAL_TIME', 1.0))
                    cps = data['counts'] / (real_time if real_time > 0 else 1.0)

                    # 3. Interpolacija na zajedničku energetsku osu
                    # Ovo osigurava da se svi folderi (10264, 19511) savršeno poravnaju
                    f_interp = interp1d(current_energy, cps, bounds_error=False, fill_value=0)
                    aligned_cps = f_interp(master_energy)

                    all_spectra.append(aligned_cps)
                    # Format oznake: "Folder/ImeFajla"
                    file_labels.append(f"{os.path.basename(subdir)}/{file}")
                    print(f"Dodat {file}")
                except Exception as e:
                    print(f"Preskačem {file}: {e}")

    return np.array(all_spectra), master_energy, file_labels


def plot_heatmap_plt(matrix, energy_axis, labels):
    plt.figure(figsize=(12, 8))

    # LogNorm automatski primenjuje logaritamsko skaliranje intenziteta
    img = plt.imshow(matrix,
                     aspect='auto',
                     extent=(energy_axis[0], energy_axis[-1], 0, len(labels)),
                     cmap='viridis',
                     norm=LogNorm(vmin=1, vmax=matrix.max()))

    plt.colorbar(img, label='Odbroji po sekundi (logaritamska skala)')
    plt.xlabel('Energija (keV)')
    plt.ylabel('Indeks fajla / Uzorak')
    plt.title('Spektralna toplotna mapa')

    # Opciono: Prikaz konkretnih naziva fajlova na Y-osi
    # (Preporučuje se samo ako ima < 50 fajlova, inače je nečitljivo)
    # plt.yticks(range(len(labels)), labels, fontsize=6)

    plt.tight_layout()
    plt.show()


import plotly.express as px
import plotly.io as pio
from matplotlib.colors import LogNorm


def plot_heatmap_px(data_matrix, x_axis, y_labels):
    # Primena log10 transformacije – vrednosti <= 0 zamenjujemo malim brojem da izbegnemo log(0)
    data_matrix_log = np.log10(np.where(data_matrix <= 0, 1e-10, data_matrix))

    fig = px.imshow(
        data_matrix_log,
        x=x_axis,      # Energetske vrednosti na X-osi
        y=y_labels,    # ID merenja na Y-osi
        aspect='auto', # Razvlačenje da ispuni prostor
        color_continuous_scale='Magma',  # Naučna paleta boja
        labels=dict(x="Energija (keV)", y="Redosled merenja (ID)", color="Log10(CPS)"),
        title="Interaktivna spektralna toplotna mapa (složeni podaci)"
    )

    fig.update_layout(
        xaxis_title_text="Energija (keV)",
        yaxis_title_text="Redosled merenja (ID)",
        coloraxis_colorbar=dict(title="Log10(CPS)")
    )
    fig.show()


# ─── Putanja do foldera sa podacima ───────────────────────────────────────────
PATH = '/Users/aleksandra/Downloads/VincaInternship-master/aurora-antico1-prova1/10264'

# Prikupljamo sve spektre i interpolujemo ih na zajedničku energetsku osu (2048 binova, 0–30 keV)
data_matrix, x_axis, y_labels = aggregate_data(PATH, energy_bins=2048, energy_max=30.0)

# Crtamo toplotnu mapu svih spektara
plot_heatmap_plt(data_matrix, x_axis, y_labels)
