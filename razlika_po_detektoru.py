"""
razlika_po_detektoru.py
────────────────────────────────────────────────────────────────────
Elementi: Ca, Ti, Fe, Cu, Pb_La  (bez Sn)
Detektori: 10264, 19511

Izlaz:
  rezultati/razlika_po_detektoru/mape_10264.png        – mape prova1+prova2 za det. 10264
  rezultati/razlika_po_detektoru/mape_19511.png        – mape prova1+prova2 za det. 19511
  rezultati/razlika_po_detektoru/razlike_svi_elementi.png – razlike prova1−prova2
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ─── Putanje ──────────────────────────────────────────────────────
NPY1  = os.path.join('rezultati', '_npy_cache', 'prova1')
NPY2  = os.path.join('rezultati', '_npy_cache', 'prova2')
IZLAZ = os.path.join('rezultati', 'razlika_po_detektoru')
os.makedirs(IZLAZ, exist_ok=True)

DETEKTORI = ['10264', '19511']
ELEMENTI  = ['Ca', 'Ti', 'Fe', 'Cu', 'Pb_La']

NASLOVI = {
    'Ca':    'Ca – Kalcijum',
    'Ti':    'Ti – Titanijum',
    'Fe':    'Fe – Gvožđe',
    'Cu':    'Cu – Bakar',
    'Pb_La': 'Pb – Olovo Lα',
}

PALETE = {
    'Ca':    'gray',
    'Ti':    'hot',
    'Fe':    'Reds',
    'Cu':    'Greens',
    'Pb_La': 'Purples',
}

# ─── Učitavanje ───────────────────────────────────────────────────
mape = {}
for det in DETEKTORI:
    for el in ELEMENTI:
        mape[(det, 'prova1', el)] = np.load(os.path.join(NPY1, f'{det}_{el}.npy'))
        mape[(det, 'prova2', el)] = np.load(os.path.join(NPY2, f'{det}_{el}.npy'))

print("Mape učitane.")

N = len(ELEMENTI)  # 5


# ══════════════════════════════════════════════════════════════════
#  MAPE ELEMENATA – jedna slika po detektoru (2 reda: prova1 / prova2)
# ══════════════════════════════════════════════════════════════════

def crtaj_mape_detektora(det):
    fig, axes = plt.subplots(2, N, figsize=(5 * N, 9))

    for c, el in enumerate(ELEMENTI):
        for r, prova in enumerate(['prova1', 'prova2']):
            mapa = mape[(det, prova, el)]
            vmax = np.percentile(mapa, 99)

            ax = axes[r, c]
            im = ax.imshow(
                mapa,
                origin='upper', aspect='auto',
                cmap=PALETE[el],
                vmin=0, vmax=vmax,
                interpolation='nearest'
            )
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label('counts', fontsize=7)
            cbar.ax.tick_params(labelsize=7)

            ax.set_title(f"{NASLOVI[el]}\n{prova}", fontsize=9, fontweight='bold')
            ax.set_xlabel('Kolona', fontsize=7)
            ax.set_ylabel('Red', fontsize=7)
            ax.tick_params(labelsize=6)

    fig.suptitle(
        f"Mape elemenata – Detektor {det}\n"
        f"Red 1: prova1  |  Red 2: prova2",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    put = os.path.join(IZLAZ, f'mape_{det}.png')
    plt.savefig(put, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sačuvano: {os.path.abspath(put)}")

for det in DETEKTORI:
    crtaj_mape_detektora(det)


# ══════════════════════════════════════════════════════════════════
#  RAZLIKE prova1 − prova2  (2 reda: detektori × 5 kolona: elementi)
# ══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, N, figsize=(5 * N, 9))

for r, det in enumerate(DETEKTORI):
    for c, el in enumerate(ELEMENTI):
        m1 = mape[(det, 'prova1', el)]
        m2 = mape[(det, 'prova2', el)]
        razlika = m1.astype(np.float64) - m2.astype(np.float64)

        abs_max = np.percentile(np.abs(razlika), 99)
        if abs_max == 0:
            abs_max = 1.0

        ax = axes[r, c]
        im = ax.imshow(
            razlika,
            origin='upper', aspect='auto',
            cmap='RdBu_r',
            vmin=-abs_max, vmax=abs_max,
            interpolation='nearest'
        )
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label('counts\n(prova1 − prova2)', fontsize=7, labelpad=4)
        cbar.ax.tick_params(labelsize=7)

        ax.set_title(f"Det. {det}  –  {NASLOVI[el]}", fontsize=9, fontweight='bold')
        ax.set_xlabel('Kolona', fontsize=7)
        ax.set_ylabel('Red', fontsize=7)
        ax.tick_params(labelsize=6)

fig.suptitle(
    "Razlike mapa elemenata: prova1 − prova2\n"
    "CRVENO = više u prova1  |  PLAVO = više u prova2  |  BELO = nema razlike",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
put = os.path.join(IZLAZ, 'razlike_svi_elementi.png')
plt.savefig(put, dpi=150, bbox_inches='tight')
plt.close()
print(f"Sačuvano: {os.path.abspath(put)}")
