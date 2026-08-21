"""core.py - bootstrap access to the validated neurips-restore code.

Everything in neurips_submission builds ON TOP of neurips-restore/src
(measured forward model, data generator, U-Net, evaluation harness).
That code is FROZEN for this submission: fix bugs there only with the
whole team's sign-off, never fork it silently.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUBMISSION_ROOT = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(SUBMISSION_ROOT)
SRC = os.path.join(REPO_ROOT, "neurips-restore", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if SUBMISSION_ROOT not in sys.path:
    sys.path.insert(0, SUBMISSION_ROOT)

import forward_model as fm                      # noqa: E402
import datagen as dg                            # noqa: E402
import eval as ev                               # noqa: E402
from model import RestorationUNet, count_params  # noqa: E402,F401

ELEMENTS = fm.ELEMENTS
FRONTAL_SHAPE = fm.FRONTAL_SHAPE   # (60, 120)
TILTED_SHAPE = fm.TILTED_SHAPE     # (45, 80)

RESULTS_DIR = os.path.join(SUBMISSION_ROOT, "results")
FIGURES_DIR = os.path.join(SUBMISSION_ROOT, "figures")
MVP_CKPT = os.path.join(REPO_ROOT, "neurips-restore", "experiments",
                        "checkpoint.pt")
