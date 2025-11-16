"""Command-line interface for blackbody_mcmc"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import load_data
from .diagnostics import autocorrelation, gelman_rubin
from .mcmc_library import run_emcee
from .mcmc_manual import run_mcmc
from .model import log_posterior, model_intensity
from .plotting import plot_data_and_model, plot_histograms, plot_traces
