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


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Fit blackbody data using Bayesian MCMC"
    )
    parser.add_argument(
        "data_file",
        help="CSV file with wavelength,intensity columns",
    )
    parser.add_argument(
        "--n-steps-manual",
        type=int,
        default=50_000,
        help="Number of steps for manual Metropolis sampler",
    )
    parser.add_argument(
        "--n-steps-emcee",
        type=int,
        default=10_000,
        help="Number of steps for emcee sampler",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to store plots and outputs",
    )
    parser.add_argument(
        "--proposal-scale",
        type=float,
        nargs=2,
        metavar=("SIGMA_T", "SIGMA_A"),
        default=(50.0, 0.1),
        help="Proposal std devs for (T, A) in manual sampler",
    )
    return parser.parse_args()

