"""
Script for running Tau2 via inspect evals.

"""

from inspect_ai import eval
from inspect_evals.tau2 import tau2_airline, tau2_retail, tau2_telecom
from pathlib import Path


LOG_DIR = str(Path(__file__).parent / "eval-logs")


task = tau2_airline(
    message_limit=100,
)

eval(
    task, 
    model="openai/gpt-5.4-2026-03-05",
    log_dir=LOG_DIR,
    model_roles={'user': 'openai/gpt-4.1-2025-04-14'}
) 
#Model options
# gpt-5.4-2026-03-05
# gpt-5-mini-2025-08-07
# gpt-5-nano-2025-08-07

# meant to run inside the tau2 folder as the working directory
#run with [uv run python tau2_run.py]
#then use [inspect view --log-dir "./eval-logs"] to see results

