"""Push and pull eval folders via the shared HF helpers."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EVALS_DIR = REPO_ROOT / "evals"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class PullDatasetFn(Protocol):
    def __call__(
        self,
        output_dir: Path,
        token: str | None = None,
        allow_patterns: list[str] | None = None,
    ) -> None: ...


class PushDatasetFn(Protocol):
    def __call__(
        self,
        eval_logs_dir: Path,
        scan_results_dir: Path,
        validation_dir: Path | None = None,
        scanner_name: str | None = None,
        dataset_subdir: str = "xstest",
        token: str | None = None,
        commit_message: str | None = None,
    ) -> None: ...


def load_shared_helpers() -> tuple[PullDatasetFn, PushDatasetFn]:
    """Load the shared HF dataset helper functions from the repo tools package."""
    pull_module = importlib.import_module("tools.pull_hf_dataset")
    push_module = importlib.import_module("tools.push_hf_dataset")
    return pull_module.pull_dataset, push_module.push_dataset


def eval_dir(eval_name: str) -> Path:
    """Return the local HF dataset directory for an eval."""
    return EVALS_DIR / eval_name


def push_eval(
    eval_name: str,
    scanner_name: str | None = None,
    token: str | None = None,
    commit_message: str | None = None,
) -> None:
    """Push one local eval folder from evals/ into its dataset subdirectory."""
    _, push_dataset = load_shared_helpers()
    target_dir = eval_dir(eval_name)

    push_dataset(
        eval_logs_dir=target_dir / "eval-logs",
        scan_results_dir=target_dir / "scan-results",
        validation_dir=target_dir / "validation",
        scanner_name=scanner_name,
        dataset_subdir=eval_name,
        token=token,
        commit_message=commit_message
        or f"Update evals/{eval_name}",
    )


def pull_evals(eval_name: str | None = None, token: str | None = None) -> None:
    """Pull the full dataset, or one eval subtree, directly into evals/."""
    pull_dataset, _ = load_shared_helpers()
    allow_patterns = [f"{eval_name}/**"] if eval_name else None
    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    pull_dataset(output_dir=EVALS_DIR, token=token, allow_patterns=allow_patterns)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Push or pull scanner validation data with evals as "
            "the local dataset root."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    push_parser = subparsers.add_parser(
        "push",
        help="Push evals/<eval_name> to Hugging Face",
    )
    push_parser.add_argument(
        "eval_name",
        help="Name of the eval folder under evals/",
    )
    push_parser.add_argument(
        "--scanner-name",
        help=(
            "Optional scanner name for the remote scan-results/<scanner-name>/ path. "
            "If omitted, the full local scan-results tree is uploaded as-is."
        ),
    )
    push_parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token (default: $HF_TOKEN)",
    )
    push_parser.add_argument(
        "--commit-message",
        help="Optional commit message for the Hugging Face push",
    )

    pull_parser = subparsers.add_parser(
        "pull",
        help="Pull the full dataset, or one eval subtree, into evals",
    )
    pull_parser.add_argument(
        "eval_name",
        nargs="?",
        help="Optional eval name to pull only one subtree",
    )
    pull_parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token (default: $HF_TOKEN)",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if args.command == "push":
        push_eval(
            eval_name=args.eval_name,
            scanner_name=args.scanner_name,
            token=args.token,
            commit_message=args.commit_message,
        )
        return

    pull_evals(eval_name=args.eval_name, token=args.token)


if __name__ == "__main__":
    main()
