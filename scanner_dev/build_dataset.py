"""Build a local Scout staging directory for scanner development runs."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

# specify which config to use for this build script
from tool_access.config import DEFAULT_BUILD_DIR, REPO_ROOT, SPLITS, VALIDATION_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage a manifest-defined subset of eval logs into one local directory "
            "and generate merged validation metadata."
        )
    )
    parser.add_argument(
        "--split",
        choices=sorted(SPLITS),
        default="dev",
        help="Which manifest split to stage (controls default file names).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="CSV manifest describing the corpus (defaults to the split's manifest).",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help="Output directory for generated eval logs and CSVs.",
    )
    parser.add_argument(
        "--validated-only",
        action="store_true",
        help="Stage only manifest rows with include_in_validation=yes.",
    )
    parser.add_argument(
        "--eval",
        dest="eval_names",
        action="append",
        default=[],
        help="Limit staging to one or more Eval values from the manifest.",
    )
    parser.add_argument(
        "--method",
        dest="methods",
        action="append",
        default=[],
        help="Limit staging to one or more method values from the manifest.",
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        default=[],
        help="Limit staging to one or more model values from the manifest.",
    )
    return parser.parse_args()


def parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default

    normalized = value.strip().lower()
    if not normalized:
        return default
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False

    raise ValueError(f"Unsupported boolean value: {value!r}")


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def resolve_manifest_path(value: str, *, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()

    repo_relative_path = (REPO_ROOT / path).resolve()
    manifest_relative_path = (base_dir / path).resolve()
    return repo_relative_path if repo_relative_path.exists() else manifest_relative_path


def resolve_unique_eval_path(eval_file: str) -> Path:
    matches = list((REPO_ROOT / "evals").rglob(eval_file))
    if not matches:
        raise FileNotFoundError(
            f"Could not find eval log {eval_file!r} under evals/. "
            "If it is only on Hugging Face, pull the relevant eval subtree first."
        )
    if len(matches) > 1:
        joined = ", ".join(repo_relative(path) for path in matches)
        raise ValueError(f"Ambiguous eval log {eval_file!r}: {joined}")
    return matches[0].resolve()


def resolve_unique_validation_path(graded_file: str) -> Path:
    matches = list((REPO_ROOT / "evals").rglob(graded_file))
    if not matches:
        raise FileNotFoundError(
            f"Could not find validation file {graded_file!r} under evals/."
        )
    if len(matches) > 1:
        joined = ", ".join(repo_relative(path) for path in matches)
        raise ValueError(f"Ambiguous validation file {graded_file!r}: {joined}")
    return matches[0].resolve()


@dataclass(frozen=True)
class ManifestRow:
    line_number: int
    eval_name: str
    method: str
    model: str
    samples: str
    expected_v_rate: str
    graded_file: str
    eval_dir: str
    eval_log_path: Path
    validation_path: Path | None
    include_in_dev: bool
    include_in_validation: bool

    @property
    def staged_filename(self) -> str:
        return self.eval_log_path.name

    @classmethod
    def from_csv(
        cls,
        row: dict[str, str],
        line_number: int,
        *,
        manifest_dir: Path,
    ) -> "ManifestRow":
        eval_file = row.get("eval_file", "").strip()
        if not eval_file:
            raise ValueError(f"Row {line_number}: missing eval_file")

        eval_log_value = row.get("eval_log_path", "").strip()
        eval_log_path = (
            resolve_manifest_path(eval_log_value, base_dir=manifest_dir)
            if eval_log_value
            else resolve_unique_eval_path(eval_file)
        )
        if not eval_log_path.exists():
            raise FileNotFoundError(
                f"Row {line_number}: missing transcript file {repo_relative(eval_log_path)}"
            )

        graded_file = row.get("graded_file", "").strip()
        validation_value = row.get("validation_path", "").strip()
        validation_path: Path | None
        if validation_value:
            validation_path = resolve_manifest_path(validation_value, base_dir=manifest_dir)
        elif graded_file:
            validation_path = resolve_unique_validation_path(graded_file)
        else:
            validation_path = None

        if validation_path is not None and not validation_path.exists():
            raise FileNotFoundError(
                f"Row {line_number}: missing validation file {repo_relative(validation_path)}"
            )

        include_in_dev = parse_bool(row.get("include_in_dev"), default=True)
        include_in_validation = parse_bool(
            row.get("include_in_validation"),
            default=validation_path is not None,
        )

        # if include_in_validation and validation_path is None:
        #     print(
        #         f"Warning: Row {line_number}: include_in_validation=yes but no validation_path; "
        #         f"treating as include_in_validation=no"
        #     )
        #     include_in_validation = False
        if include_in_validation and not include_in_dev:
            raise ValueError(
                f"Row {line_number}: validation rows must also be included in the dev corpus"
            )

        eval_dir = row.get("eval_dir", "").strip()
        if not eval_dir:
            eval_dir = "/".join(repo_relative(eval_log_path).split("/")[:2])

        return cls(
            line_number=line_number,
            eval_name=row.get("Eval", "").strip(),
            method=row.get("method", "").strip(),
            model=row.get("models", "").strip(),
            samples=row.get("samples", "").strip(),
            expected_v_rate=row.get("expected_v_rate", "").strip(),
            graded_file=graded_file,
            eval_dir=eval_dir,
            eval_log_path=eval_log_path,
            validation_path=validation_path,
            include_in_dev=include_in_dev,
            include_in_validation=include_in_validation,
        )


def load_manifest(manifest_path: Path) -> list[ManifestRow]:
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [
            ManifestRow.from_csv(row, index + 2, manifest_dir=manifest_path.parent)
            for index, row in enumerate(reader)
        ]

    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    duplicate_paths: dict[str, list[int]] = {}
    duplicate_stage_names: dict[str, list[int]] = {}
    for row in rows:
        duplicate_paths.setdefault(row.eval_log_path.resolve().as_posix(), []).append(row.line_number)
        duplicate_stage_names.setdefault(row.staged_filename, []).append(row.line_number)

    repeated_paths = {
        path: line_numbers
        for path, line_numbers in duplicate_paths.items()
        if len(line_numbers) > 1
    }
    if repeated_paths:
        formatted = "; ".join(
            f"{repo_relative(Path(path))} on lines {line_numbers}"
            for path, line_numbers in repeated_paths.items()
        )
        raise ValueError(f"Manifest contains duplicate transcript paths: {formatted}")

    repeated_stage_names = {
        name: line_numbers
        for name, line_numbers in duplicate_stage_names.items()
        if len(line_numbers) > 1
    }
    if repeated_stage_names:
        formatted = "; ".join(
            f"{name} on lines {line_numbers}"
            for name, line_numbers in repeated_stage_names.items()
        )
        raise ValueError(f"Manifest contains duplicate staged filenames: {formatted}")

    return rows


def select_rows(rows: list[ManifestRow], args: argparse.Namespace) -> list[ManifestRow]:
    selected = list(rows)

    if args.validated_only:
        selected = [row for row in selected if row.include_in_validation]

    if args.eval_names:
        allowed = {name.strip() for name in args.eval_names if name.strip()}
        selected = [row for row in selected if row.eval_name in allowed]

    if args.methods:
        allowed = {name.strip() for name in args.methods if name.strip()}
        selected = [row for row in selected if row.method in allowed]

    if args.models:
        allowed = {name.strip() for name in args.models if name.strip()}
        selected = [row for row in selected if row.model in allowed]

    return selected


def stage_eval_logs(rows: list[ManifestRow], eval_logs_dir: Path) -> int:
    if eval_logs_dir.exists():
        shutil.rmtree(eval_logs_dir)
    eval_logs_dir.mkdir(parents=True, exist_ok=True)
    stale_transcripts_dir = eval_logs_dir.parent / "transcripts"
    if stale_transcripts_dir.exists():
        shutil.rmtree(stale_transcripts_dir)

    staged = 0
    for row in rows:
        if not row.include_in_dev:
            continue

        destination = eval_logs_dir / row.staged_filename
        try:
            os.link(row.eval_log_path, destination)
        except OSError:
            shutil.copy2(row.eval_log_path, destination)
        staged += 1

    return staged


def write_validation_csv(rows: list[ManifestRow], output_path: Path) -> int:
    merged_rows: dict[str, dict[str, str]] = {}
    validation_files = sorted(
        {
            row.validation_path.resolve()
            for row in rows
            if row.include_in_validation and row.validation_path is not None
        }
    )

    if not validation_files:
        print(
            f"Warning: no validation files in selected manifest rows; "
            f"writing header-only {repo_relative(output_path)}"
        )

    for validation_path in validation_files:
        with validation_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            missing_columns = [name for name in VALIDATION_COLUMNS if name not in reader.fieldnames]
            if missing_columns:
                joined = ", ".join(missing_columns)
                raise ValueError(
                    f"{repo_relative(validation_path)} is missing required columns: {joined}"
                )

            for row in reader:
                transcript_id = row["id"].strip()
                normalized = {
                    "id": transcript_id,
                    "target": row["target"].strip(),
                    "predicate": row["predicate"].strip() or "eq",
                }
                previous = merged_rows.get(transcript_id)
                if previous is not None and previous != normalized:
                    raise ValueError(
                        f"Conflicting validation labels for transcript {transcript_id}"
                    )
                merged_rows[transcript_id] = normalized

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(VALIDATION_COLUMNS))
        writer.writeheader()
        for transcript_id in sorted(merged_rows):
            writer.writerow(merged_rows[transcript_id])

    return len(merged_rows)


def write_provenance_csv(rows: list[ManifestRow], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        fieldnames = [
            "Eval",
            "method",
            "model",
            "samples",
            "expected_v_rate",
            "graded_file",
            "eval_dir",
            "eval_log_path",
            "staged_eval_log_path",
            "validation_path",
            "include_in_dev",
            "include_in_validation",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        rows_written = 0
        for row in rows:
            writer.writerow(
                {
                    "Eval": row.eval_name,
                    "method": row.method,
                    "model": row.model,
                    "samples": row.samples,
                    "expected_v_rate": row.expected_v_rate,
                    "graded_file": row.graded_file,
                    "eval_dir": row.eval_dir,
                    "eval_log_path": repo_relative(row.eval_log_path),
                    "staged_eval_log_path": f"build/eval-logs/{row.staged_filename}",
                    "validation_path": (
                        repo_relative(row.validation_path)
                        if row.validation_path is not None
                        else ""
                    ),
                    "include_in_dev": "yes" if row.include_in_dev else "no",
                    "include_in_validation": "yes" if row.include_in_validation else "no",
                }
            )
            rows_written += 1

    return rows_written


def main() -> None:
    args = parse_args()
    split = SPLITS[args.split]
    manifest_path = (args.manifest or split["manifest"]).resolve()
    build_dir = args.build_dir.resolve()

    rows = load_manifest(manifest_path)
    rows = select_rows(rows, args)
    dev_rows = [row for row in rows if row.include_in_dev]
    if not dev_rows:
        raise ValueError("No manifest rows matched the requested staging filters.")

    eval_logs_dir = build_dir / "eval-logs"
    validation_path = split["validation"]
    provenance_path = build_dir / split["provenance"]

    staged_count = stage_eval_logs(rows, eval_logs_dir)
    validation_count = write_validation_csv(rows, validation_path)
    provenance_count = write_provenance_csv(rows, provenance_path)

    print(f"Manifest: {manifest_path}")
    print(f"Staged eval logs: {staged_count}")
    print(f"Eval log staging dir: {eval_logs_dir}")
    print(f"Merged validation rows: {validation_count}")
    print(f"Validation CSV: {validation_path}")
    print(f"Provenance rows: {provenance_count}")
    print(f"Provenance CSV: {provenance_path}")


if __name__ == "__main__":
    main()
