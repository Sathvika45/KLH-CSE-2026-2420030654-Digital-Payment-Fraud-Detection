
"""
Dataset loader for digital payment fraud datasets.
Supports Credit Card Fraud, PaySim, and Fraud E-commerce.
Includes IP range-to-country mapping for the E-commerce dataset.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.validator import (
    DatasetConfig,
    FILENAME_ALIASES,
    SUPPORTED_DATASETS,
)


@dataclass
class LoadResult:
    """Encapsulates loaded dataset and sampling metadata."""

    df: pd.DataFrame
    dataset_config: DatasetConfig
    source_paths: List[Path]
    total_records: int
    loaded_records: int
    is_sampled: bool = False
    sample_size: Optional[int] = None
    sampling_method: Optional[str] = None
    random_state: Optional[int] = None
    unmatched_ip_count: Optional[int] = None


def get_default_data_dir() -> Path:
    """Returns the default project data/raw directory."""
    return Path(__file__).resolve().parent.parent.parent / "data" / "raw"


def check_raw_data_status(
    data_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Checks availability of primary and secondary dataset files."""

    if data_dir is None:
        data_dir = get_default_data_dir()

    data_dir = Path(data_dir)
    status: Dict[str, Dict[str, Any]] = {}

    if not data_dir.exists():
        for dataset_id, config in SUPPORTED_DATASETS.items():
            status[dataset_id] = {
                "display_name": config.display_name,
                "found": False,
                "primary_path": None,
                "secondary_path": None,
                "secondary_found": config.secondary_filename is None,
                "expected_primary": config.primary_filename,
                "expected_secondary": config.secondary_filename,
            }
        return status

    existing_files = {
        path.name.lower(): path
        for path in data_dir.iterdir()
        if path.is_file()
    }

    for dataset_id, config in SUPPORTED_DATASETS.items():
        primary_match = existing_files.get(config.primary_filename.lower())

        if primary_match is None:
            for alias, mapped_id in FILENAME_ALIASES.items():
                if mapped_id == dataset_id and alias.lower() in existing_files:
                    primary_match = existing_files[alias.lower()]
                    break

        secondary_match = None
        if config.secondary_filename:
            secondary_match = existing_files.get(
                config.secondary_filename.lower()
            )

        status[dataset_id] = {
            "display_name": config.display_name,
            "found": primary_match is not None,
            "primary_path": primary_match,
            "secondary_path": secondary_match,
            "secondary_found": (
                secondary_match is not None
                if config.secondary_filename
                else True
            ),
            "expected_primary": config.primary_filename,
            "expected_secondary": config.secondary_filename,
        }

    return status


def map_ip_addresses_to_country(
    df: pd.DataFrame,
    ip_mapping_path: Path,
) -> tuple[pd.DataFrame, int]:
    """
    Maps numeric IP addresses to countries using inclusive IP ranges.

    Unmatched or invalid IP addresses are labelled 'Unknown'.
    Returns the updated DataFrame and unmatched IP count.
    """

    required_mapping_columns = {
        "lower_bound_ip_address",
        "upper_bound_ip_address",
        "country",
    }

    mapping_df = pd.read_csv(ip_mapping_path)

    missing_columns = required_mapping_columns - set(mapping_df.columns)
    if missing_columns:
        raise ValueError(
            f"IP mapping file is missing columns: {sorted(missing_columns)}"
        )

    if "ip_address" not in df.columns:
        raise ValueError("E-commerce data is missing the 'ip_address' column.")

    mapping_df["lower_bound_ip_address"] = pd.to_numeric(
        mapping_df["lower_bound_ip_address"], errors="coerce"
    )
    mapping_df["upper_bound_ip_address"] = pd.to_numeric(
        mapping_df["upper_bound_ip_address"], errors="coerce"
    )

    mapping_df = mapping_df.dropna(
        subset=[
            "lower_bound_ip_address",
            "upper_bound_ip_address",
            "country",
        ]
    ).sort_values("lower_bound_ip_address").reset_index(drop=True)

    lower = mapping_df["lower_bound_ip_address"].to_numpy(dtype="float64")
    upper = mapping_df["upper_bound_ip_address"].to_numpy(dtype="float64")
    countries = mapping_df["country"].to_numpy()

    if len(lower) == 0:
        raise ValueError("IP mapping file contains no valid IP ranges.")

    if (lower > upper).any():
        raise ValueError("IP mapping file contains invalid ranges.")

    ip_values = pd.to_numeric(
        df["ip_address"], errors="coerce"
    ).to_numpy(dtype="float64")

    valid_ip = np.isfinite(ip_values)

    country_values = np.full(len(df), "Unknown", dtype=object)

    valid_positions = np.flatnonzero(valid_ip)
    if len(valid_positions):
        candidate_indices = (
            np.searchsorted(
                lower,
                ip_values[valid_positions],
                side="right",
            ) - 1
        )

        has_candidate = candidate_indices >= 0

        matched_positions = valid_positions[has_candidate]
        matched_indices = candidate_indices[has_candidate]

        inside_range = (
            ip_values[matched_positions] <= upper[matched_indices]
        )

        country_values[matched_positions[inside_range]] = (
            countries[matched_indices[inside_range]]
        )

    df = df.copy()
    df["ip_country"] = country_values

    unmatched_count = int((country_values == "Unknown").sum())

    return df, unmatched_count


def load_dataset(
    dataset_id: str,
    data_dir: Optional[Path] = None,
    custom_filepath: Optional[Path] = None,
    sample_size: Optional[int] = None,
    random_state: int = 42,
    join_identity: bool = True,
) -> LoadResult:
    """
    Loads a supported fraud dataset.

    E-commerce IP mapping is performed before sampling.
    Sampling is stratified by the target column where possible.
    """

    if dataset_id not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset_id '{dataset_id}'. "
            f"Supported: {list(SUPPORTED_DATASETS.keys())}"
        )

    config = SUPPORTED_DATASETS[dataset_id]

    if data_dir is None:
        data_dir = get_default_data_dir()

    data_dir = Path(data_dir)
    source_paths: List[Path] = []

    if custom_filepath:
        primary_path = Path(custom_filepath)
        if not primary_path.exists():
            raise FileNotFoundError(
                f"Custom dataset file not found: {primary_path}"
            )
    else:
        status = check_raw_data_status(data_dir)
        dataset_status = status.get(dataset_id, {})
        primary_path = dataset_status.get("primary_path")

        if primary_path is None or not primary_path.exists():
            raise FileNotFoundError(
                f"Dataset '{config.display_name}' not found in '{data_dir}'. "
                f"Expected: '{config.primary_filename}'."
            )

    source_paths.append(primary_path)

    df = pd.read_csv(primary_path)

    unmatched_ip_count = None

    if dataset_id == "ecommerce":
        mapping_path = (
            data_dir / config.secondary_filename
            if config.secondary_filename
            else None
        )

        if mapping_path is None or not mapping_path.exists():
            raise FileNotFoundError(
                "E-commerce IP mapping file not found. "
                f"Expected: '{config.secondary_filename}'."
            )

        source_paths.append(mapping_path)

        df, unmatched_ip_count = map_ip_addresses_to_country(
            df,
            mapping_path,
        )

    total_records = len(df)
    loaded_records = total_records
    is_sampled = False
    sampling_method = None

    if sample_size is not None and sample_size <= 0:
        raise ValueError("sample_size must be greater than zero.")

    if sample_size is not None and sample_size < total_records:
        target_col = config.target_column

        if target_col in df.columns and df[target_col].nunique() > 1:
            sampling_method = "stratified"
            sampled_groups = []

            for _, group in df.groupby(target_col):
                group_count = max(
                    1,
                    round(sample_size * len(group) / total_records),
                )
                group_count = min(group_count, len(group))

                sampled_groups.append(
                    group.sample(
                        n=group_count,
                        random_state=random_state,
                    )
                )

            df = pd.concat(sampled_groups)

            if len(df) > sample_size:
                df = df.sample(
                    n=sample_size,
                    random_state=random_state,
                )

            df = df.reset_index(drop=True)
        else:
            sampling_method = "random"
            df = df.sample(
                n=sample_size,
                random_state=random_state,
            ).reset_index(drop=True)

        loaded_records = len(df)
        is_sampled = True

    return LoadResult(
        df=df,
        dataset_config=config,
        source_paths=source_paths,
        total_records=total_records,
        loaded_records=loaded_records,
        is_sampled=is_sampled,
        sample_size=sample_size if is_sampled else None,
        sampling_method=sampling_method,
        random_state=random_state if is_sampled else None,
        unmatched_ip_count=unmatched_ip_count,
    )
