"""
Dataset loading and processing utilities.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from mutant.core.scenario import Scenario


def _validate_columns(
    item: dict[str, Any], text_columns: list[str], title_column: str | None
) -> None:
    for col in text_columns:
        if col not in item:
            raise ValueError(
                f"Required text column '{col}' not found in item: {item.keys()}"
            )
    if title_column and title_column not in item:
        raise ValueError(
            f"Required title column '{title_column}' not found in item: {item.keys()}"
        )


def _build_scenario(
    item: dict[str, Any],
    text_columns: list[str],
    title_column: str | None,
    default_title: str,
) -> Scenario:
    if len(text_columns) == 1:
        description = str(item[text_columns[0]])
    else:
        description = "\n\n".join(
            f"{col.replace('_', ' ').title()}:\n{item[col]}" for col in text_columns
        )

    title = (
        str(item[title_column])
        if title_column and title_column in item
        else default_title
    )

    # Store all fields in context to preserve rich data
    context = {k: v for k, v in item.items() if k != title_column}

    return Scenario(title=title, description=description, context=context)


def load_json(
    path: str | Path,
    text_column: str | None = None,
    text_columns: list[str] | None = None,
    title_column: str | None = None,
) -> list[Scenario]:
    """Load scenarios from a JSON array of objects.

    Parameters
    ----------
    path:
        Path to the JSON file.
    text_column:
        Deprecated. Use text_columns.
    text_columns:
        List of keys containing the scenario description.
    title_column:
        Optional key containing the scenario title.
    """
    path = Path(path)
    if text_columns is None:
        if isinstance(text_column, list):
            text_columns = text_column
        else:
            text_columns = [text_column] if text_column else ["text"]

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON file must contain an array of objects.")

    scenarios = []
    for i, item in enumerate(data):
        _validate_columns(item, text_columns, title_column)
        scenarios.append(
            _build_scenario(item, text_columns, title_column, f"Scenario {i + 1}")
        )

    return scenarios


def load_jsonl(
    path: str | Path,
    text_column: str | None = None,
    text_columns: list[str] | None = None,
    title_column: str | None = None,
) -> list[Scenario]:
    """Load scenarios from a JSON Lines file."""
    path = Path(path)
    if text_columns is None:
        if isinstance(text_column, list):
            text_columns = text_column
        else:
            text_columns = [text_column] if text_column else ["text"]

    scenarios = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            _validate_columns(item, text_columns, title_column)
            scenarios.append(
                _build_scenario(item, text_columns, title_column, f"Scenario {i + 1}")
            )
    return scenarios


def load_csv(
    path: str | Path,
    text_column: str | None = None,
    text_columns: list[str] | None = None,
    title_column: str | None = None,
) -> list[Scenario]:
    """Load scenarios from a CSV file."""
    path = Path(path)
    if text_columns is None:
        if isinstance(text_column, list):
            text_columns = text_column
        else:
            text_columns = [text_column] if text_column else ["text"]

    scenarios = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, item in enumerate(reader):
            _validate_columns(item, text_columns, title_column)
            scenarios.append(
                _build_scenario(item, text_columns, title_column, f"Scenario {i + 1}")
            )
    return scenarios


def load_dataframe(
    df: Any,
    text_column: str | None = None,
    text_columns: list[str] | None = None,
    title_column: str | None = None,
) -> list[Scenario]:
    """Load scenarios from a Pandas or Polars DataFrame."""
    if text_columns is None:
        if isinstance(text_column, list):
            text_columns = text_column
        else:
            text_columns = [text_column] if text_column else ["text"]

    # Use standard dict conversion to support both pandas and polars
    if hasattr(df, "to_dict"):
        # Pandas df.to_dict("records")
        records = df.to_dict(orient="records")
    elif hasattr(df, "to_dicts"):
        # Polars df.to_dicts()
        records = df.to_dicts()
    else:
        raise TypeError("df must be a pandas or polars DataFrame.")

    scenarios = []
    for i, item in enumerate(records):
        _validate_columns(item, text_columns, title_column)
        scenarios.append(
            _build_scenario(item, text_columns, title_column, f"Scenario {i + 1}")
        )
    return scenarios


def load_huggingface(
    dataset: Any,
    text_column: str | None = None,
    text_columns: list[str] | None = None,
    title_column: str | None = None,
) -> list[Scenario]:
    """Load scenarios from a HuggingFace Dataset."""
    if text_columns is None:
        if isinstance(text_column, list):
            text_columns = text_column
        else:
            text_columns = [text_column] if text_column else ["text"]

    scenarios = []
    for i, item in enumerate(dataset):
        _validate_columns(item, text_columns, title_column)
        scenarios.append(
            _build_scenario(item, text_columns, title_column, f"Scenario {i + 1}")
        )
    return scenarios
