"""Client-side long/wide reshape helpers (2026-07-27).

``pivot_longer()`` / ``pivot_wider()`` reshape a DataFrame returned by
:meth:`Client.get` or :meth:`Client.get_local` using the ``layout`` /
``time_columns`` / ``id_columns`` / ``value_columns`` / ``measure_name_column``
metadata attached to it (see ``eolas_data.meta.attach_meta``).

These are deliberately **explicit, separate functions** — they are never
called automatically by :meth:`Client.get` / :meth:`Client.get_local`, and
they **refuse** rather than guess a shape:

* No ``layout`` metadata attached (and none derivable via ``name=``/``client=``)
  → :class:`~eolas_data.exceptions.ReshapeError`.
* ``layout in ("feature", "entity")`` → refuse (a feature/entity row is the
  natural grain; melting or pivoting is meaningless).
* A geometry column (``geometry_wkt`` string, or an active GeoDataFrame
  geometry column) is present → refuse (melting geometry repeats the
  WKT/WKB payload once per measure row).

This mirrors the reasoning that led to ``Dataset.plot_dataset()`` (and R's
``eolas_plot()``) being removed in v1.3.0 — see ``dataset.py``: eolas does not
ship a helper that has to *guess* per-dataset shape from column names like
``date``/``period``/``value``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

import pandas as pd

from .exceptions import ReshapeError

if TYPE_CHECKING:  # pragma: no cover
    from .client import Client

__all__ = ["pivot_longer", "pivot_wider"]

_GEOMETRY_COLUMNS = {"geometry_wkt", "geometry"}


def _is_geodataframe(df: pd.DataFrame) -> bool:
    try:
        import geopandas as gpd
    except ImportError:
        return False
    return isinstance(df, gpd.GeoDataFrame)


def _frame_attr(df: pd.DataFrame, key: str) -> Any:
    """Read an eolas attribute — checks `.attrs` (always set, incl. GeoDataFrame)
    before the object attribute (set for plain Dataset/DataFrame only — see
    meta.attach_meta, which skips object.__setattr__ for GeoDataFrames)."""
    attrs = getattr(df, "attrs", None)
    if isinstance(attrs, dict) and key in attrs and attrs[key] not in (None, {}):
        return attrs[key]
    return getattr(df, key, None)


def _as_list(v: Any) -> list:
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return list(v)
    return [v]


def _dataset_label(df: pd.DataFrame, name: Optional[str]) -> str:
    return name or _frame_attr(df, "eolas_name") or "this dataset"


def _attach_reshaped_meta(
    out: pd.DataFrame,
    src: pd.DataFrame,
    *,
    layout: str,
    time_columns: list,
    id_columns: list,
    value_columns: list,
    measure_name_column: str,
) -> pd.DataFrame:
    """Rewrite the eolas layout metadata on a reshaped frame so it describes the
    NEW shape, not the source shape. Without this a `pivot_longer(wide)` result
    still says `layout="wide"` and `pivot_wider()` on it wrongly errors
    "already wide" — the round-trip is only usable if the result re-declares
    itself long/wide. pandas melt/pivot do not carry `.attrs` reliably, so we
    set them explicitly."""
    out.attrs["eolas_meta"] = {
        "layout": layout,
        "time_columns": [c for c in time_columns if c in out.columns],
        "id_columns": [c for c in id_columns if c in out.columns],
        "value_columns": value_columns,
        "measure_name_column": measure_name_column,
    }
    name = _frame_attr(src, "eolas_name")
    if name:
        out.attrs["eolas_name"] = name
    return out


def _series_id_map(column_meta: Any) -> dict:
    """Column-name -> series_id from the eolas_columns glossary, if any."""
    if column_meta is None:
        return {}
    if isinstance(column_meta, pd.DataFrame):
        if column_meta.empty or "series_id" not in column_meta.columns:
            return {}
        records = column_meta.to_dict("records")
    elif isinstance(column_meta, list):
        records = column_meta
    else:
        return {}
    out: dict[str, str] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        sid = rec.get("series_id")
        col = rec.get("name")
        if sid and col:
            out[col] = sid
    return out


def _get_eolas_meta(
    df: pd.DataFrame,
    *,
    name: Optional[str] = None,
    client: Optional["Client"] = None,
) -> dict:
    """Resolve the DatasetMeta dict attached to `df`, or fetch it by name.

    Checked in order: `df.attrs["eolas_meta"]` / `df.eolas_meta` (attached by
    Client.get()/get_local() with meta=True — the default, on BOTH the live
    and bulk-cache paths, since both funnel through
    Client._attach_dataset_meta -> meta.split_meta/attach_meta). Falls back to
    a fresh `client.info(name)` call when nothing is attached (e.g.
    meta=False was used, or `df` was loaded from disk with no attrs).
    """
    meta = _frame_attr(df, "eolas_meta")
    if not meta:
        ds_name = name or _frame_attr(df, "eolas_name")
        if ds_name and client is not None:
            try:
                meta = client.info(ds_name)
            except Exception:
                meta = None
    if not meta:
        raise ReshapeError(
            "Cannot reshape: no eolas layout metadata attached to this DataFrame. "
            "Pass a frame from Client.get()/get_local() with meta=True (the "
            "default), or call pivot_longer(df, name=..., client=...) / "
            "pivot_wider(df, name=..., client=...) to fetch it directly."
        )
    return meta


def _reshape_guard(df: pd.DataFrame, meta: dict, *, name: Optional[str] = None) -> str:
    """Shared refusal gate for pivot_longer()/pivot_wider(). Returns the
    lower-cased layout string on success; raises ReshapeError otherwise."""
    ds_name = _dataset_label(df, name)
    layout = meta.get("layout")
    if not layout:
        raise ReshapeError(
            f"Cannot reshape {ds_name!r}: no `layout` metadata. eolas refuses to "
            "guess a date/period/value shape from column names — this is exactly "
            "the silent-wrongness class Dataset.plot_dataset() was removed for "
            "in v1.3.0. See https://docs.eolas.fyi for the layout contract."
        )
    layout = str(layout).strip().lower()
    if layout in ("feature", "entity"):
        kind = (
            "a feature table (one row per real-world entity, e.g. a parcel or "
            "address)"
            if layout == "feature"
            else "an entity table (one row per case/claim/award)"
        )
        raise ReshapeError(
            f"Cannot reshape {ds_name!r}: layout is {layout!r} — {kind}. "
            "Melting or pivoting would be meaningless."
        )
    if any(c in df.columns for c in _GEOMETRY_COLUMNS) or _is_geodataframe(df):
        raise ReshapeError(
            f"Cannot reshape {ds_name!r}: a geometry column is present. Melting "
            "geometry repeats the WKT/WKB payload once per measure row — drop it "
            "first with geometry=False on the fetch call."
        )
    return layout


def pivot_longer(
    df: pd.DataFrame,
    *,
    names_to: Optional[str] = None,
    values_to: str = "value",
    name: Optional[str] = None,
    client: Optional["Client"] = None,
) -> pd.DataFrame:
    """Reshape a wide eolas DataFrame to long format.

    Melts the dataset's ``value_columns`` (from the ``layout`` metadata
    attached by :meth:`Client.get` / :meth:`Client.get_local`) into a
    measure-name column and a value column. Requires ``layout == "wide"``
    metadata — see the module docstring for the full refusal list.

    If the wide table's columns carry a per-column ``series_id`` mapping
    (the ``eolas_columns`` glossary attribute — see
    :meth:`Dataset.column_label`), a ``series_id`` column is added to the
    result mapping each measure back to its original series id.

    Args:
        df: A DataFrame from ``Client.get()``/``Client.get_local()`` with
            ``layout = "wide"`` metadata attached.
        names_to: Name of the output column holding the former column
            names. ``None`` (default) uses the dataset's
            ``measure_name_column`` metadata, falling back to ``"measure"``.
        values_to: Name of the output value column. Default ``"value"``.
        name: Dataset identifier, used only to fetch metadata when `df` has
            none attached. Not required when `df` carries ``eolas_meta``.
        client: A :class:`Client` instance, used only with `name` to fetch
            metadata when none is attached.

    Returns:
        A long-format ``pd.DataFrame``.

    Raises:
        ReshapeError: no layout metadata, layout is ``feature``/``entity``,
            a geometry column is present, layout is not ``wide``, or none of
            the metadata ``value_columns`` are present in `df`.

    Example:
        >>> wide = client.get("rbnz_b1_exchange_rates_monthly")
        >>> long = pivot_longer(wide)
    """
    meta = _get_eolas_meta(df, name=name, client=client)
    layout = _reshape_guard(df, meta, name=name)
    ds_name = _dataset_label(df, name)

    if layout != "wide":
        hint = (
            " This dataset is already long — see pivot_wider() to go the " "other way."
            if layout == "long"
            else ""
        )
        raise ReshapeError(
            f"Cannot pivot_longer {ds_name!r}: layout is {layout!r}, not "
            f"'wide'.{hint}"
        )

    value_cols = [c for c in _as_list(meta.get("value_columns")) if c in df.columns]
    if not value_cols:
        raise ReshapeError(
            f"Cannot pivot_longer {ds_name!r}: none of the metadata "
            "`value_columns` are present in this DataFrame. The schema may "
            "have changed since metadata was cached; try force=True on the "
            "fetch call."
        )
    id_cols = [
        c
        for c in _as_list(meta.get("id_columns"))
        if c in df.columns and c not in value_cols
    ]
    if not id_cols:
        id_cols = [c for c in df.columns if c not in value_cols]

    measure_col = names_to or meta.get("measure_name_column") or "measure"

    out = df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name=measure_col,
        value_name=values_to,
    )
    out = out.dropna(subset=[values_to]).reset_index(drop=True)

    sid_map = _series_id_map(_frame_attr(df, "eolas_columns"))
    if sid_map and "series_id" not in out.columns:
        out["series_id"] = out[measure_col].map(sid_map)

    # Result is now long — re-declare it so pivot_wider() can reverse it.
    return _attach_reshaped_meta(
        out,
        df,
        layout="long",
        time_columns=_as_list(meta.get("time_columns")),
        id_columns=id_cols,
        value_columns=[values_to],
        measure_name_column=measure_col,
    )


def pivot_wider(
    df: pd.DataFrame,
    *,
    names_from: Optional[str] = None,
    values_from: Optional[str] = None,
    id_cols: Optional[Sequence[str]] = None,
    values_fill: Optional[Any] = None,
    name: Optional[str] = None,
    client: Optional["Client"] = None,
) -> pd.DataFrame:
    """Reshape a long eolas DataFrame to wide format.

    Spreads one long-format value column into multiple columns keyed by a
    "names" column. Requires ``layout == "long"`` metadata — see the module
    docstring for the full refusal list.

    Unlike ``pivot_longer``, this never has a bare "always safe" default: a
    long table with more than one dimension column can legitimately be
    pivoted on several different keys, and picking the wrong one silently
    produces a cartesian blow-up (one row per id-combination x measure,
    mostly ``NaN``). ``names_from``/``values_from`` must therefore be either
    supplied explicitly, or unambiguously derivable — a single
    ``value_columns`` entry plus a ``measure_name_column`` in the attached
    metadata.

    Args:
        df: A DataFrame from ``Client.get()``/``Client.get_local()`` with
            ``layout = "long"`` metadata attached.
        names_from: Column to spread into new column names. ``None``
            (default) derives it from the dataset's ``measure_name_column``
            metadata.
        values_from: Column to fill the new wide columns with. ``None``
            (default) derives it from ``value_columns`` metadata — only
            when that list has exactly one entry present in `df`.
        id_cols: Columns to keep as row keys. ``None`` (default) uses the
            dataset's ``id_columns`` metadata intersected with `df`'s
            columns, falling back to every other column.
        values_fill: Value to fill missing cells with after pivoting
            (forwarded to ``DataFrame.fillna`` on the pivoted result).
        name: Dataset identifier, used only to fetch metadata when `df` has
            none attached.
        client: A :class:`Client` instance, used only with `name` to fetch
            metadata when none is attached.

    Returns:
        A wide-format ``pd.DataFrame``.

    Raises:
        ReshapeError: no layout metadata, layout is ``feature``/``entity``,
            a geometry column is present, layout is not ``long``,
            ``names_from``/``values_from`` cannot be resolved unambiguously,
            or a resolved column is missing from `df`. ``ValueError``
            (raised by ``pandas.DataFrame.pivot``) propagates unchanged if
            ``id_cols``/``names_from`` do not uniquely identify rows — this
            is intentional: silently aggregating duplicates would hide a
            real data problem.

    Example:
        >>> long = client.get("some_long_series_table")
        >>> wide = pivot_wider(long)
    """
    meta = _get_eolas_meta(df, name=name, client=client)
    layout = _reshape_guard(df, meta, name=name)
    ds_name = _dataset_label(df, name)

    if layout != "long":
        hint = (
            " This dataset is already wide — see pivot_longer() to go the " "other way."
            if layout == "wide"
            else ""
        )
        raise ReshapeError(
            f"Cannot pivot_wider {ds_name!r}: layout is {layout!r}, not "
            f"'long'.{hint}"
        )

    if names_from is None or values_from is None:
        derived_names = names_from or meta.get("measure_name_column")
        value_cols_meta = [
            c for c in _as_list(meta.get("value_columns")) if c in df.columns
        ]
        derived_values = values_from or (
            value_cols_meta[0] if len(value_cols_meta) == 1 else None
        )
        if not derived_names or not derived_values:
            raise ReshapeError(
                f"Cannot pivot_wider {ds_name!r}: names_from/values_from were "
                "not supplied and cannot be derived unambiguously. Metadata "
                f"has measure_name_column={meta.get('measure_name_column')!r} "
                f"and value_columns={value_cols_meta!r}. Supply both "
                "explicitly: pivot_wider(df, names_from=..., values_from=...)."
            )
        names_from = names_from or derived_names
        values_from = values_from or derived_values

    if names_from not in df.columns:
        raise ReshapeError(
            f"Cannot pivot_wider {ds_name!r}: names_from column {names_from!r} "
            "not found in this DataFrame."
        )
    if values_from not in df.columns:
        raise ReshapeError(
            f"Cannot pivot_wider {ds_name!r}: values_from column "
            f"{values_from!r} not found in this DataFrame."
        )

    if id_cols is None:
        meta_id_cols = [c for c in _as_list(meta.get("id_columns")) if c in df.columns]
        resolved_id_cols = [
            c for c in meta_id_cols if c not in (names_from, values_from)
        ]
        if not resolved_id_cols:
            resolved_id_cols = [
                c for c in df.columns if c not in (names_from, values_from)
            ]
    else:
        resolved_id_cols = list(id_cols)

    wide = df.pivot(index=resolved_id_cols, columns=names_from, values=values_from)
    wide = wide.reset_index()
    wide.columns.name = None
    if values_fill is not None:
        wide = wide.fillna(values_fill)
    # Result is now wide — re-declare it so pivot_longer() can reverse it. The new
    # measure columns are everything that isn't a retained id column.
    new_value_cols = [c for c in wide.columns if c not in resolved_id_cols]
    return _attach_reshaped_meta(
        wide,
        df,
        layout="wide",
        time_columns=_as_list(meta.get("time_columns")),
        id_columns=resolved_id_cols,
        value_columns=new_value_cols,
        measure_name_column="",
    )
