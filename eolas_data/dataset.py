from __future__ import annotations

import pandas as pd


class Dataset(pd.DataFrame):
    """A pandas DataFrame with eolas dataset metadata.

    Behaves exactly like a DataFrame — all pandas operations work normally.
    Extra attributes:
        eolas_name:   Dataset identifier (e.g. ``"nz_cpi"``).
        eolas_source: Data source label (e.g. ``"Stats NZ"``).
    """

    _metadata = ["eolas_name", "eolas_source"]

    @property
    def _constructor(self):
        return Dataset

    def __repr__(self) -> str:
        name   = getattr(self, "eolas_name",   "") or ""
        source = getattr(self, "eolas_source", "") or ""
        if name:
            header = f"# Dataset: {name}"
            if source:
                header += f" [{source}]"
            header += f"\n# {len(self)} rows\n"
            return header + pd.DataFrame.__repr__(self)
        return pd.DataFrame.__repr__(self)

    # ------------------------------------------------------------------
    # plot_dataset() was removed in v1.3.0.
    #
    # It auto-picked `date` and `value` columns and drew a single matplotlib
    # line — but datasets with a dimension column (multiple series per date)
    # produced silent zigzag traces. Rather than ship a helper that has to
    # know each dataset's shape, plotting is now the caller's responsibility.
    # `Dataset` subclasses `DataFrame`, so any matplotlib / seaborn / plotly
    # workflow works straight out of the box:
    #
    #     import matplotlib.pyplot as plt
    #     df.plot(x="date", y="value")
    #
    # See README for one-liners.
    # ------------------------------------------------------------------
