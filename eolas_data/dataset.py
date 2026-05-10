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

    def plot_dataset(self, ax=None, **kwargs):
        """Quick line chart using matplotlib.

        Returns the matplotlib Axes object so you can customise further.
        Requires matplotlib: ``pip install eolas-data[plot]``.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plot_dataset(). "
                "Install with: pip install eolas-data[plot]"
            )

        date_col  = "date"  if "date"  in self.columns else self.columns[0]
        value_col = "value" if "value" in self.columns else self.columns[1]

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        ax.plot(self[date_col], self[value_col], color="#2563eb", linewidth=1.5, **kwargs)

        name   = getattr(self, "eolas_name",   "") or ""
        source = getattr(self, "eolas_source", "") or ""

        if name:
            ax.set_title(name, fontweight="bold", fontsize=13)
        ax.set_xlabel("")
        ax.spines[["top", "right"]].set_visible(False)

        caption = f"Source: {source} · eolas.fyi" if source else "eolas.fyi"
        ax.figure.text(0.99, 0.01, caption, ha="right", fontsize=8, color="#9ca3af")

        plt.tight_layout()
        return ax
