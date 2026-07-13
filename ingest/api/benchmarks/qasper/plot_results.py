"""Reuse the dataset-agnostic SVG/HTML plotting code used by SciQ."""

# Importing common adds ``ingest/api`` to sys.path when this file is executed
# directly, just like the other QASPER entry points.
from common import API_ROOT as _API_ROOT  # noqa: F401
from benchmarks.sciq.plot_results import main, plot

__all__ = ["main", "plot"]


if __name__ == "__main__":
    main()
