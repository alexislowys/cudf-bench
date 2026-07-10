"""Backend adapters for pandas, polars, and cudf.

Each backend knows how to convert a generated pandas table to its native
format, force pending work to complete (GPU work is async), and report
its version and device for the results log.
"""

from __future__ import annotations

import platform


class Backend:
    name: str = "base"

    def version(self) -> str:
        raise NotImplementedError

    def from_pandas(self, df):
        raise NotImplementedError

    def sync(self) -> None:
        """Block until all pending work for this backend has finished."""

    def device(self) -> str:
        return f"cpu ({platform.processor() or platform.machine()})"


class PandasBackend(Backend):
    name = "pandas"

    def version(self) -> str:
        import pandas

        return pandas.__version__

    def from_pandas(self, df):
        return df.copy()


class PolarsBackend(Backend):
    name = "polars"

    def version(self) -> str:
        import polars

        return polars.__version__

    def from_pandas(self, df):
        import polars

        return polars.from_pandas(df)


class CudfBackend(Backend):
    name = "cudf"

    def __init__(self) -> None:
        try:
            import cudf  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "cudf is not installed. This backend needs an NVIDIA GPU "
                "(e.g. Google Colab with a T4 runtime: "
                "`pip install cudf-cu12`)."
            ) from exc

    def version(self) -> str:
        import cudf

        return cudf.__version__

    def from_pandas(self, df):
        import cudf

        return cudf.from_pandas(df)

    def sync(self) -> None:
        import cupy

        cupy.cuda.runtime.deviceSynchronize()

    def device(self) -> str:
        import cupy

        props = cupy.cuda.runtime.getDeviceProperties(0)
        return props["name"].decode()


_BACKENDS = {
    "pandas": PandasBackend,
    "polars": PolarsBackend,
    "cudf": CudfBackend,
}


def get_backend(name: str) -> Backend:
    try:
        cls = _BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"Unknown backend {name!r}. Choose from: {sorted(_BACKENDS)}"
        ) from None
    return cls()
