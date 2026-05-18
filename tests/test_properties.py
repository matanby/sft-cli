"""Hypothesis property tests for invariants that should hold for any input."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from sft.utils.glob import tensor_matches
from sft.utils.tensor_io import read_tensors, write_file

# ---------- glob: dotted patterns never cross segments without `**` ----------


_SEGMENT = st.text(
    alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z")),
    min_size=1,
    max_size=6,
)
_NAME = st.lists(_SEGMENT, min_size=1, max_size=4).map(lambda parts: ".".join(parts))


@given(name=_NAME)
def test_single_star_segment_never_crosses_dot(name: str) -> None:
    """`a.*.b` against `a.x.y.b` (3 inner segments) must NOT match."""
    parts = name.split(".")
    if len(parts) < 2:
        return  # not enough structure
    pattern = ".".join(["*" if i == 1 else p for i, p in enumerate(parts)])
    assert tensor_matches(name, pattern)
    inflated = ".".join([parts[0], parts[1], "extra", *parts[2:]])
    if "extra" not in parts:
        assert not tensor_matches(inflated, pattern)


@given(name=_NAME)
def test_dotless_pattern_matches_across_segments(name: str) -> None:
    """A pattern with no `.` should match a substring anywhere in the name."""
    non_dots = [c for c in name if c != "."]
    if not non_dots:
        return
    mid = non_dots[len(non_dots) // 2]
    pattern = f"*{mid}*"
    assert tensor_matches(name, pattern)


# ---------- tensor_io: save/load is a bytes-identical round-trip ----------


@st.composite
def _np_array(draw) -> np.ndarray:
    dtype = draw(
        st.sampled_from(
            [np.float32, np.float16, np.float64, np.int8, np.int16, np.int32, np.int64]
        )
    )
    shape = draw(
        st.lists(st.integers(min_value=1, max_value=5), min_size=0, max_size=3)
    )
    if dtype in (np.float32, np.float16, np.float64):
        elements = st.floats(
            allow_nan=False,
            allow_infinity=False,
            width=32 if dtype is np.float32 else 64,
        )
    else:
        info = np.iinfo(dtype)
        elements = st.integers(min_value=int(info.min), max_value=int(info.max))
    flat = draw(
        st.lists(
            elements,
            min_size=int(np.prod(shape)) if shape else 1,
            max_size=int(np.prod(shape)) if shape else 1,
        )
    )
    return (
        np.array(flat, dtype=dtype).reshape(shape)
        if shape
        else np.array(flat[0], dtype=dtype)
    )


@given(arr=_np_array())
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_tensor_io_round_trip(
    arr: np.ndarray, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """write_file then read_tensors recovers the same array bit-for-bit."""
    if arr.ndim == 0:
        return  # safetensors doesn't allow 0-d
    path = tmp_path_factory.mktemp("rt") / "x.safetensors"
    write_file(path, {"x": arr})
    got = read_tensors(path)["x"]
    np.testing.assert_array_equal(got, arr)
    assert got.dtype == arr.dtype
