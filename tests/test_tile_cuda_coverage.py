from __future__ import annotations

from neuralfn.tile_cuda import coverage_report


def test_tile_cuda_coverage_has_no_unaccounted_inventory() -> None:
    report = coverage_report()

    assert report.complete
    assert report.accounted == report.total_inventory
    assert report.missing == ()
    assert report.by_status.get("torch_fallback", 0) == 0
    assert report.by_status["tile"] >= 129
