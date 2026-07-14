import numpy as np

from cellpose import tta


def test_candidate_regions_discards_small_components():
    cellprob = np.zeros((16, 16), np.float32)
    cellprob[1:3, 1:3] = 1
    cellprob[8:14, 8:14] = 1
    regions = tta._candidate_regions(cellprob, threshold=0, min_size=10, max_points=16)
    assert len(regions) == 1
    assert len(regions[0]) == 16


def test_region_flow_tta_returns_finite_flow_without_mutating_input():
    dP = np.zeros((2, 32, 32), np.float32)
    dP[0, 8:24, 8:24] = 1
    cellprob = np.zeros((32, 32), np.float32)
    cellprob[8:24, 8:24] = 1
    refined = tta.refine_flow_regions(
        dP, cellprob, steps=1, niter=2, min_region_size=10, max_points=32,
        device="cpu")
    assert refined.shape == dP.shape
    assert np.isfinite(refined).all()
    assert np.array_equal(dP[0, 8:24, 8:24], np.ones((16, 16), np.float32))
