import numpy as np

from cellpose import dynamics


def test_endpoint_histogram_counts_each_candidate_pixel():
    dP = np.zeros((2, 8, 8), dtype=np.float32)
    cellprob = np.zeros((8, 8), dtype=np.float32)
    cellprob[2:4, 3:6] = 1

    histogram, endpoints, inds = dynamics.endpoint_histogram(
        dP, cellprob, niter=1, rpad=2, return_endpoints=True)

    assert histogram.shape == (12, 12)
    assert histogram.sum() == len(inds[0]) == len(endpoints)
    # Zero flow keeps candidate pixels at their original positions.
    assert histogram[4, 5] == 1
