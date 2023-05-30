from typing import Optional

import numpy as np
from numpy.typing import NDArray

from connectivity.PMI import SPMI_1epoch


def initialize_matrix(data: NDArray, out_path: Optional[str] = None):
    assert len(data.shape) == 2, "Data must be a 2D array, shape: (channel_num, sample_num)"
    spmi = SPMI_1epoch(data, 5, 1)
    for i in range(spmi.shape[0]):
        for j in range(0, i):
            spmi[i, j] = spmi[j, i]
    if out_path is not None:
        np.savetxt(out_path, spmi, delimiter=",")   
    return spmi