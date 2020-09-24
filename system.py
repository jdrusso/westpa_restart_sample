import numpy as np
from west import WESTSystem
from westpa.binning import RectilinearBinMapper

import logging
log = logging.getLogger(__name__)
log.debug('loading module %r' % __name__)

pcoord_len = 51
pcoord_dtype = np.float32


class System(WESTSystem):
    def initialize(self):
        self.pcoord_ndim = 1
        self.pcoord_len = pcoord_len
        self.pcoord_dtype = pcoord_dtype

        binbounds = [0.00, 1.0, 1.25, 1.5, 1.72, 2.0, 2.25, 2.5, 2.58, 2.65, 2.72, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4,
            3.45, 3.5, 3.55, 3.6, 3.65, 3.7, 3.72, 3.75, 3.77, 3.8, 3.825, 3.85, 3.875, 3.9, 3.925, 3.95, 3.975, 4.0,
            4.05, 4.1, 4.15, 4.2, 4.25, 4.3, 4.35, 4.4, 4.45, 4.5, 4.55, 4.6, 4.7, 4.8, 4.85, 4.9, 4.95, 5.0, 5.05,
            5.1, 5.18, 5.25, 5.3, 5.35, 5.4, 5.45, 5.5, 5.55, 5.6, 5.65, 5.7, 5.75, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4,
            6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 11.0, 11.5, 12.0, 12.5, 12.75, 13.0, 13.25, 13.5, 13.75, 14.0,
            14.5, 15.0, 15.5, 15.75, 16.0, 16.25, 16.5, 17.0, 17.3,  17.6, 18.0, 18.3, 18.6, 19.0, 19.5, 20, 'inf']

        self.bin_mapper = RectilinearBinMapper([binbounds])
        self.bin_target_counts = np.empty((self.bin_mapper.nbins,), np.int)
        self.bin_target_counts[...] = 10
