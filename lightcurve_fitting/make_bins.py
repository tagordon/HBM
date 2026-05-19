import numpy as np

#  bins_short  bins_long  bins_center  bins_width

start = 2.8627040
end = 5.0627574
width = 0.0101387 * 2

bins_short = np.arange(start, end, width)
bins_long = bins_short + width
bins_center = bins_short + width * 0.5
width = np.ones_like(bins_short) * width

np.savetxt('bins_395M.txt', np.vstack([bins_short, bins_long, bins_center, width]).T)
