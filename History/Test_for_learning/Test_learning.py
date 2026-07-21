#%%
import numpy as np
from scipy.stats import wilcoxon

# Example data: differences between paired measurements
shifted_i = np.array([-2.1, -1.8, -2.5, -3.0, -2.7, -1.9])

# Perform one-sided Wilcoxon signed-rank test
# H0: median(shifted_i) == 0
# H1: median(shifted_i) > 0
try:
    stat_i, p_i = wilcoxon(shifted_i, alternative='greater')
    print(f"Wilcoxon statistic: {stat_i}")
    print(f"p-value: {p_i}")
except ValueError as e:
    print(f"Error: {e}")
