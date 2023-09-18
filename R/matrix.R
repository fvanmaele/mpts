library(readr)
options(scipen=6)
matrix <- read_csv('matrix_norm_diag.csv')

# coverage(max ST) >= coverage(max LF) must hold for every matrix
all(matrix['max_st'] >= matrix['max_lf'])

# retrieve indices
idx_1a = which(matrix['diag'] == 1)
idx_1b = which(matrix['diag_dom'] == 1)
idx_0a = which(matrix['diag'] == 0)
idx_0b = which(matrix['diag_dom'] == 0)

# matrices with diagonal coverage 1 must be diagonally dominant
stopifnot(length(setdiff(idx_1a, idx_1b)) == 0)

# but the converse does not hold (example: swang1)
diff_1b = setdiff(idx_1b, idx_1a)
diff_0b = setdiff(idx_0b, idx_0a)
diff_0a = setdiff(idx_0a, idx_0b)

# Find set of matrices where coverage(max LF) <= coverage(max ST) - 0.2
matrix_2 <- matrix[which(matrix['max_lf'] <= matrix['max_st'] - 0.2), ]

# Limit to matrices with coverage(max ST) < 0.9
matrix_2 <- matrix_2[which(matrix_2['max_st'] < 0.9), ]
