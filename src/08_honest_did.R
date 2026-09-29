# 08_honest_did.R — Rambachan & Roth (2023) sensitivity of the RQ1 event study
# to violations of parallel trends ("Honest DiD"), relative-magnitudes bound.
#
# Reads the step 07 event-study coefficients and their clustered covariance
# matrix (aggregate results only, no microdata) and reports robust 95%
# confidence sets for the average post-period coefficient (2022Q4-2025Q4) as
# Mbar grows. Mbar bounds each post-period violation of parallel trends by
# Mbar times the largest pre-period one; the breakdown value is the largest
# Mbar at which the robust set still excludes zero.
#
# Uses the authors' reference implementation, the HonestDiD R package.
#
# Usage:
#   Rscript src/08_honest_did.R --results data/out/results \
#       [--file rq1_task_composition.csv] [--mbar-max 2] [--mbar-step 0.1]

suppressPackageStartupMessages(library(HonestDiD))

args <- commandArgs(trailingOnly = TRUE)
opt <- list(results = "data/out/results", file = "rq1_task_composition.csv",
            `mbar-max` = "2", `mbar-step` = "0.1", target = "average")
i <- 1
while (i <= length(args)) {
  key <- sub("^--", "", args[i])
  if (!key %in% names(opt)) stop("unknown option ", args[i])
  opt[[key]] <- args[i + 1]
  i <- i + 2
}

res_path <- file.path(opt$results, opt$file)
vcov_path <- file.path(opt$results, sub("\\.csv$", "_vcov.csv", opt$file))
if (!file.exists(vcov_path)) stop(vcov_path, " not found: re-run step 07, which now saves it")

res <- read.csv(res_path, stringsAsFactors = FALSE)
res <- res[res$period %in% c("pre", "post"), ]
res <- res[order(res$quarter), ]
V <- as.matrix(read.csv(vcov_path, row.names = 1, check.names = FALSE))
V <- V[res$term, res$term]

# HonestDiD needs consecutive quarters with the reference quarter between the
# last pre and first post quarter (it is omitted, so it is not in betahat).
q_index <- function(q) as.integer(substr(q, 1, 4)) * 4 + as.integer(substr(q, 6, 6))
if (any(diff(q_index(res$quarter)) != 1 &
        !(res$period[-nrow(res)] == "pre" & res$period[-1] == "post"))) {
  stop("event-study quarters are not consecutive (e.g. the pandemic-dropped ",
       "specification); relative magnitudes need an unbroken pre-period")
}

n_pre <- sum(res$period == "pre")
n_post <- sum(res$period == "post")
l_vec <- if (opt$target == "average") rep(1 / n_post, n_post) else
  basisVector(as.integer(opt$target), n_post)
mbar <- seq(0, as.numeric(opt$`mbar-max`), by = as.numeric(opt$`mbar-step`))

cat(sprintf("[honest] %s: %d pre and %d post quarters, target = %s post coefficient\n",
            opt$file, n_pre, n_post, opt$target))
robust <- createSensitivityResults_relativeMagnitudes(
  betahat = res$estimate, sigma = V, numPrePeriods = n_pre, numPostPeriods = n_post,
  l_vec = l_vec, Mbarvec = mbar, alpha = 0.05)
original <- constructOriginalCS(betahat = res$estimate, sigma = V, numPrePeriods = n_pre,
                                numPostPeriods = n_post, l_vec = l_vec, alpha = 0.05)

point <- sum(l_vec * res$estimate[res$period == "post"])
out <- rbind(
  data.frame(Mbar = NA, lb = original$lb, ub = original$ub, method = "Original"),
  data.frame(Mbar = robust$Mbar, lb = robust$lb, ub = robust$ub, method = robust$method))
out$excludes_zero <- out$lb > 0 | out$ub < 0
out$estimate <- point
out$target <- opt$target
out$honestdid_version <- as.character(packageVersion("HonestDiD"))

excl <- out[!is.na(out$Mbar), ]
breakdown <- if (!excl$excludes_zero[1]) NA else {
  first_in <- which(!excl$excludes_zero)
  if (length(first_in) == 0) Inf else excl$Mbar[first_in[1] - 1]
}
out$breakdown_Mbar <- breakdown

dest <- file.path(opt$results, sub("\\.csv$", "_honest_did.csv", opt$file))
write.csv(out, dest, row.names = FALSE)
print(out[, c("Mbar", "lb", "ub", "method", "excludes_zero")], row.names = FALSE)
if (is.na(breakdown)) {
  cat("[honest] breakdown Mbar: none (the robust set includes zero already at Mbar = 0)\n")
} else if (is.infinite(breakdown)) {
  cat(sprintf("[honest] breakdown Mbar: above %s (zero excluded over the whole grid)\n",
              opt$`mbar-max`))
} else {
  cat(sprintf("[honest] breakdown Mbar = %s (grid step %s)\n", breakdown, opt$`mbar-step`))
}
cat("[honest] written to", dest, "\n")
