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
#       [--file rq1_task_composition.csv] [--mbar-max 2] [--mbar-step 0.25]
#       [--refine-to 0.01]   (breakdown bisected to this precision; Inf = no refining)

suppressPackageStartupMessages(library(HonestDiD))

args <- commandArgs(trailingOnly = TRUE)
opt <- list(results = "data/out/results", file = "rq1_task_composition.csv",
            `mbar-max` = "2", `mbar-step` = "0.25", `refine-to` = "0.01",
            target = "average")
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
# The robust set is found by testing points on a grid. HonestDiD's default grid
# is centred on zero, so a precisely estimated non-zero effect can fall
# outside it and every point is rejected. Centre it on the point estimate and
# widen it while a bound sits on the grid's edge (an open-ended set).
point <- sum(l_vec * res$estimate[res$period == "post"])
post_idx <- which(res$period == "post")
sd_theta <- sqrt(drop(t(l_vec) %*% V[post_idx, post_idx] %*% l_vec))

robust_at <- function(m) {
  half <- 20 * sd_theta
  for (attempt in 1:4) {
    r <- createSensitivityResults_relativeMagnitudes(
      betahat = res$estimate, sigma = V, numPrePeriods = n_pre, numPostPeriods = n_post,
      l_vec = l_vec, Mbarvec = m, alpha = 0.05,
      grid.lb = point - half, grid.ub = point + half, gridPoints = 1000)
    tol <- 2 * half / 999
    edge <- is.finite(r$lb) && is.finite(r$ub) &&
      (r$lb <= point - half + tol || r$ub >= point + half - tol)
    if (!edge || attempt == 4) break
    half <- half * 4
  }
  data.frame(Mbar = m, lb = r$lb, ub = r$ub, method = r$method, at_grid_edge = edge)
}

# Each Mbar is independent, so run them on all cores. Then refine the
# breakdown value by bisection between the last Mbar that excludes zero and
# the first that includes it, instead of paying for a fine grid everywhere.
cores <- max(1, parallel::detectCores())
t0 <- Sys.time()
run_many <- function(ms) {
  parts <- parallel::mclapply(ms, robust_at, mc.cores = cores)
  bad <- vapply(parts, inherits, logical(1), what = "try-error")
  if (any(bad)) stop("HonestDiD failed at Mbar ", paste(ms[bad], collapse = ", "), ": ",
                     as.character(parts[bad][[1]]))
  do.call(rbind, parts)
}
robust <- run_many(mbar)
excludes0 <- function(d) d$lb > 0 | d$ub < 0
refine_to <- as.numeric(opt$`refine-to`)
repeat {
  robust <- robust[order(robust$Mbar), ]
  e <- excludes0(robust)
  if (!is.finite(refine_to) || !e[1] || all(e)) break
  hi <- robust$Mbar[which(!e)[1]]
  lo <- max(robust$Mbar[robust$Mbar < hi & e])
  if (hi - lo <= refine_to) break
  robust <- rbind(robust, run_many((lo + hi) / 2))
}
cat(sprintf("[honest] %d Mbar values on %d cores in %.1f min\n", nrow(robust), cores,
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))
original <- constructOriginalCS(betahat = res$estimate, sigma = V, numPrePeriods = n_pre,
                                numPostPeriods = n_post, l_vec = l_vec, alpha = 0.05)

# HonestDiD reports an empty set (+Inf, -Inf) when no grid point is accepted,
# which also happens when its LP solver fails; never write that as a result.
if (any(!is.finite(robust$lb) | !is.finite(robust$ub))) {
  print(robust)
  stop("HonestDiD returned non-finite bounds (solver failure or empty set); ",
       "check the log above")
}

out <- rbind(
  data.frame(Mbar = NA, lb = original$lb, ub = original$ub, method = "Original",
             at_grid_edge = FALSE),
  robust)
if (any(out$at_grid_edge)) {
  cat(sprintf("[honest] note: at Mbar %s the robust set reaches the widest grid tried (point +/- %.3g), so that bound is open-ended\n",
              paste(out$Mbar[out$at_grid_edge], collapse = ", "), 20 * 4^3 * sd_theta))
}
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
  cat(sprintf("[honest] breakdown Mbar = %.3g (next value tested includes zero; precision %s)\n",
              breakdown, opt$`refine-to`))
}
cat("[honest] written to", dest, "\n")
