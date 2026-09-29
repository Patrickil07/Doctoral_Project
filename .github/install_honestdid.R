# Installs the HonestDiD R package (Rambachan & Roth) for step 08 on GitHub
# runners: from CRAN when available, otherwise from the authors' GitHub.
options(repos = c(CRAN = Sys.getenv("RSPM", "https://cloud.r-project.org")))
if (!requireNamespace("HonestDiD", quietly = TRUE)) {
  try(install.packages("HonestDiD"), silent = TRUE)
}
if (!requireNamespace("HonestDiD", quietly = TRUE)) {
  if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
  remotes::install_github("asheshrambachan/HonestDiD", upgrade = "never")
}
cat("HonestDiD", as.character(packageVersion("HonestDiD")), "\n")
