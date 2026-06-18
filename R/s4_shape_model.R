# Stage 4: morphometric shape model with Momocs (Elliptical Fourier + PCA).
# For each class builds an Out, runs efourier(norm=TRUE), then PCA, and exports
# the generative pieces (mean coefficients, eigenvectors, sdev, scores) as CSV so
# the Python generator can sample PC scores and invert the EFA itself.

suppressWarnings(suppressMessages({
  .libPaths(Sys.getenv("R_LIBS_USER"))
  library(Momocs)
}))

GLYPHS <- "C:/matrix/data/glyphs"
MODEL  <- "C:/matrix/data/model"
REPORTS <- "C:/matrix/reports"
dir.create(MODEL, showWarnings = FALSE, recursive = TRUE)
K <- 16  # harmonics

read_outlines <- function(name) {
  df <- read.csv(file.path(GLYPHS, paste0("outlines_", name, "_al.csv")))
  ids <- unique(df$id)
  coo <- lapply(ids, function(i) {
    s <- df[df$id == i, ]
    s <- s[order(s$k), ]
    as.matrix(s[, c("x", "y")])
  })
  names(coo) <- ids
  # drop degenerate (tiny) outlines: keep those with bbox diag above 5th pct
  diag <- sapply(coo, function(m) sqrt(diff(range(m[,1]))^2 + diff(range(m[,2]))^2))
  keep <- diag >= quantile(diag, 0.03)
  list(coo = coo[keep], ids = ids[keep])
}

fit_class <- function(name) {
  cat("\n== class", name, "==\n")
  ro <- read_outlines(name)
  out <- Out(ro$coo)
  # outlines are pre-aligned (centered, scaled, start fixed) -> no Momocs norm,
  # so the natural upright orientation + slant stay in the coefficients.
  ef0 <- efourier(out, nb.h = K, norm = FALSE, start = TRUE)
  pca0 <- PCA(ef0)
  # --- robust outlier trim: drop shapes far from the class core in PC space ---
  ve0 <- (pca0$sdev^2) / sum(pca0$sdev^2)
  k <- max(3, which(cumsum(ve0) >= 0.90)[1])
  md2 <- rowSums(sweep(pca0$x[, 1:k, drop = FALSE], 2, pca0$sdev[1:k], "/")^2)
  keep <- md2 <= qchisq(0.985, df = k)
  cat(sprintf("  outlier trim: kept %d / %d (k=%d)\n", sum(keep), length(keep), k))

  coo_keep <- ro$coo[keep]
  ef <- efourier(Out(coo_keep), nb.h = K, norm = FALSE, start = TRUE)
  coe <- ef$coe
  pca <- prcomp(coe, center = TRUE, scale. = FALSE)  # export-grade PCA on clean set
  varexp <- (pca$sdev^2) / sum(pca$sdev^2)
  npc99 <- which(cumsum(varexp) >= 0.99)[1]
  cat(sprintf("  n=%d  harmonics=%d  coe_dim=%d  PCs for 99%%=%d\n",
              nrow(coe), K, ncol(coe), npc99))
  cat("  top var explained:", paste0(round(100*varexp[1:min(6,length(varexp))],1), "%"), "\n")

  write.csv(data.frame(mean = pca$center), file.path(MODEL, paste0("efa_", name, "_mean.csv")), row.names = FALSE)
  write.csv(as.data.frame(pca$rotation), file.path(MODEL, paste0("efa_", name, "_rotation.csv")), row.names = FALSE)
  write.csv(data.frame(sdev = pca$sdev), file.path(MODEL, paste0("efa_", name, "_sdev.csv")), row.names = FALSE)
  write.csv(as.data.frame(pca$x), file.path(MODEL, paste0("efa_", name, "_scores.csv")), row.names = FALSE)
  write.csv(as.data.frame(coe), file.path(MODEL, paste0("efa_", name, "_coe.csv")), row.names = FALSE)
  # keep ids of kept glyphs (for joint 0 outer/inner pairing downstream)
  write.csv(data.frame(id = ro$ids[keep]), file.path(MODEL, paste0("efa_", name, "_ids.csv")), row.names = FALSE)
  saveRDS(list(ef = ef, pca = pca, K = K), file.path(MODEL, paste0("efa_", name, ".rds")))

  # diagnostic plots (Momocs on the clean set)
  mpca <- PCA(ef)
  png(file.path(REPORTS, paste0("s4_", name, "_pca.png")), 900, 700)
  tryCatch(plot_PCA(mpca, title = paste("PCA", name)), error = function(e) plot(mpca))
  dev.off()
  png(file.path(REPORTS, paste0("s4_", name, "_meanshape.png")), 700, 700)
  ms <- MSHAPES(ef)
  coo_plot(ms, main = paste("mean shape", name))
  dev.off()
  invisible(npc99)
}

info <- list()
for (nm in c("1", "blob", "0_outer", "0_inner")) {
  info[[nm]] <- fit_class(nm)
}
write.csv(data.frame(K = K), file.path(MODEL, "efa_info.csv"), row.names = FALSE)
cat("\nDONE shape model. Models in", MODEL, "\n")
