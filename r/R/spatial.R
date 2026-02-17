#' Calculate 2-Point Correlation Function
#'
#' Computes the spatial 2-Point Correlation Function (2PCF) to analyze galaxy clustering.
#'
#' @param ra Vector of Right Ascension coordinates.
#' @param dec Vector of Declination coordinates.
#' @param r_max Maximum separation distance (degrees).
#' @param n_bins Number of bins.
#' @return Data frame containing r (separation) and xi (correlation).
#' @import spatstat
#' @export
calculate_spatial_correlation <- function(ra, dec, r_max = 5.0, n_bins = 20) {
  
  # Convert to point pattern
  # Assume approximate flat sky for small fields or project
  # For rigorous cosmology, we need 3D distance or angular separation on sphere.
  # Using simple Euclidean approximation for now as placeholder for 'spatstat' usage.
  
  window <- spatstat.geom::owin(range(ra), range(dec))
  pp <- spatstat.geom::ppp(ra, dec, window = window)
  
  # K-function estimation
  k_est <- spatstat.explore::Kest(pp, r = seq(0, r_max, length.out = n_bins))
  
  # Convert K to xi (approximation: xi = K/(pi*r^2) - 1 for 2D Poisson?)
  # Actually, Pair Correlation Function g(r) is related to 1 + xi(r).
  g_est <- spatstat.explore::pcf(k_est, method = "b")
  
  data.frame(
    r = g_est$r,
    xi = g_est$pcf - 1 # Excess probability
  )
}
