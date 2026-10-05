# Methods

## Walker Delta T/P/F convention

For `T` satellites in `P` evenly spaced circular orbital planes, with
`S = T / P` satellites per plane:

- Plane index p = 0 .. P - 1
- Slot index  s = 0 .. S - 1
- RAAN of plane p:

  Ω_p = p · 360° / P

- Argument of latitude of satellite (p, s) at t = 0:

  u_{p,s}(0) = s · 360° / S  +  p · F · 360° / T  +  u_0

F is restricted to the range [0, P-1]; F and F + P give identical constellations.
This matches Walker's original 1984 definition and the convention used by STK
and Vallado.

For the default **Walker Delta 24/6/1**:

- 24 satellites, 6 planes, 4 satellites per plane
- RAAN spacing 60°
- In-plane spacing 90°
- Walker phase increment 15° between adjacent planes

## Propagator

Version 1 uses an analytic two-body circular propagator.  For each satellite:

- n = sqrt(μ / a³)
- u(t) = u_0 + n · t
- r(t) is the ECI position vector obtained by the standard rotation
  R_3(−Ω) · R_1(−i) · R_3(−u) · [a, 0, 0]ᵀ

J2, drag, higher-order gravity and SGP4 are deliberately omitted: these are
designed constellations, not TLE-derived spacecraft.  The `Propagator`
protocol lets a J2 secular model be added later without touching anything
above this layer.

Constants (SI):

- μ = 3.986004418 × 10¹⁴ m³/s²
- R_Earth = 6 378 137 m
- ω_Earth = 7.2921150 × 10⁻⁵ rad/s (sidereal rate)

## Earth rotation

ECI positions are rotated to ECEF by angle θ(t) = θ_0 + ω_Earth · t.  θ_0 is a
user input; since no absolute epoch is specified it just shifts the maps in
longitude.  Long-term statistics do not depend on it.

## Ground grid

Cell-centred latitude/longitude grid with weights w ∝ (sin φ_n − sin φ_s) · Δλ,
normalised so the weights sum to 1 over the covered region.  All global
percentages, mean statistics and histograms are area weighted.  A uniform-grid
percentage without area weighting over-weights the poles by a factor of
1 / cos φ and should not be used.

## Visibility

For a spherical Earth of radius R_E and a satellite at altitude h, a ground
point sees the satellite above elevation ε if and only if

λ ≤ λ_max(h, ε) = arccos( R_E · cos ε / (R_E + h) ) − ε

where λ is the Earth-centre angle between the satellite's ECEF position and
the ground point.  This is exact for the spherical model and lets the hot
loop be a single matrix multiply:

- cos λ_(sat, ground) = r̂_sat · r̂_ground
- visible if cos λ ≥ cos λ_max

The explicit elevation formula sin ε = û · ρ / |ρ| is also implemented and is
used by the unit tests to prove the two tests give identical results.

## Coverage metrics

- **Instantaneous coverage**: at least `k` satellites visible from the point
  right now.  `k = 1` by default.
- **Coverage gap**: a maximal interval during which fewer than `k` satellites
  are visible.  Gaps bounded by the start or end of the simulation window are
  **censored**: their length is a lower bound.  Censored gaps are included in
  the maximum-gap metric (which is itself a lower bound in that case) and in
  the response-time integral, but excluded from the revisit mean/max.
- **Revisit interval**: start-to-start time between successive access
  intervals at a point.  Requires at least two observed access starts.  A
  point that is continuously covered or never covered has an undefined revisit
  (shown as "n/a").
- **Mean response time**: time-averaged wait until the next access,
  Σ g_i² / (2 · T).  Includes censored gaps; this matches the STK formulation.

Durations are only resolved to within one time step `dt`.  At 550 km with a
10° elevation mask the maximum pass duration is about 8 min, so `dt = 60 s`
is usually fine.  A warning appears in the UI if `dt` would be too coarse.

## Software notes

- All angles internal to the package are radians.
- All distances internal to the package are metres.
- All times internal to the package are seconds.
- The streaming coverage engine keeps O(G) state, where G is the number of
  ground-grid points.  Memory does not scale with the number of time steps.
- Chunk sizes are chosen so that the peak float allocation per chunk stays
  below about 160 MB.
