# Time-Mapped Spectral Methods for Singularity Formation in Inviscid Burgers' Equation

## Overview

This project investigates the formation of singularities in nonlinear partial differential equations using the inviscid Burgers' equation:

\[
\frac{\partial u}{\partial t}
+u\frac{\partial u}{\partial x}=0.
\]

The project explores how numerical methods can capture the steepening of solution gradients as a shock forms in finite time. Several numerical approaches are implemented in Python and compared to investigate their accuracy, convergence and ability to resolve the solution close to singularity formation.

## Numerical Methods

The project investigates the following methods:

- **Linear advection:** Analytical solutions and a first-order upwind scheme with forward Euler time integration.
- **Inviscid Burgers' equation:** Numerical solutions, shock-time estimation and analysis of gradient blow-up.
- **Time mapping:** A nonlinear time transformation that maps the finite-time singularity to infinite mapped time.
- **Fourier pseudo-spectral methods:** Spectral differentiation using the Fast Fourier Transform (FFT), combined with fourth-order Runge–Kutta (RK4) time integration.
- **Dealiasing and convergence:** Investigation of numerical errors at different spatial resolutions and comparison with published results.

## Key Findings

- The first-order upwind scheme captures the general solution behaviour but loses accuracy as shock formation approaches.
- Time mapping enables the solution to be studied closer to the singularity, although the low-order numerical method produces significant errors.
- The Fourier pseudo-spectral method substantially improves numerical accuracy and delays the divergence of errors in the mapped system.
- Increasing spatial resolution improves the results, while the experiments indicate that the effect of dealiasing depends on the regime being studied.

## Report

The full undergraduate project report is available in [`MT4599_project.pdf`](MT4599_project.pdf).

## References

The project builds on established work on Burgers' equation, spectral methods and time-mapped formulations, including:

- Burgers, J. M. (1948). *A mathematical model illustrating the theory of turbulence*.
- Bustamante, M. D. (2011). *3D Euler equations and ideal MHD mapped to regular systems: Probing the finite-time blowup hypothesis*.
- Trefethen, L. N. (2000). *Spectral Methods in MATLAB*.

See the full report for the complete bibliography.
