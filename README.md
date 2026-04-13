# AeroMHD-CUDA
I built the original version of this solver ealrier , but 3D fluid grids are too much for CPUs to handle., so I refactored the entire engine to run on the GPU. It’s a modular 3D Magnetohydrodynamics (MHD) solver optimized for high Mach RMI simulations. C++ CUDA kernels via CuPy . It includes a full V&amp;V suite, Brio-Wu, Alfvén convergence.
Copyright (c) 2026 Abdelrahman Shaltout. All Rights Reserved.
No part of this software may be used, reproduced, or distributed in any form or by any means without the prior written permission of the copyright owner.

## MHD-CUDA 

This is a 3D Magnetohydrodynamics package built to study the Richtmyer-Meshkov Instability RMI. I originally developed a CPU-based version , but 3D fluid grids quickly outgrow standard processors. This version offloads the heavy lifting to the GPU using CuPy and CUDA kernels I had structured.

I built and tuned this on a Quadro P1000 .

#Archcitecture
config.py: Centralized physical constants and grid parameters.
physics.py: Contains the C++ CUDA source for the HLLD Riemann solver, MUSCL reconstruction, and Rankine-Hugoniot math.
solver.py: Handles the time stepping logic (SSP-RK3), the 3D grid management, and diagnostics.
main.py: The entry point for verification suites and production runs.

# Performance & Physics

The goal was high resolution tracking of plasma interfaces under extreme conditions (Mach 10+).

HLLD Riemann Solver: Chosen specifically for its ability to resolve contact discontinuities and intermediate waves better than basic HLL.

Divergence Control: Uses a hybrid approach with GLM and Powell source terms to keep the ∇⋅B error from blowing up.

NumPy 2.0 : Patched to handle the recent np.trapz removal/deprecation. It runs on Python 3.10 through 3.12.

# V&V

Solver includes an automated verification suite:

Brio-Wu Shock Tube: Validates the solver against five distinct wave structures.

Alfven Wave Convergence: Measured at a 1.96 convergence order,essentially perfect 2nd-order accuracy.

Contact Discontinuity Test: Verified zero-pressure-blip preservation across density jumps.

Current Research: The "Magnetic Sieve"

The code is currently set up to explore a novel "Magnetic Sieve" experiment. Instead of a uniform background field, I’m testing spatially modulated magnetic fields. The idea is to see if we can "sculpt" RMI into organized plasma micro-jets rather than letting it collapse into chaotic turbulence.

# Quick Start

Assuming you have an NVIDIA GPU and a working CUDA toolkit:

pip install cupy-cuda12x nvidia-cuda-nvrtc-cu12  # match your CUDA version
python main.py
