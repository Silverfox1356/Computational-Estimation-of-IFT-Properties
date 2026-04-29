## Methodology & Quality Checks

### Step 1: Calibration (`attempt_auto_calibration`)
**Objective**: Determine the spatial resolution (meters per pixel) by detecting the physical needle's width in the image.
**Methodology**:
1. Computes the initial average width of the top 10 rows of the detected edges to get a rough guess.
2. Uses a robust statistical approach (`robust_stable_needle_width`) to find the stable needle width over a sample length.
3. Calculates `pixel_to_m` by dividing the known `needle_tip_diameter_mm` by the measured pixel width.
**Thresholds & Parameters**:
- `sample_length_mm` = 2.0 mm (length of needle to sample)
- `skip_top_rows` = 5 (skip the very top of the image to avoid boundary artifacts)
- `min_sample_rows` = 10
- `sigma_clip` = 2.5 (removes outliers outside 2.5 standard deviations from the median width)

### Step 2: Baseline Detection (`attempt_auto_baseline`)
**Objective**: Identify the vertical position (y-coordinate) where the drop emerges from the needle (the baseline).
**Methodology**:
1. Scans the edge profile from top to bottom.
2. Finds the first row where the horizontal width exceeds the detected needle width by a specified percentage.
**Thresholds & Parameters**:
- `deviation_percent` = 5.0% (width must exceed needle width by 5%)
- `start_skip_rows` = 10 (ignores the first 10 valid rows to prevent premature triggering on noise)

### Step 3: IFT Analysis (`calculate_physics`)
**Objective**: Fit the experimental drop profile to the theoretical Young-Laplace equation to determine the Interfacial Tension (IFT) and drop volume.
**Methodology**:
1. **Filtering**: Applies a Savitzky-Golay filter to smooth the left and right edges.
2. **Initial Guesses**: 
   - Fits a circle to the bottom 12% of the drop to estimate apex coordinates and initial radius ($R_0$).
   - Computes an initial $\beta$ estimate using the empirical DS/DE method.
3. **Robust Optimization**: Performs a simultaneous least-squares fit optimizing 4 parameters: $x_{axis}$, $y_{apex}$, $\ln(R_0)$, and $\beta$. Uses a `soft_l1` loss function to minimize the impact of outliers.
4. **Uncertainty Propagation**: Propagates uncertainties from the fit covariance, needle diameter tolerance, and fluid densities to compute the final IFT error bounds.
5. **Worthington Number**: Computes the Worthington number ($W_o$) to ensure the drop is sufficiently deformed by gravity for a reliable IFT measurement.
**Thresholds & Quality Checks (QC)**:
- **Filtering**: `SAVGOL_WINDOW` = 15, `SAVGOL_POLY` = 3
- **Fit Trim**: `FIT_TRIM_TOP_PX` = 3 (trims top 3 pixels from the baseline to avoid neck artifacts)
- **QC 1 (Fit RMSE)**: Root Mean Square Error $\le 1.5$ px
- **QC 2 (Beta Range)**: $\beta \in [0.05, 1.50]$
- **QC 3 (Convergence)**: Optimizer status must be $> 0$
- **QC 4 (Pre-fit translation)**: Translation asymmetry $\le 1.0$ px
- **QC 5 (Pre-fit tilt)**: Drop tilt slope $\le 0.005$
- **QC 6 (Post-fit shape)**: Shape asymmetry $\le 1.0$ px
- **QC 7 (Worthington No.)**: $W_o \ge 0.30$

### Step 4: Domain Extraction (`contour_to_domain` & `build_domain_polygon`)
**Objective**: Construct a complete, closed 2-D cross-section polygon of the pendant drop for FEM. The domain includes the full liquid column inside the needle, the needle tip steps, and the free-surface drop contour on both sides.
**Methodology**:
1. **Coordinate Shift**: Shifts the vertical coordinates so $z=0$ is at the top of the liquid neck, placing the baseline at a calculated `neck_height`.
2. **Apex Reconstruction**: If the bottom of the drop does not naturally reach $r=0$ due to imaging limits, a parabolic fit is used to extrapolate the apex.
3. **Smoothing**: Linearly tapers the very bottom points to exactly $r=0$ to ensure a sharp, closed tip.
4. **Polygon Assembly**: Stitches together the right inner wall, right tip step, right free surface, left free surface (mirrored), left tip step, left inner wall, and top wall into a single counter-clockwise polygon.
5. **Deduplication**: Removes overlapping consecutive points to prevent meshing errors.
**Thresholds & Parameters**:
- `neck_height_factor` = 0.8 (Neck height is set to 80% of the visible drop height).
- `apex_recon_points` = 8 (Number of points generated during parabolic apex reconstruction).
- `apex_smooth_points` = 6 (Number of points near the apex linearly tapered to $r=0$).
- `needle_id_mm` = Assumed to be 70% of the Outer Diameter (OD) if not explicitly provided.

### Step 5: Mesh Generation (`generate_mesh`)
**Objective**: Generate a high-quality, unstructured 2-D triangular mesh of the pendant drop domain using the Gmsh API, ensuring smooth transitions and high resolution at critical boundaries.
**Methodology**:
1. **Preprocessing**: The domain polygon is cleaned, downsampled (to prevent over-constraining the mesh boundary), and smoothed.
2. **Curvature-based Sizing**: Estimates the local Menger curvature at each boundary point. Higher curvature regions are assigned smaller target mesh sizes.
3. **Background Fields**:
   - Uses a distance-based `Threshold` field to gradually increase element size from the boundary towards the center.
   - Places `Ball` fields at high-gradient regions: the drop apex and the needle neck junctions to force extra refinement.
4. **Meshing & Optimization**: 
   - Uses the Frontal-Delaunay algorithm for structured-like unstructured meshing.
   - Applies multiple passes of Laplacian smoothing and a final Netgen optimization step to maximize triangle quality.
**Thresholds & Parameters**:
- `downsample_target` = 250 (Maximum number of boundary points retained to avoid overly dense boundaries).
- `smooth_window` = 2 (Moving average window size for smoothing the free surface polygon).
- `size_min_frac` = 0.010 (Minimum element size as a fraction of the maximum domain dimension).
- `size_max_frac` = 0.045 (Maximum element size inside the bulk).
- `growth_rate` = 1.25 (Controls the element size transition from boundary to bulk).
- `optimise_steps` = 5 (Number of Laplacian optimization passes).
**Quality Checks (Metrics tracked)**:
- **Minimum Angle**: Ensures no skinny/sliver triangles (Target minimum angle $\ge 25^\circ$).
- **Maximum Aspect Ratio**: Ratio of the longest edge to the shortest altitude (Should ideally be $< 3$).
- **Area Ratio**: Ratio between the largest and smallest triangle areas (Target $< 100-300$).

### Step 6: Assembling FEM Matrix (`assemble_fem_system`)
**Objective**: Construct the Finite Element Method (FEM) matrices for axisymmetric diffusion across the pendant-drop domain, applying appropriate boundary conditions.
**Methodology**:
1. **Mass Matrix ($H$)**: Computes a lumped-consistent mass matrix. For axisymmetry, each elemental contribution is weighted by the radial centroid ($R_c$).
2. **Stiffness Matrix ($K$)**: Computes the stiffness matrix incorporating the diffusion coefficient $D$. Includes the standard diffusion term and an exact axisymmetric correction term.
3. **Boundary Classification**: Classifies mesh boundary edges geometrically into 'wall' (needle inner wall, tip step, top wall) and 'interface' (free surface).
4. **Boundary Terms ($K_b$, $F$)**: Applies a Robin Boundary Condition *only* on the interface edges. This introduces a mass-transfer coefficient ($k$) representing the flux $-D \frac{\partial c}{\partial n} = k (c - c_\infty)$.
**Thresholds & Parameters**:
- $D = 10^{-9}$ m²/s (Default diffusion coefficient for aqueous species).
- $k = 10^{-5}$ m/s (Default mass-transfer coefficient).
- Robin boundary condition assumes normalized bulk concentration $c_\infty = 1.0$.
**Quality Checks (Sanity Checks)**:
- **H Symmetry**: Strict symmetry of mass matrix (Frobenius error $< 10^{-12}$).
- **K Near-Symmetry**: Stiffness matrix near-symmetry (due to axisymmetric term). Error $< 10^{-10}$ is OK, $< 10^{-8}$ is a Warning.
- **Diagonal Positivity**: $H_{diag} \ge 0$ and $K_{diag} > 0$ to ensure well-posed diffusion.
- **Sparsity**: $H$ and $K$ sparsity must both be $< 1\%$.
- **F-Vector Consistency**: The load vector $F$ must be non-zero *only* at the geometrically defined interface nodes.

### Step 7: Running the Simulation (`solve_diffusion`)
**Objective**: Perform a forward time-stepping simulation of the diffusion process inside the drop to predict the interface concentration $C_s(t)$ and the resulting surface tension $\gamma(t)$ over time.
**Methodology**:
1. **Time Discretization**: Uses the unconditionally stable $\theta$-scheme (where $\theta = 0.5$ is Crank-Nicolson, $\theta = 1$ is fully implicit).
2. **System Setup**: Computes the constant left-hand side matrix $A = H/\Delta\tau + \theta K_{total}$ and dynamic right-hand side $B$.
3. **Time Stepping**: Solves the linear sparse system $A \cdot C^{n+1} = B \cdot C^n + F_{total}$ at each timestep.
4. **Stability Clip**: Clips unphysical concentration values to enforce $C \in [0, 1]$.
5. **Interface Tracking**: Extracts the mean concentration $C_s$ strictly at the interface nodes.
6. **Surface Tension Model**: Maps $C_s(t)$ to $\gamma(t)$ using an Equation of State (EoS).
**Thresholds & Parameters**:
- $\theta = 0.7$ (Default implicitness parameter, Crank-Nicolson-like but damped for stability).
- $\Delta\tau = 0.01$ (Dimensionless time step).
- `n_steps = 100` (Default number of simulation steps).
- **Equation of State**: Linear model parameters $\gamma_0 = 72.0$ mN/m (clean interface) and $\gamma_\infty = 35.0$ mN/m (fully saturated interface).
**Quality Checks (Diagnostics)**:
- **Concentration Bounds**: Tracks minimum and maximum concentrations at each step to ensure physical bounds.
- **Numerical Stability**: Checks for the presence of `NaN` values to detect numerical instability.
- **Dimensionless Biot Number**: Computes $k_D = (k \cdot r_n) / D$ as a physical sanity check for the balance of mass transfer and diffusion.