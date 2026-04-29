```mermaid
flowchart TD

A["Step 1: Curve Extraction (CV)
Input: Image
Output: Drop contour + r(z) data"]

B["Step 2: Domain Construction
Input: r(z) curve
Output: Closed polygon (drop domain)"]

C["Step 3: Polygon Processing
Input: Raw polygon
Output: Cleaned + smoothed polygon"]

D["Step 4: Mesh Generation (Gmsh)
Input: Polygon
Output: Nodes + Triangles + Boundary lines"]

E["Step 5: FEM Matrix Assembly
Input: Mesh (nodes, elements)
Output: H, K, K_bnd, F matrices"]

F["Step 6: Time Integration (θ-scheme)
Input: FEM matrices + initial condition
Output: Concentration field C(r,z,t)"]

G["Step 7: Interface Extraction
Input: Concentration field
Output: Interface concentration C_int(t)"]

H["Step 8: IFT Prediction
Input: C_int + calibration curve
Output: Predicted IFT γ(t)"]

I["Step 9: Error Calculation
Input: Predicted IFT + Experimental IFT
Output: Error E(D, k_D)"]

J["Step 10: Parameter Optimization
Input: Error function
Output: Optimal D and k_D"]

A --> B --> C --> D --> E --> F --> G --> H --> I --> J