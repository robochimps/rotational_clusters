rovib_me_pmax24
----
This folder contains files with rovibrational energies and matrix elements of various operators, such as electric dipole, spin-rotation, etc., calculated using vibrational basis set truncated at the polyad number $P_\text{max}=24$.

File naming conventions
---
* Files **h2s_enr_1000...** and **h2s_me_1000...** contain energies and matrix elements for the first 1000 states per each $J$ and symmetry.

* Files **h2s_enr_cluster...** and **h2s_me_cluster...** contain energies and matrix elements only for rotational cluster states in the ground vibrational state. The corresponding cluster states are listed in file [h2s_cluster_states_id_pmax24.txt](../h2s_cluster_states_id_pmax24.txt)

File generation
---
These files are generated using the following codes: [h2s_cart_me.py](../h2s_cart_me.py) (or [h2s_cart_me.ipynb](../h2s_cart_me.ipynb) notebook).

How to read these files
---
For guidance on opening and reading these files, refer to the [h2s_hyperfine.ipynb](../h2s_hyperfine.ipynb) notebook.