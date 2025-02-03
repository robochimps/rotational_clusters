rovib_wf_pmax20
----
This folder contains files with rovibrational energies, wavefunction coefficients, and vibrational matrix elements of various operators, calculated using vibrational basis set truncated at the polyad number $P_\text{max}=20$.

File naming conventions
---
* **h2s_coefficients_pmax20_j`j`.h5** is an HDF5 file containing energies, wavefunction coefficients, and quantum numbers of rovibrational states corresponding to $J$=`j`.

* **h2s_energies_pmax20_jmax60.txt.gz** is an ASCII file with rovibrational energies and assignments of all rovibrational states for $J=0..60$.

* **h2s_vibme_pmax20.h5** is an HDF5 file containing vibrational matrix elements of various operators, such as dipole moment and spin-rotation tensor.

File generation
---
These files are generated using the following code: [h2s_rovib.ipynb](../h2s_rovib.ipynb)

How to read these files
---
For guidance on opening and reading these files, refer to the [h2s_rovib.ipynb](../h2s_rovib.ipynb) notebook.