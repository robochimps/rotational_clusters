# Rotational_clusters
### Nuclear spin-symmetry breaking in rotational cluster states

This repository contains supplementary code for the manuscript:
> Andrey Yachmenev and Guang Yang, "Nuclear spin symmetry breaking and spin polarization in rotational energy level clusters", *Phys. Rev. Research* **7** (2025) L032047, [DOI:10.1103/k7nd-nnxg](https://doi.org/10.1103/k7nd-nnxg), [arXiv:2503.20695 [physics]](https://doi.org/10.48550/arXiv.2503.20695)

The package performs variational calculations of the rovibrational energies of the triatomic molecule $\text{H}_2\text{S}$ up to high rotational excitations ($J \leq 60$).
At high $J$-values, some rotational states of $\text{H}_2\text{S}$ cluster into groups of four, an effect associated with spontaneous symmetry breaking caused by centrifugal and Coriolis couplings.
Additionally, the package performs computations of nuclear hyperfine effects (e.g., spin-rotation interaction) and the Stark effect.

Repository structure
---
- **[h2s_rovib.ipynb](h2s_rovib.ipynb)**: Computes the rovibrational energies and wavefunctions of $\text{H}_2\text{S}$, storing the results in files within the `rovib_wf_pmax24` folder.

- **[h2s_clusters.ipynb](h2s_clusters.ipynb)**: Identifies cluster states in both the ground and excited vibrational states by analyzing rotational density distributions. The resulting cluster state indices for different $J$ values and symmetries are stored in files, such as [h2s_cluster_states_id_pmax24.txt](h2s_cluster_states_id_pmax24.txt).

- **[h2s_cart_me.py](h2s_cart_me.py)**: Computes rovibrational matrix elements for various operators, including dipole moment and spin-rotation coupling tensors, for selected rovibrational states. The results are stored in files within the `rovib_me_pmax24` folder.

- **[h2s_hyperfine.ipynb](h2s_hyperfine.ipynb)**: Computes hyperfine spin-rotation interactions, plots rotational and spin-density distributions, and evaluates the Stark effect for rotational cluster states.

Citation
---
If you use this code in your research, please cite:

```bibtex
@article{Yachmenev_PRR7_2025,
  author  = {A. Yachmenev and G. Yang},
  title   = {Nuclear spin symmetry breaking and spin polarization in rotational energy level clusters},
  year    = {2025},
  journal = {Phys. Rev. Research},
  volume = {7},
  pages = {L032047},
  url = {https://doi.org/10.1103/k7nd-nnxg},
  DOI = {10.1103/k7nd-nnxg},
  archiveprefix = {arXiv},
  arxivid = {2503.20695},
  eprint = {2503.20695},
  primaryclass = {physics},
  arxiv = {https://arxiv.org/abs/2503.20695},
}
```

Contact
---
For questions or feedback, feel free to open an issue or reach out to the authors directly via andrey.yachmenev@robochimps.com
