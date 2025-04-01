# Distribution System State Estimation (DSSE) using JAX

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [System Requirements](#system-requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Citation](#citation)

---

## Project Overview <a name="project-overview"></a>
This repository implements a novel DSSE approach that combines:
- OpenDSS for power flow solving
- JAX for accelerated gradient descent
- Smart meter data via py-dss-interface

### Key Innovations:
- Eliminates explicit Jacobian computation through JAX's automatic differentiation
- Achieves 40-60% faster computation vs traditional methods
- Improves accuracy in unbalanced systems
- Enables GPU-accelerated performance

[Read the full paper](https://doi.org/10.1109/TPEC63981.2025.10906977)

---

## Key Features <a name="key-features"></a>
- ⚡ **GPU-accelerated state estimation**
- ⚖️ **Weighted Least Squares (WLS) optimization**
- 📈 **Automated visualization**
- 🔌 **Supports IEEE 13/37/123 bus systems**
- ⚙️ **OpenDSS integration for real-world modeling**

---

## System Requirements <a name="system-requirements"></a>
- Python 3.9-3.11
- Windows 10/11 (for OpenDSS compatibility)
- NVIDIA GPU (recommended) with:
  - CUDA 12.1+
  - cuDNN 8.9+
- 8GB+ RAM

---

## Installation <a name="installation"></a>

### 1. Create Virtual Environment
```bash
conda create -n dss_env python=3.11
conda activate dss_env
pip install -r requirement.txt
```

---

## Usage <a name="usage"></a>

### 1. Run State Estimation  
```bash
python JAX_Estimate.py --case=13  # Options: 13, 37, 123
```

### 2. Visualize Results  
```bash
python plot_results.py
```

---

## Citation <a name="citation"></a>

If you use this work in your research, please cite:

```bibtex
@INPROCEEDINGS{10906977,
  author={Dutta, Rochisnu and Alla, Ravi Teja and Matavalam, Amarsagar Reddy Ramapuram and Swarup, K. Shanti},
  booktitle={2025 IEEE Texas Power and Energy Conference (TPEC)},
  title={Distribution System State Estimation Using Automatic Differentiation with JAX},
  year={2025},
  volume={},
  number={},
  pages={1-6},
  keywords={Jacobian matrices;Accuracy;Machine learning;Linear programming;Smart meters;Computational efficiency;State estimation;Load flow;State estimation;WLS;distribution system;JAX;openDSS;automatic differentiation;gradient descent},
  doi={10.1109/TPEC63981.2025.10906977}
}
```

---

## Contributing

Contributions are welcome! If you find a bug or have a feature request, please open an issue or submit a pull request.

