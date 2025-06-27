# cellpose_plus

---

[![PyPi](https://badge.fury.io/py/cellpose_plus.svg)](https://badge.fury.io/py/cellpose_plus)
![License](https://img.shields.io/github/license/ITMO-MMRM-lab/cellpose_plus?style=flat&logo=opensourceinitiative&logoColor=white&color=blue)
[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

---

## Overview

Cellpose Plus is a comprehensive tool designed for the morphological analysis of stained cell images, enabling users to extract critical features related to cell structure and organization. Its seamless workflow allows for efficient processing from raw images to insightful metrics, facilitating advanced study in cell biology and research.

---

## Table of Contents

- [Core features](#core-features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Examples](#examples)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---
## Core features

1. **Morphological Analysis Tool**: Cellpose Plus is primarily designed for morphological analysis of stained cell images, enabling users to extract detailed features related to cell structure and organization.
2. **Image Segmentation**: Utilizes advanced segmentation techniques to accurately differentiate between cells and nuclei in various cell images, leveraging a forking of the Cellpose framework for enhanced capabilities.
3. **Feature Extraction Algorithms**: Includes multiple algorithms to extract specific morphological features such as area, roundness, and center coordinates of cells and nuclei after segmentation.
4. **Single Workflow Processing**: Provides a comprehensive workflow from raw image input to output metrics, enabling users to efficiently manage and process stained cell images without switching tools.

---

## Installation

Install cellpose_plus using one of the following methods:

**Using PyPi:**

```sh
pip install cellpose_plus
```

---

To install the package using `conda`, follow these steps:
1. Open an `anaconda` prompt or command prompt.
2. Create a new environment for CPU only:
```bash
   conda create -n cellpose_plus 'python==3.9' pytorch
```
3. Activate the new environment:
```bash
   conda activate cellpose_plus
```
4. For NVIDIA GPUs, run:
```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
5. Finally, install Cellpose Plus:
```bash
   pip install cellpose-plus[gui]
```
## Getting Started

To get started with Cellpose Plus, you need to run it for segmentation tasks. Here are the steps to use it:

### Running Cellpose Plus
To run Cellpose Plus using the GUI, follow these steps:
1. Launch the command line terminal/Anaconda Prompt.
2. Activate the environment you created:
```bash
   conda activate cellpose_plus
```
3. Launch the GUI:
```bash
   python -m cellpose
```
4. Drag and drop your desired image for segmentation.

### Important Note
Make sure to set a pixel-to-micrometer (μm) conversion value (μm per pixel) in the GUI to calculate the areas correctly.

### Example Usage
![demo_gif](https://raw.githubusercontent.com/ITMO-MMRM-lab/cellpose/refs/heads/main/repo/demo_cellpose_plus.gif)
After the segmentation process, save the masks in a folder with the same name as the image and analyze results in CSV format.

You can also experiment with the feature extraction metrics in the Cellpose plus workflow by referring to the online example provided via Google Colab:
[Open In Colab](https://colab.research.google.com/drive/1_yDbBQb0Ndc4QcTvONOUbfVziwB6Ykev?authuser=1#scrollTo=imGtXZPMu_al).

---

## Examples

Examples of how this should work and how it should be used are available [here](https://github.com/ITMO-MMRM-lab/cellpose_plus/tree/main/docs/notebook.rst).

---

## Documentation

A detailed cellpose_plus description is available [here](https://github.com/ITMO-MMRM-lab/cellpose_plus/tree/main/docs).

---

## Contributing

- **[Report Issues](https://github.com/ITMO-MMRM-lab/cellpose_plus/issues)**: Submit bugs found or log feature requests for the project.

---

## License

This project is protected under the BSD 3-Clause "New" or "Revised" License. For more details, refer to the [LICENSE](https://github.com/ITMO-MMRM-lab/cellpose_plus/tree/main/LICENSE) file.

---

## Citation

If you use this software, please cite it as below.

### APA format:

    ITMO-MMRM-lab (2023). cellpose_plus repository [Computer software]. https://github.com/ITMO-MMRM-lab/cellpose_plus

### BibTeX format:

    @misc{cellpose_plus,

        author = {ITMO-MMRM-lab},

        title = {cellpose_plus repository},

        year = {2023},

        publisher = {github.com},

        journal = {github.com repository},

        howpublished = {\url{https://github.com/ITMO-MMRM-lab/cellpose_plus.git}},

        url = {https://github.com/ITMO-MMRM-lab/cellpose_plus.git}

    }

---
