# Random Matrices Study

> Numerical Linear Algebra – **Assignment 4** (2025)  
> [![Build Status](https://img.shields.io/badge/status-cool-lightgrey)](#)

This project numerically explores high-dimensional phenomena in **Gaussian random matrices**:

* **Norm concentration** — how ‖ · ‖₂ of *N*(0, 1) vectors collapses around √m.  
* **Inner-product behaviour** — empirical convergence to *N*(0, 1) after normalisation.  
* **Worst-case coherence** — (Gumbel) law for the maximum column correlation.

All derivations and detailed discussion live in **[`docs/report.pdf`](./docs/report.pdf)**
---

## 📋 Table of Contents
1. [Project Overview](#project-overview)  
2. [Repository Structure](#repository-structure)  
3. [Quick Start](#quick-start)  
4. [Running the Analyses](#running-the-analyses)  
5. [Testing](#testing)  
6. [License](#license)

---

## Project Overview
The workflow is intentionally simple:

1. **Generate** an *m × n* Gaussian matrix (or many of them in parallel).  
2. **Sample** the statistic of interest (norms, inner products, or maximum correlation).  
3. **Plot** histograms against the corresponding theoretical density.  

Python scripts under `src/` handle steps 1–2; Matplotlib/Seaborn manage step 3
---

## Repository Structure

```bash
├── docs/
│   ├── images/            #auto-generated plots land here
│   ├── bibliography.bib   #references for the report
│   ├── report.pdf         #final write-up
│   └── report.typ         #Typst source
├── src/
│   └── assignment.ipynb   #step-by-step exploration
├── written_assignment.pdf #original problem statement
├── requirements.txt    
├── LICENSE
└── README.md  
````

---

## Quick Start
```bash
# 1) clone and create an isolated env (Python ≥ 3.11)
git clone https://github.com/arthurabello/nla-assignment-4.git
cd nla-assignment-4
python -m venv .venv && source .venv/bin/activate

# 2) install requirements
pip install -r requirements.txt
````

---

## Running the Script

Go to the [Jupyter Notebook]()

## License

Distributed under the MIT License – see [`LICENSE`](./LICENSE) for details.

---

### Authors

* **[Arthur Rabello Oliveira](https://github.com/arthurabello)**
* **[Henrique Coelho Beltrão](https://github.com/riqueu)**

