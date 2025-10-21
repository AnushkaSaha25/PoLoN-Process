# PoLoN-Process
Poisson Log-Normal (PoLoN) Process for Non-Parametric Prediction of Count Data
# Poisson Log-Normal (PoLoN) Process for Non-Parametric Prediction of Count Data

### Overview
Modeling datasets of integer counts is crucial in physics and other scientific disciplines, where measurements involve discrete, non-negative quantities.  
Traditional approaches such as Poisson regression often struggle to capture complex, non-linear relationships.  
The **Poisson Log-Normal (PoLoN) process** combines the flexibility of Gaussian Processes with the Poisson-lognormal distribution to accurately model integer count data.

This repository contains the implementation, synthetic data experiments, and real-world applications (e.g., Higgs Boson signal analysis and bike rental prediction) for the PoLoN framework.

---

### Key Features
- Non-parametric model for count data prediction  
- Combines Poisson likelihood with Gaussian Process prior  
- Handles discrete, non-negative data  
- Captures correlations through kernel-based inference  
- Supports both synthetic and real-world datasets  

---

### Repository Structure
### 🧩 Examples

You can reproduce the examples used in the paper by running:

```bash
python generate_synthetic_data.py


