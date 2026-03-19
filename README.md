# Neural ODE Modelling of Drone Ground Effect

[![Julia](https://img.shields.io/badge/Language-Julia-9558b2.svg)](https://julialang.org/)
[![MATLAB](https://img.shields.io/badge/Data-MATLAB-ed8b00.svg)](https://www.mathworks.com/products/matlab.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the full implementation of the **EE5311: Differentiable and Probabilistic Computing** course project by Group 15. We utilize the **Universal Differential Equations (UDE)** framework to model complex non-linear **Ground Effects** encountered by drones during vertical landing phases.

## 🚀 Project Highlights

* **Grey-box Modelling Architecture**: We integrated known physical laws (gravity, thrust, drag) with a neural residual term $\phi_{\theta}$ to create a structured differential equation.
* **Physics-Inspired Gating**: Implementation of a Sigmoid gating mechanism ($h_c = 0.5$m) ensures the neural components only activate in the near-ground regime, preserving global physical consistency.
* **Kinematic Fidelity**: By analyzing **Phase Portraits** ($v$ vs $h$), we successfully identified and mitigated the "unphysical curls" typically found in pure black-box models, ensuring our model respects the $\dot{h} = -v$ constraint.
* **Significant Precision Gains**: Our Grey-box UDE achieved a Global Altitude RMSE of **~0.0178m**, representing a **77% error reduction** in the critical near-ground regime compared to the white-box baseline.

## 📊 Core Results

| Model Paradigm | Global RMSE (m) | Near-ground RMSE ($h < 0.5$m) | Improvement (%) |
| :--- | :--- | :--- | :--- |
| White-box (Fixed Physics) | 0.0682 | 0.1014 | Baseline |
| Pure Black-box (MLP) | 0.0621 | 0.0612 | 9.2% |
| **Grey-box UDE (Ours)** | **0.0178** | **0.0225** | **73.9%** |

## 📂 Repository Structure

* **/data_generation**: Contains `generate_landing_data.m` for synthesizing noisy drone trajectory datasets in MATLAB.
* **/src**: Core Julia training script `train_ude.jl` using `Lux.jl` and `SciMLSensitivity.jl`.
* **/analysis**: Diagnostic script `plot_results.jl` for generating the 10-panel performance dashboard (Energy decay, Phase portraits, Heatmaps).
* **EE5311_CA1_Group_15.pdf**: The full technical report and project blog.

## 🛠️ Usage

### 1. Generate Data
Run `generate_landing_data.m` in MATLAB to populate the `/data` folder.

### 2. Setup Environment
Ensure Julia 1.9+ is installed. Install dependencies:
```julia
using Pkg; Pkg.add(["DifferentialEquations", "Lux", "SciMLSensitivity", "Zygote", "OptimizationOptimisers", "ComponentArrays", "Interpolations"])
```

### 3. Train Models
To begin the training process for the Black-box and Grey-box models, execute the following command in your terminal:
```bash
julia src/train_ude.jl
```

### 4. Visualize Diagnostics
After training, run the plotting script to generate the final 10-panel analytical dashboard:

```bash
julia analysis/plot_results.jl
```

## Authors (Group 15)

Wang Shiming: Formulated physical equations, optimized simulation data, and authored core report sections.

Tang Shaorou: Conducted theoretical white-box analysis and implemented MATLAB dynamics simulations.

Li Yukun (Leader): Developed Black-box and Grey-box source code, designed ablation studies, and tuned loss function weights.

Wu Junke: Selected optimization algorithms and conducted quantitative performance evaluations.

Xu Zixuan: Handled data post-processing, generated visualization figures, and analyzed force field heatmaps.

If you find this project helpful for your SciML research, please give this repository a⭐!
