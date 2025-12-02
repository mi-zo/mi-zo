# MI-ZO
This repository contains code and project resources for **MI-ZO**, a method for multi-information estimation with zeroth-order optimisation to control an in-scene camera in 3D scenes.

MI-ZO enables off-the-shelf vision–language models (VLMs) trained on 2D visual inputs to process and reason over 3D scenes without resorting to pretraining or finetuning.

- **Paper:** *Video and Language Alignment in 2D Systems for 3D Multi-object Scenes with Multi-Information Derivative-Free Control*
- **Authors:** Jason Armitage, Rico Sennrich  
- **Conference:** IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)  
- **Project page:** https://mi-zo.github.io/mi-zo/  
- **Open-access preprint:** (arXiv link to be added)

---

## Overview

Cross-modal systems trained on 2D visual inputs encounter a dimensional shift when performing tasks with 3D reconstructed scenes. In MI-ZO, control of an in-scene camera bridges this gap. The problem is reduced to predicting a sequence of camera actions on the 3D scene that returns a correct response from the VLM with as few views as possible.

A multi-information metric over mixed continuous and discrete entropy sources is optimised with a zeroth-order algorithm to minimise redundancies or regret. The resulting multi-information with active regret minimisation MI\(_\text{ar}\) metrics guide a controller that selects high-information viewpoints on 3D multi-object scenes w.r.t. textual descriptions.

Key characteristics:

- **Inference-only:** no access to model parameters, no backpropagation through the VLM.
- **Low-data:** operates with a very small number of demonstrations and online feedback.
- **Multi-object 3D scenes:** designed for settings where models are required to reason over scenes with multiple similar objects.

---

## Method: MI-ZO in Brief
Our method is made up of two core components:

1. **Multi-information with active regret minimisation**

   We estimate multi-information over \( n > 2 \) sources combining:
   - global and object-level measures based on colour spaces (e.g. LAB, HSV),
   - local visual features,
   - linguistic features derived from textual descriptions.

   The resulting metrics (e.g. GO-LED-OL\(_\text{ar}\), GH-LED\(_\text{ar}\)) express the information capacity of the multimodal inputs and maximise the distance between correct and incorrect VLM decisions.

2. **Efficient controller for the in-scene camera**

   A controller based on polynomial regression and least-squares approximation uses our multi-information measures to predict camera actions. Evaluation is structured in two stages:
   - a **measurement round** with a default sequence of actions, and
   - a **correction round** where predicted actions prioritise informative viewpoints.

---

## Benchmarks and Data

This repository will host sample code and inputs for the cross-modal benchmarks introduced in the paper:

- **GeoProperties-3DS**  
  Scenes with group of rocks and other geological features are extracted from large 3D reconstructions of locations on Mars. Tests are desgined to demonstrate the contribution of control methods in reducing error rates of VLMs when performing analysis in planetary science.

- **FeatureID-3DS**  
  Virtual scenes populated with multiple objects where the description refers to a feature that is only visible from selected viewpoints. This evaluation focuses on the improvements provided by control methods when assessing features of objects in 3D scenes.

- **PartialView-3DS**  
  Multi-object scenes separated by a partition so that one object is fully or partially occluded. This benchmark is designed to measure how well control methods can assist VLMs trained on 2D data to handle occlusions.

Links to dataset downloads, generation scripts, and documentation will be added here.

---

## Repository Structure (planned)

The current structure is minimal and will be extended along the following lines:

- `docs/` – project page and documentation (GitHub Pages source).
  - `index.md` – main project page for MI-ZO.
  - `assets/` – CSS and images for figures and samples.
- `mi_zo/` – implementation of MI-ZO metrics and the zeroth-order controller.
- `configs/` – experiment configurations (VLMs, benchmarks, camera settings).
- `scripts/` – utilities for dataset preparation, rendering, and evaluation.
- `experiments/` – example runs, logs, and analysis notebooks.

---

## Citation

If you use MI-ZO, the diagnostic, or the benchmarks in your work, please cite:

```bibtex
@inproceedings{armitage_mizo,
  title     = {Video and Language Alignment in 2D Systems for 3D Multi-object Scenes with Multi-Information Derivative-Free Control},
  author    = {Armitage, Jason and Sennrich, Rico},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2026}
}
