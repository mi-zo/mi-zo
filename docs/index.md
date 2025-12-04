---
layout: default
title: MI-ZO
---

# MI-ZO

**Video and Language Alignment in 2D Systems for 3D Multi-object Scenes with Multi-Information Derivative-Free Control**  
Jason Armitage, Rico Sennrich  
University of Zurich, Switzerland  

---

## Overview

Cross-modal systems trained on 2D visual inputs are presented with a dimensional shift when processing 3D scenes. An in-scene camera bridges the dimensionality gap but requires learning a control module. In MI-ZO, the problem is reduced to predicting a sequence of camera actions on the 3D scene that returns a correct response from the vision–language model with the least number of views.

We propose a straightforward approach to predicting a VLM’s errors and to tuning an in-scene controller guided by a measure that quantifies both the information presented in the scene and model uncertainty. Our multi-information metric combines information from several variables while limiting redundancies – or regret – to a minimum, and is optimised with a zeroth-order algorithm.

MI-ZO enables off-the-shelf vision–language models trained on 2D data to adapt online to object occlusions and differentiate features, improving performance on cross-modal tasks in 3D multi-object scenes without resorting to pretraining or finetuning.

---

## MI-ZO in One Figure

![Textures of objects in 3D scenes vary depending on the camera position.](assets/img/mi-zo-fig1-camera-views.png)

<span class="fig-caption">
Textures of objects in 3D scenes vary in appearance depending on the position of the in-scene camera. Aligning a description with a scene is harder when referenced objects belong to a group such as a single boulder in an outcrop on Mars. An optimal sequence of viewpoints improves the decisions of VLMs trained on 2D data where understanding a 3D reconstructed scene relies on a set of views.
</span>

---

## Method

### Multi-information estimation with active regret minimisation

We opt for a form of mutual information estimation over multiple variables as the basis for our metric. Our multi-information metric overcomes theoretical challenges of estimating multivariate mutual information over \( n > 2 \) mixed discrete and continuous variables using active regret minimisation and zeroth-order optimisation.

The method considers multiple entropy sources derived from the current viewpoint and description, including:

- global visual variables from perceptual colour spaces (e.g. CIELAB, HSV),
- object-level measurements from segmented object masks,
- local edge density estimates for object boundaries,
- simple linguistic counts for noun phrases and descriptors.

Sources are combined in a weighted mixture distribution. A zeroth-order optimisation step updates mixing weights using only VLM feedback (correct / incorrect labels) so that overlaps between sources are reduced and the distance between correct and incorrect decisions in the score distributions is increased. Variants such as GO-LED-OL\(_{\text{ar}}\) and GH-LED\(_{\text{ar}}\) serve as expressive multi-information measures over more than two variables.

### Controller for an in-scene camera

We formulate the challenge of promoting accurate and fast reasoning by VLMs on 3D scenes as predicting a sequence of camera actions on the scene that returns a correct response with the least number of views.

Evaluation is organised in two rounds:

1. **Measurement round**  
   A default sequence of camera actions explores the scene. For each viewpoint, MI-ZO computes multi-information scores and records VLM decisions on scene-level questions.

2. **Correction round**  
   A controller based on polynomial regression, least-squares approximation, and an interaction matrix predicts a new sequence of camera actions. The controller prioritises viewpoints with high information content and low expected regret, using feedback from demonstrations and the measurement round.

The controller is tailored to low-data settings and runs entirely at inference time, without access to VLM parameters or backpropagation.

---

## Diagnostic: UC-3DS-MI

![Samples of uniform and complex scenes from the UC-3DS-MI dataset.](assets/img/uc-3ds-mi-samples.png)

<span class="fig-caption">
UC-3DS-MI is a diagnostic set of 3D scenes with uniform and complex polygon objects viewed from multiple viewpoints. The dataset is designed to illustrate how minimising regret leads to accurate expression of information in both simple and complex scenes.
</span>

UC-3DS-MI consists of uniform and complex scenes with abstract polygon objects placed in different positions on a floor mesh. Language descriptions and scenes are split between concentration on colour or geometry. The diagnostic is used to demonstrate when multivariate mutual information estimates become nonpositive due to redundant sources, and how active regret minimisation restores additive behaviour of the metric over \( n > 2 \) variables.

---

## Benchmarks on 3D Multi-Object Scenes

We introduce three cross-modal benchmarks with 3D multi-object scenes to evaluate control methods for in-scene cameras.

### GeoProperties-3DS: reasoning over geological properties

![Samples from GeoProperties-3DS.](assets/img/geoproperties-samples.png)

<span class="fig-caption">
Samples from the GeoProperties-3DS benchmark with close-ups of properties correctly identifying a single object in the scene. Descriptions refer to the largest rock or boulder and match scenes on the right.
</span>

GeoProperties-3DS consists of sets of scenes extracted from 3D mesh models of rocks, regolith, and other geological features generated from observations from Mars rover missions. Each collection contains five scenes and a single textual summary where only one scene matches the description. The descriptions refer to physical properties of formations and outcrops that are visible from a subset of viewpoints.

The benchmark is designed to assess methods that reduce the likelihood of false positives by 2D VLMs when reasoning over visual properties for planetary science. Our controller with GO-LED-OL\(_{\text{ar}}\) and GH-LED\(_{\text{ar}}\) reduces balanced error rate compared to both VLM-only baselines and standard control algorithms.

### FeatureID-3DS: feature identification under short runs

![Samples from the FeatureID-3DS dataset.](assets/img/featureid-samples.png)

<span class="fig-caption">
Samples from the FeatureID-3DS dataset. Scenes contain multiple objects where the description refers to a feature that is only visible from selected viewpoints.
</span>

FeatureID-3DS focuses on feature identification in virtual 3D environments. Scenes are composed from object models and a floor mesh, with human-generated descriptions that refer to features visible only from specific viewpoints. A VLM provides a boolean response on world features given viewpoint–description pairs.

To assess prioritisation of high-information viewpoints, camera action budgets in the correction round are deliberately small. Our controller with MI\(_{\text{ar}}\) metrics improves accuracy under these restricted action counts and prioritises viewpoints that expose features with strong visual prominence, while wall-clock time increases roughly linearly with the number of camera actions.

### PartialView-3DS: adaptation to occlusions

![Samples from the PartialView-3DS dataset.](assets/img/partialview-samples.png)

<span class="fig-caption">
Samples from the PartialView-3DS dataset. Scenes contain two objects separated by a partition so that one object is partially or fully occluded in every view.
</span>

PartialView-3DS consists of scenes with two objects located on opposite sides of a partition. At each viewpoint, one object is partially or fully occluded. A set of candidate descriptions is presented, and at the end of a sequence of views the system must select the summary matching the full scene.

The benchmark is designed to test adaptation of control methods under persistent occlusion. Our Poly+ZO+MI controller with MI\(_{\text{ar}}\) metrics adapts camera actions to return views that improve VLM performance and outperforms standard control algorithms and neural controllers optimised with stochastic gradient descent.

---

## Results (Qualitative Summary)

For our three benchmarks MI-ZO:

- Separates correct and incorrect VLM decisions into distinct regions of the multi-information score distributions
- Benefits from additional feedback on correctness labels when active regret minimisation is applied
- Improves accuracy and reduces balanced error rate compared to VLM-only baselines and classical control methods.

Qualitative analysis shows that in the correction round the controller tends to replace low-information views with viewpoints that expose critical features or reduce ambiguity between similar objects.

---

## Resources

- **Code outline (outline code to assist understanding of our proposed methods):** https://github.com/mi-zo/mi-zo/tree/main/code  
- **Benchmark samples:** single samples for GeoProperties-3DS, FeatureID-3DS, and PartialView-3DS

For questions about this project please contact [jason.armitage@uzh.ch](mailto:jason.armitage@uzh.ch)


