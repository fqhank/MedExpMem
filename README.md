# MedExpMem: Adapting Experience Memory for Differential Diagnosis

> **MICCAI 2026 Early Accept**

[![arXiv](https://img.shields.io/badge/arXiv-2605.22872-b31b1b.svg)](https://arxiv.org/abs/2605.22872)
![MICCAI](https://img.shields.io/badge/MICCAI-2026-blue)
![Status](https://img.shields.io/badge/Release-Preview-orange)

## 🚧 Preview Version

> **This repository is currently a preview release of MedExpMem.**

We are actively organizing the code and documentation for the full open-source release.  
Some implementations, configurations, and interfaces may therefore be updated in future versions.

---

## 🧠 Introduction

**MedExpMem** is an experience-memory framework that enables medical vision-language models to learn reusable differential-diagnosis experience from their own diagnostic errors, without updating model parameters.

---

## 💡 Motivation

Clinical expertise is more than static medical knowledge.

Experienced clinicians gradually learn:

- which diseases are easily confused,
- which subtle findings distinguish them,
- and which reasoning patterns tend to lead to diagnostic errors.

In contrast, existing medical VLMs typically handle each case independently and do not naturally accumulate such diagnostic experience.

MedExpMem asks:

> **Can medical AI learn from its previous diagnostic failures and reuse that experience when encountering future cases?**

---

## 🔍 Method Overview

MedExpMem builds and utilizes model-specific diagnostic experience through three main stages:

### 1. Blind-Spot Discovery

The model performs diagnosis without access to memory. Incorrect predictions reveal its intrinsic diagnostic blind spots.

### 2. Experience Refinement

Diagnostic failures are transformed into structured **pairwise differential experience**, capturing:

- confusion points,
- key discriminators,
- decision rules,
- and previous reasoning errors.

The experience is further refined through reflective re-diagnosis.

### 3. Experience-Augmented Diagnosis

For a new case, relevant experience is retrieved from memory and incorporated into the reasoning process to help the model avoid previously observed diagnostic mistakes.

```text
Diagnostic Failure
       ↓
Blind-Spot Discovery
       ↓
Pairwise Differential Experience
       ↓
Reflective Refinement
       ↓
Experience Memory
       ↓
Retrieval for New Cases
       ↓
Experience-Augmented Diagnosis
```

---

## 📊 Main Results

MedExpMem was evaluated across multiple vision-language models on a radiology benchmark covering **11 subspecialties**.

It consistently improves diagnostic performance across different model families and scales, achieving improvements of **up to +7.0 percentage points** in diagnostic accuracy.

These results demonstrate that **model-specific diagnostic experience can effectively complement static parametric knowledge and conventional external knowledge retrieval**.

---

## 📝 Citation

If you find this work useful, please consider citing:

```bibtex
@article{feng2026medexpmem,
  title   = {MedExpMem: Adapting Experience Memory for Differential Diagnosis},
  author  = {Feng, Qianhan and Huang, Zhongzhen and Zhu, Yakun and
             Gu, Yannian and Chu, Winnie Chiu Wing and Zhang, Xiaofan and
             Dou, Qi},
  journal = {arXiv preprint arXiv:2605.22872},
  year    = {2026}
}
```

The official MICCAI 2026 citation will be updated after publication.
