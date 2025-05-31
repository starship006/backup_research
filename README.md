# Self-Repair in Language Models

[![Paper](https://img.shields.io/badge/arXiv-2402.15390-b31b1b.svg)](https://arxiv.org/abs/2402.15390)
[![ICML 2024](https://img.shields.io/badge/ICML-2024-blue.svg)](https://icml.cc/virtual/2024/poster/34973)
[![SeT LLM @ ICLR 2024](https://img.shields.io/badge/SeT%20LLM%20%40%20ICLR-2024%20(Oral)-green.svg)](https://set-llm.github.io/)

This repository contains the code and experimental framework for [**"Explorations of Self-Repair in Language Models"**](https://arxiv.org/abs/2402.15390), demonstrating how transformer models compensate for ablated attention heads through distributed repair mechanisms. 

**Code Organization:** All the files in `FINAL_figure_files` hold the most important code, while other miscellaneous files have been stored in `extra`. By default, the code wasn't written to be read and is messy: for any questions regarding the code, contact Cody Rushing at thisiscodyr@gmail.com

## Abstract

Prior interpretability research studying narrow distributions has preliminarily identified **self-repair**: a phenomenon where if components in large language models are ablated, later components will change their behavior to compensate. 

Our work builds off this past literature, demonstrating that self-repair exists on a variety of model families and sizes when ablating individual attention heads on the full training distribution. We further show that on the full training distribution self-repair is:

- **Imperfect**: the original direct effect of the head is not fully restored
- **Noisy**: the degree of self-repair varies significantly across different prompts (sometimes overcorrecting beyond the original effect)

We highlight two different mechanisms that contribute to self-repair:
1. Changes in the final LayerNorm scaling factor (which can repair up to 30% of the direct effect) 
2. Sparse sets of neurons implementing Anti-Erasure

We additionally discuss the implications of these results for interpretability practitioners and close with a more speculative discussion on the mystery of why self-repair occurs in these models at all, highlighting evidence for the **Iterative Inference hypothesis** in language models, a framework that predicts self-repair.

<img width="952" alt="image" src="https://github.com/user-attachments/assets/45636f76-8ade-41c4-800a-e2096f1e761c" />
