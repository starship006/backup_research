# Explorations of Self-Repair in Language Models

This Repository holds the code used to write [Explorations of Self-Repair in Language Models](https://arxiv.org/abs/2402.15390). All the files in `FINAL_figure_files` hold the most important code, wihle other miscellaneous files have been stored in extra.

By defualt, the code wasn't written to be read and is messy: as such, for any questions regarding the code, contact Cody Rushing at thisiscodyr@gmail.com


## Abstract

Prior interpretability research studying narrow distributions has preliminarily identified self-repair, a phenomena where if components in large language models are ablated, later components will change their behavior to compensate. Our work builds off this past literature, demonstrating that self-repair exists on a variety of models families and sizes when ablating individual attention heads on the full training distribution. We further show that on the full training distribution self-repair is imperfect, as the original direct effect of the head is not fully restored, and noisy, since the degree of self-repair varies significantly across different prompts (sometimes overcorrecting beyond the original effect). We highlight two different mechanisms that contribute to self-repair, including changes in the final LayerNorm scaling factor (which can repair up to 30% of the direct effect) and sparse sets of neurons implementing Anti-Erasure. We additionally discuss the implications of these results for interpretability practitioners and close with a more speculative discussion on the mystery of why self-repair occurs in these models at all, highlighting evidence for the Iterative Inference hypothesis in language models, a framework that predicts self-repair. 
