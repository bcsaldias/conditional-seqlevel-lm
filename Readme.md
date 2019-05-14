# Multiply-conditioned text generation with a sequence-level loss

The resources in this repository are part of the final class project for Harvard CS 287. Experiments and results are fully described in the write up that eventually would be made available.

## Abstract

Language generation focuses on producing fluent sentences, commonly by learning to predict the next word in a sequence.
Conditional generation allows a model to constrain this generation on additional attributes such as style or another source language. However, most conditional models are still trained by minimizing cross-entropy loss, without explicitly building the conditions into the loss. In this work, we develop a REINFORCE-like algorithm to penalize failures to match the desired constraints and to help reduce the mismatch between the loss and evaluation metrics. We implement and train a baseline model according to the current state-of-the-art for comparison. We train one- and two-labels conditional models using this learning procedure, and obtain predictions with both greedy and beam search. Our models greatly improve the accuracy with which generation satisfies the desired constraints, with no decrease in performance in terms of fluency (perplexity). Finally, we provide the community with the implementations of the baseline model and our proposed training framework.
