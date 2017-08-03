# Label-Propagation with Seq2Seq

## 0. Introduction

This project implements label propagation with seq2seq.
It applies [Neural Graph Machines](https://arxiv.org/abs/1703.04818) to Seq2Seq.

### Problem Settings

* Semi-supervised learning techniques such as label propagation are used to solve classification (finite-categories) problems.
* And these approaches produce improvements at some extent.
* However, these approaches are not applied well to solve continuous target (infinite-categories) problems. 
* Therefore, I want to tackle this problem using Neural Graph Machines in this project.


### Some Details

* I test the performance in Neural Machine Translation problem.
* For calculating distance between nodes, I use L1, L2, and Mahalanobis distance metrics.
* Presentation info is given in https://goo.gl/whAbB1

----

### You can explore the whole project code by following jupyter notebook codes.

> **toy_example.ipynb** : contruct 2D sinc function with biased parallel data and unbiased non-parallel data

> **preprocessing.ipynb** : preprocess sentences

> **graph_operations.ipynb** : construct graph from source sentences

> **neural_graph_machines-benchmark.ipynb** : Default Encoder-Attention-Decoder Neural Translation Model

> **neural_graph_machines.ipynb** : Neural Graph Machine Model


## 1. Data Preparation

Experiments are done with IWSLT English-Vietnamese data set.

You can download using **download.sh**

I also use monolingual data from http://www.manythings.org/anki/.

After that, run **preprocessing.ipynb**

## 2. Graph Construction

run **graph_operations.ipynb**

## 3. Experiments

run **neural_graph_machines-benchmark.ipynb** and **neural_graph_machines.ipynb**