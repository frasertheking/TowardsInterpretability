<div align="center">

![logo](https://github.com/frasertheking/TowardsInterpretability/blob/main/figures/banner.png?raw=true)

Understanding physical features in toy precipitation models using sparse autoencoders, maintained by [Fraser King](https://frasertheking.com/)

</div>

## Overview

Building on insights from [Toy Models of Superpositon](https://transformer-circuits.pub/2022/toy_model/index.html), we conduct a top-down analysis of commonly used single-hidden-layer physical models for precipitation, aiming to identify what physical features (if any) these models learn. While we do not claim these findings are fully interpretable, we highlight this technique as a valuable step toward understanding the internal decision-making processes of such models and pinpointing their strengths, weaknesses, and uncertainties for broader application across the Geosciences. By clarifying these processes, we can also bolster trust in applied machine learning systems—an increasingly critical goal as these models are used to inform high-stakes decisions. Overall, this work seeks to examine the core structure of neural networks, their sensitivities to observation-based inputs and initial conditions, and the ways in which they self-organize to address complex predictive tasks.

![overview](https://github.com/frasertheking/TowardsInterpretability/blob/main/figures/im1.png?raw=true)

The overarching structure of our approach, adapted from the work of Nelson Elhage and Christopher Olah, is illustrated below. In this model-agnostic framework, we first train sparse autoencoders on MLP activations, and then examine the bottleneck layer activations to uncover potential monosemantic physical features. These features may otherwise be obscured by the polysemantic nature of the default MLP activations—an effect commonly attributed to network superposition. We anticipate this approach could generalize well to more complex, multi-layer networks in future work.

![overview](https://github.com/frasertheking/TowardsInterpretability/blob/main/figures/im2.png?raw=true)

In this repository, we provide a select set of Jupyter notebooks and processing scripts that can be used to reproduce our results, or be adapted to new problems in the future.

## Installation

    git clone https://github.com/frasertheking/TowardsInterpretability.git
    conda env create -f env.yml
    conda activate circuits

## Google Colab Examples

**Classifier:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lNTkPtO2ZNvSHTR7r03Gu1FqKEdTra30?usp=sharing)

**Regressor:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_31ggr3UmPeBC5xsldeYVQ54yoWvyX3r?usp=sharing)

## NN Inspector

If you'd like to play around with these models in real-time, check out our WIP [interactive neural network inspector application repository](https://github.com/frasertheking/Inspector/).

![training](https://github.com/frasertheking/TowardsInterpretability/blob/main/figures/training.gif?raw=true)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Note that, as a living project, code is not as clean as it could (should) be, and unit tests need to be produced in future iterations to maintain stability.

## Authors & Contact

- *Fraser King, University of Michigan, (kingfr@umich.edu)
- Claire Pettersen, University of Michigan
- Derek Posselt, NASA Jet Propulsion Laboratory
- Sarah Ringerud, NASA Goddard Space Flight Center
- Yan Xie, University of Michigan

## Funding

This project was primarily funded by NASA New (Early Career) Investigator Program (NIP) grant at the [University of Michigan](https://umich.edu).
