# open-truco

<p align="center">
  <img src="assets/basto.svg" height="50" alt="Basto" />
  <img src="assets/copa.svg" height="50" alt="Copa" />
  <img src="assets/espada.svg" height="50" alt="Espada" />
  <img src="assets/oro.svg" height="50" alt="Oro" />
</p>

This repository contains tools for training and evaluating reinforcement learning agents for the South American card game [Truco](https://es.wikipedia.org/wiki/Truco_argentino), built on top of Google DeepMind's [OpenSpiel](https://github.com/deepmind/open_spiel) framework.

The full details of the project can be found in our [paper](paper/paper.pdf).

## Installation

Use the project manager [uv](https://uv.dev/) to install **open-truco**.

```bash
git clone https://github.com/segusantos/open-truco.git
cd open-truco
uv sync
git clone https://github.com/segusantos/open_spiel.git
cd open_spiel
git checkout truco
uv run ./install.sh
uv pip install -e .
cd ..
```

## Usage

You can train and evaluate agents using the provided scripts. For instance, to train an NFSP agent, execute:

```bash
uv run python -m scripts.train_nfsp
```
