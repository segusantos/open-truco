# open-truco

This repository contains tools for training and evaluating reinforcement learning agents for the South American card game [Truco](https://es.wikipedia.org/wiki/Truco_argentino), built on top of [OpenSpiel](https://github.com/deepmind/open_spiel).

## Quick Start

To set up the development environment, run the following commands ([uv](https://uv.dev/) is used as project manager):

```bash
uv venv
source .venv/bin/activate
git clone https://github.com/segusantos/open_spiel.git
cd open_spiel
git checkout truco
./install.sh
uv pip install -e .
cd .. 
```
