# Information Bazaar

Code for **Redesigning Information Markets in the Era of Language Models** (COLM 2024).

Martin Weiss\*, Nasim Rahaman\*, Manuel Wüthrich, Yoshua Bengio, Li Erran Li†, Bernhard Schölkopf†, Chris Pal†
<br>\* † Equal contribution, random order.

[Paper (OpenReview)](https://openreview.net/forum?id=Zq9Dfj4nBo) · [Earlier version on arXiv](https://arxiv.org/abs/2403.14443)

## Abstract

> Information markets face many challenges leading to instability, inefficiency, and failure, ultimately reducing incentives for the creation and distribution of high-quality information. A long-standing issue for information markets is the Buyer's Inspection Paradox: buyers need to inspect information to assess its value, while sellers must limit inspection to prevent unauthorized use or theft. This paradox results from the information asymmetry present in the market, where sellers know more about the quality of their goods than buyers. This work proposes an information market design that leverages language models to mitigate the Buyer's Inspection Paradox by enabling inspection, comparison, and purchase of information, while algorithmically preventing expropriation. Our experiments (a) show methods that improve the economic rationality of language models, (b) investigate how language model behaviour changes with the price of goods, and (c) evaluate the simulated cost-efficiency of the proposed market under various conditions.

## Status

This is the research code as it was used for the paper. It hasn't been updated since November 2023.

- The dataset isn't in the repo (`data/` is gitignored). The templates expect `data/final_dataset_with_metadata.json` and `data/final_dataset_embeddings.db`.
- Some scripts and notebooks still point at paths on the Mila cluster.
- It uses the pre-0.1 `guidance` API (`guidance.llms`), so current `guidance` releases won't work.
- Dependencies aren't pinned. `requirements.txt` only covers the `llamapi` server.

## Layout

| Path | Contents |
|---|---|
| `bazaar/` | The marketplace: buyer and vendor agents and the simulation loop (`simulator.py`), bulletin board, quotes and principals (`schema.py`), BM25 and embedding retrieval (`database.py`), LLM prompts including the chain-of-thought, direct and debate quote-selection prompts (`lem_utils.py`), dataset construction from arXiv and OpenAlex (`dataset_builder.py`), answer evaluation (`eval.py`, `likert_eval.py`, `elo_eval.py`) |
| `llamapi/` | FastAPI + Celery server that serves Hugging Face models (Llama 2 in the paper) to the agents |
| `scripts/` | `speedmain.py` runs a simulation from a template. `run_rational_choice.py` and `single_good_demand_curve.py` are the rational-choice and price experiments. Slurm launchers. |
| `templates/` | Simulation configs: `FUP-X` (GPT-4), `FUP-NOINSPECT-X` (same, but buyers only see metadata before buying), `BM25-HEURISTIC-X` (BM25 keyword baseline), `BASE-X` (Llama 2 70B via `llamapi`) |
| `mpic_scripts/` | Cluster launchers for the `llamapi` workers |
| `notebooks/` | Dataset generation, analysis and plotting |

## Running a simulation

With the dataset in place:

```bash
pip install -e .
pip install git+https://github.com/inferno-pytorch/speedrun.git
export OPENAI_API_KEY=...   # GPT-3.5 / GPT-4 agents and the GPT-4 judge
export HF_AUTH_TOKEN=...    # Llama 2 weights, if serving them with llamapi
python scripts/speedmain.py experiments/FUP-0 --inherit templates/FUP-X
```

## Citation

```bibtex
@inproceedings{weiss2024redesigning,
  title     = {Redesigning Information Markets in the Era of Language Models},
  author    = {Weiss, Martin and Rahaman, Nasim and W{\"u}thrich, Manuel and Bengio, Yoshua and Li, Li Erran and Sch{\"o}lkopf, Bernhard and Pal, Chris},
  booktitle = {First Conference on Language Modeling},
  year      = {2024},
  url       = {https://openreview.net/forum?id=Zq9Dfj4nBo}
}
```

## License

The code is released under the [MIT License](LICENSE). `notebooks/aistackexchange.json` and `notebooks/large-language-models_questions.json` contain questions and answers from [AI Stack Exchange](https://ai.stackexchange.com), which their authors license under CC BY-SA.
