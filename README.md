# najdi-llm

Data generation, fine-tuning and serving for a Najdi Saudi Arabic chat model.

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env
```

Put your `HF_TOKEN` in `.env`. `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID` are optional and send progress messages.

Install the extra you need for a stage: `generate`, `train`, `serve` or `mix`, for example `pip install -e ".[train]"`.

All Hugging Face repo names in `configs/` are empty. Before you run a stage, fill in the ones it uses: inputs are repos you already have, and outputs should be new names so nothing gets overwritten. A bare name like `my-mix` is created under the account of your `HF_TOKEN`. In `configs/chat-data.yaml`, set `creator_name` if the assistant should say who made it.

## Stages

| Stage | Command | Hardware |
|---|---|---|
| Science and math data | `najdi science -c configs/science.yaml` | GPU |
| Medical data | `najdi science -c configs/medical.yaml` | GPU |
| Everyday chat and identity data | `najdi chat-data -c configs/chat-data.yaml` | GPU |
| Clean science, medical and Voho | `najdi clean -c configs/clean.yaml` | CPU |
| Build the training mix | `najdi mix -c configs/mix.yaml` | CPU |
| Compare base models | `najdi compare -c configs/compare.yaml` | GPU |
| Fine-tune | `najdi train -c configs/train.yaml` | GPU |
| Chat page | `najdi serve -c configs/serve.yaml` | GPU |

Run them in this order: science, medical, chat-data, clean, mix, train, serve.

Change any config value for one run with `--set`:

```bash
najdi train -c configs/train.yaml --set epochs=3 --set push_merged=true
```

Generation stages start in pilot mode. Set `pilot: false` for the full run. Stopped runs continue from where they left off. Every Hugging Face repo is created private.

## Docker

```bash
make build
make train
make serve
```

Outputs go to `runs/`. The Hugging Face cache is kept in a Docker volume. The chat page runs on port 7860.

## Tests

```bash
make test
make lint
```
