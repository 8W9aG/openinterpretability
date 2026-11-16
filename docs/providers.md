# Running experiments with other model providers

An experiment loop has two kinds of model. The **driver** runs the Claude Code session that plans, audits and lands each experiment. The **proposer and reviewer** write the hypothesis and the probes. This page shows how to drive a loop with a model from a provider other than Anthropic (issue #3245).

## Profiles

`openinterp/data/providers.toml` holds one profile per provider. A profile gives the provider's base URL, its model names and any environment that it needs. It also says where its API key lives. A profile never holds the key itself.

```bash
uv run open-interpretability provider list     # each profile, its model, and whether its key is set
```

| Profile | Driver model | How Claude Code reaches it | Key |
| --- | --- | --- | --- |
| `anthropic` | `claude-haiku-5-5` | directly | `ANTHROPIC_API_KEY`, or Claude Code's own login |
| `deepseek` | `deepseek-flash[1m]` | DeepSeek's Anthropic-compatible API | `DEEPSEEK_API_KEY` |
| `zai` | `glm-5.3-flash[1m]` | Z.ai's Anthropic-compatible API | `ZAI_API_KEY` |
| `muse` | `muse-spark-1.3-contributor` | Meta's Anthropic-compatible API | `MUSE_API_KEY` |
| `gemini` | `gemini/gemini-3.8-flash` | a LiteLLM proxy on port 4000 | `GEMINI_API_KEY` |
| `grok` | Grok, through the proxy | a LiteLLM proxy on port 4001 | `XAI_API_KEY` |

## API keys

Put each key in its environment variable, or in the macOS keychain:

```bash
security add-generic-password -s openinterp-deepseek -a "$USER" -w   # asks for the key
```

The keychain entry names are in the profiles (`key_keychain`). The key is read from the environment variable first, then from the keychain. `open-interpretability provider exec` puts it only in the environment of the command that it runs. Nothing prints it or writes it to a file.

## Drive a loop

```bash
scripts/experiment-loop.sh --provider deepseek --proposer external/deepseek-flash --reviewer external/claude-sonnet-5-5
```

`--provider` sets the driver to the profile's model, so the script passes `--model` once. `--driver` still overrides it. The proposer and reviewer are separate: with `external/<name>`, the driving session's subagents answer them. Give a reviewer that differs from the proposer, or every review is self-reviewed.

Each experiment's log ends with a `driver:` line that names the provider and its model. The session file (`.json`) gets a `driver` field. Claude Code prices every model as if it were an Anthropic one. So for any other provider, the loop keeps that figure as `claude_code_cost_usd`, and `open-interpretability costs` counts the session as unpriced.

`open-interpretability loop`, which runs with no Claude Code session, takes the same profiles. There, `--provider` gives the proposer and the reviewer from the profile's LiteLLM model (`litellm_model`):

```bash
uv run open-interpretability loop --provider deepseek --reviewer anthropic/claude-sonnet-5-5
```

## Provider notes

- **DeepSeek.** Claude Code's tool search causes a 422 error on `tool_definition`, so the profile sets `ENABLE_TOOL_SEARCH=false`. The context limit is 1M tokens, and a session can pass it. So the profile compacts at 800,000 tokens (`CLAUDE_CODE_AUTO_COMPACT_WINDOW`).
- **Z.ai GLM.** The profile also compacts at 800,000 tokens, below the 1M context.
- **Muse Spark.** The profile turns off tool search, as for DeepSeek.
- **Gemini and Grok.** They have no Anthropic-compatible API, so Claude Code talks to a local [LiteLLM](https://docs.litellm.ai/) proxy. `provider exec` starts the proxy when nothing listens on its port. It needs `litellm[proxy]` installed and a config file at the path in the profile (`~/.litellm-gemini.yaml` or `~/.litellm-grok.yaml`). The proxy reads the key from its environment variable.

A Gemini proxy config:

```yaml
model_list:
  - model_name: gemini/gemini-3.8-flash
    litellm_params:
      model: gemini/gemini-3.8-flash
      api_key: os.environ/GEMINI_API_KEY
```

Claude Code accepts only model names that it knows. So the Grok config maps an Anthropic model name to the Grok model:

```yaml
model_list:
  - model_name: claude-3-5-sonnet-20241022
    litellm_params:
      model: xai/<your Grok model>
      api_key: os.environ/XAI_API_KEY
```

## Add a provider

Add a table to `openinterp/data/providers.toml`, with the fields its header lists. Then add a row to the table above, and run `uv run pytest tests/test_providers.py`.
