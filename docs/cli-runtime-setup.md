# CLI Runtime & Environment Setup

Back to [README](../README.md) · See also [quickstart.md](quickstart.md) and [execution-modes.md](execution-modes.md)

`real` and `hybrid` mode require a working CLI runtime installation with provider bootstrap. The default benchmark execution modes (`multi`, `mock`, `martin`) do not require external services and run hermetically in Python.

## Check the CLI

```bash
openclaw status --json
openclaw agents list --json
openclaw channels list --json
```

## Install into the active conda environment

```bash
conda activate CLI-Agent
npm config set prefix "$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
npm install -g openclaw@latest
openclaw --version
```

## Start the gateway

```bash
openclaw onboard --install-daemon
openclaw gateway status
openclaw status --json
```

## Bootstrap Google-backed providers

For online Calendar, Gmail, or Tasks providers, place a Google OAuth desktop client at `~/.openclaw/client_secret.json` and bootstrap the required channels:

```bash
python scripts/login_required_channels.py \
  --split dev \
  --google-client-secret-file ~/.openclaw/client_secret.json \
  --google-email-client-secret-file ~/.openclaw/client_secret.json \
  --email-provider auto
```

## Useful provider environment variables

- `CLASHARNESS_EMAIL_PROVIDER=auto|google_api|himalaya|mock`
- `CLASHARNESS_TASKS_PROVIDER=auto|google_api|mock`
- `CLASHARNESS_CALENDAR_PROVIDER=auto|google_api|gcalcli|khal|mock`
- `CLASHARNESS_ENABLE_ONLINE_DATA=1`
- `CLASHARNESS_STRICT_ONLINE_DATA=1`
- `CLASHARNESS_GOOGLE_GMAIL_SCOPES`
- `CLASHARNESS_GOOGLE_TASKS_SCOPES`
- `CLASHARNESS_HIMALAYA_ACCOUNT`
