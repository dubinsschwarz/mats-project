#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

grep -Fq 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" 2>/dev/null || \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"

PROJECT="/workspace/mats-project"
PERSISTENT_KEY="/workspace/.secrets/mats_github"
JUPYTER_PORT=8890
MCP_PORT=4040

echo "==> Bootstrapping MATS pod"

# -------------------------------------------------------------------
# Basic paths / persistent caches
# -------------------------------------------------------------------

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="/workspace/.cache/huggingface"
export UV_CACHE_DIR="/workspace/.cache/uv"

mkdir -p "$HOME/.local/bin" "$HOME/.ssh" "$HOME/.jupyter"
chmod 700 "$HOME/.ssh" "$HOME/.jupyter"

# -------------------------------------------------------------------
# uv
# -------------------------------------------------------------------

if ! command -v uv >/dev/null 2>&1; then
    echo "==> Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# -------------------------------------------------------------------
# GitHub repo-scoped deploy key
# -------------------------------------------------------------------

if [[ -f "$PERSISTENT_KEY" ]]; then
    echo "==> Restoring GitHub deploy key"
    cp "$PERSISTENT_KEY" "$HOME/.ssh/mats_github"
    chmod 600 "$HOME/.ssh/mats_github"

    if ! grep -q '^Host github-mats$' "$HOME/.ssh/config" 2>/dev/null; then
        cat >> "$HOME/.ssh/config" <<'SSHCFG'

Host github-mats
    HostName github.com
    User git
    IdentityFile ~/.ssh/mats_github
    IdentitiesOnly yes
SSHCFG
    fi

    chmod 600 "$HOME/.ssh/config"
else
    echo "WARNING: persistent GitHub deploy key not found."
fi

# -------------------------------------------------------------------
# Codex CLI
# -------------------------------------------------------------------

if ! command -v codex >/dev/null 2>&1; then
    echo "==> Installing Codex CLI"
    curl -fsSL https://chatgpt.com/codex/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# -------------------------------------------------------------------
# Private ephemeral tokens
# -------------------------------------------------------------------

if [[ ! -f "$HOME/.jupyter/mats-token" ]]; then
    echo "==> Generating Jupyter token"
    uv run --project "$PROJECT" python - <<'PY'
from pathlib import Path
import secrets, os

p = Path.home() / ".jupyter" / "mats-token"
p.write_text(secrets.token_urlsafe(32))
os.chmod(p, 0o600)
PY
fi

if [[ ! -f "$HOME/.jupyter/mcp-token" ]]; then
    echo "==> Generating MCP token"
    uv run --project "$PROJECT" python - <<'PY'
from pathlib import Path
import secrets, os

p = Path.home() / ".jupyter" / "mcp-token"
p.write_text(secrets.token_urlsafe(32))
os.chmod(p, 0o600)
PY
fi

# -------------------------------------------------------------------
# Persistent private Jupyter server
# -------------------------------------------------------------------

if ! tmux has-session -t jupyter 2>/dev/null; then
    echo "==> Starting private Jupyter server"

    tmux new-session -d -s jupyter \
      "cd '$PROJECT' && \
       JUPYTER_TOKEN_FILE='$HOME/.jupyter/mats-token' \
       uv run jupyter lab \
       --ServerApp.ip=127.0.0.1 \
       --ServerApp.port=$JUPYTER_PORT \
       --ServerApp.port_retries=0 \
       --ServerApp.open_browser=False \
       --ServerApp.allow_root=True"
else
    echo "==> Jupyter tmux session already running"
fi

# Give Jupyter a moment to bind.
for _ in $(seq 1 30); do
    if curl -s -o /dev/null "http://127.0.0.1:$JUPYTER_PORT/lab"; then
        break
    fi
    sleep 1
done

# -------------------------------------------------------------------
# Persistent authenticated Jupyter MCP server
# -------------------------------------------------------------------

if ! tmux has-session -t jupyter-mcp 2>/dev/null; then
    echo "==> Starting authenticated Jupyter MCP server"

    tmux new-session -d -s jupyter-mcp \
      "JUPYTER_URL='http://127.0.0.1:$JUPYTER_PORT' \
       JUPYTER_TOKEN=\$(cat '$HOME/.jupyter/mats-token') \
       MCP_TOKEN=\$(cat '$HOME/.jupyter/mcp-token') \
       uvx jupyter-mcp-server@latest start \
       --transport streamable-http \
       --port $MCP_PORT"
else
    echo "==> Jupyter MCP tmux session already running"
fi

for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$MCP_PORT/api/healthz" >/dev/null; then
        break
    fi
    sleep 1
done

# -------------------------------------------------------------------
# Codex MCP configuration
# -------------------------------------------------------------------

codex mcp remove jupyter >/dev/null 2>&1 || true
codex mcp add jupyter --url "http://127.0.0.1:$MCP_PORT/mcp" >/dev/null

CONFIG="$HOME/.codex/config.toml"

if ! grep -A8 '^\[mcp_servers\.jupyter\]' "$CONFIG" \
        | grep -q 'bearer_token_env_var'; then
    sed -i '/^\[mcp_servers\.jupyter\]/a bearer_token_env_var = "JUPYTER_MCP_TOKEN"' "$CONFIG"
fi

# Make the MCP credential available to future interactive shells without
# embedding the actual token in .bashrc.
BASHRC_LINE='export JUPYTER_MCP_TOKEN="$(cat "$HOME/.jupyter/mcp-token" 2>/dev/null)"'

grep -Fq "$BASHRC_LINE" "$HOME/.bashrc" 2>/dev/null || \
    echo "$BASHRC_LINE" >> "$HOME/.bashrc"

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------

echo
echo "==> Bootstrap complete"
echo
echo "Jupyter:"
ss -ltn 2>/dev/null | grep ":$JUPYTER_PORT " || true

echo
echo "Jupyter MCP:"
curl -sf "http://127.0.0.1:$MCP_PORT/api/healthz" || true

echo
echo
echo "Git:"
git -C "$PROJECT" remote -v | head -2

echo
if codex login status >/dev/null 2>&1; then
    echo "Codex login: OK"
else
    echo "Codex login required:"
    echo "  codex login --device-auth"
fi

echo
echo "For this current shell, run:"
echo '  export JUPYTER_MCP_TOKEN="$(cat ~/.jupyter/mcp-token)"'
echo
echo "Then:"
echo "  cd $PROJECT"
echo "  codex"

# Persist Codex defaults across fresh pods.
CODEX_CONFIG="$HOME/.codex/config.toml"
mkdir -p "$HOME/.codex"
touch "$CODEX_CONFIG"

grep -q '^approval_policy = ' "$CODEX_CONFIG" 2>/dev/null || \
    sed -i '1i approval_policy = "on-request"' "$CODEX_CONFIG"

grep -q '^approvals_reviewer = ' "$CODEX_CONFIG" 2>/dev/null || \
    sed -i '1i approvals_reviewer = "auto_review"' "$CODEX_CONFIG"

grep -q '^sandbox_mode = ' "$CODEX_CONFIG" 2>/dev/null || \
    sed -i '1i sandbox_mode = "workspace-write"' "$CODEX_CONFIG"

if ! grep -q '^alternate_screen = "never"$' "$CODEX_CONFIG" 2>/dev/null; then
    if grep -q '^\[tui\]$' "$CODEX_CONFIG" 2>/dev/null; then
        sed -i '/^\[tui\]$/a alternate_screen = "never"' "$CODEX_CONFIG"
    else
        cat >> "$CODEX_CONFIG" <<'TUIEOF'

[tui]
alternate_screen = "never"
TUIEOF
    fi
fi
