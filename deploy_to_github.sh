#!/usr/bin/env bash
# =============================================================================
#  deploy_to_github.sh
#  One-click script: train PINN, generate assets, create GitHub repo, push.
# =============================================================================
set -euo pipefail
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}▶  $*${NC}"; }
success() { echo -e "${GREEN}✓  $*${NC}"; }
warn()    { echo -e "${YELLOW}⚠  $*${NC}"; }
error()   { echo -e "${RED}✗  $*${NC}"; exit 1; }

echo -e "\n${BOLD}════════════════════════════════════════════${NC}"
echo -e "${BOLD}   PINN Oscillator — GitHub Deploy Script   ${NC}"
echo -e "${BOLD}════════════════════════════════════════════${NC}\n"

# ── 0. Config — EDIT THESE ───────────────────────────────────────────────────
REPO_NAME="${REPO_NAME:-pinn_oscillator}"
GITHUB_USERNAME="${GITHUB_USERNAME:-}"      # set via env or prompted below
REPO_DESCRIPTION="Physics-Informed Neural Network (PINN) solving the damped harmonic oscillator in PyTorch"
PRIVATE="${PRIVATE:-false}"                 # set to "true" for a private repo

# Training hyperparameters (override via env vars)
OMEGA0="${OMEGA0:-2.0}"
GAMMA="${GAMMA:-0.3}"
EPOCHS="${EPOCHS:-5000}"
HIDDEN="${HIDDEN:-64}"
N_LAYERS="${N_LAYERS:-4}"
GIF_FRAMES="${GIF_FRAMES:-60}"

# ── 1. Checks ─────────────────────────────────────────────────────────────────
info "Checking prerequisites…"

command -v git    >/dev/null 2>&1 || error "git is not installed."
command -v python >/dev/null 2>&1 || command -v python3 >/dev/null 2>&1 || error "Python is not installed."
command -v gh     >/dev/null 2>&1 || error "GitHub CLI (gh) is not installed.\n  Install: https://cli.github.com"

# Ensure gh is authenticated
gh auth status >/dev/null 2>&1 || error "GitHub CLI is not authenticated.\n  Run: gh auth login"

PYTHON=$(command -v python3 2>/dev/null || command -v python)

# Prompt for username if not set
if [[ -z "$GITHUB_USERNAME" ]]; then
    GITHUB_USERNAME=$(gh api user --jq '.login' 2>/dev/null)
    [[ -z "$GITHUB_USERNAME" ]] && read -rp "  GitHub username: " GITHUB_USERNAME
fi

success "Prerequisites OK   (user: ${GITHUB_USERNAME})"

# ── 2. Python dependencies ────────────────────────────────────────────────────
info "Installing Python dependencies…"
$PYTHON -m pip install --quiet torch numpy matplotlib pillow
success "Dependencies installed"

# ── 3. Train PINN + generate assets ──────────────────────────────────────────
info "Training PINN  (ω₀=${OMEGA0}, γ=${GAMMA}, epochs=${EPOCHS})…"
info "This takes 1-5 minutes on CPU. Grab a ☕"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

$PYTHON main.py \
    --omega0        "$OMEGA0"    \
    --gamma         "$GAMMA"     \
    --epochs        "$EPOCHS"    \
    --hidden        "$HIDDEN"    \
    --n_layers      "$N_LAYERS"  \
    --gif_snapshots "$GIF_FRAMES"\
    --save_dir      results

success "Training complete"

# ── 4. Copy assets for README ─────────────────────────────────────────────────
info "Copying assets…"
mkdir -p assets
cp results/training.gif assets/
cp results/solution.png assets/
cp results/loss.png     assets/
success "Assets ready"

# ── 5. Initialise git repo ────────────────────────────────────────────────────
info "Initialising git repository…"

if [[ ! -d ".git" ]]; then
    git init -b main
    success "Git repo initialised"
else
    warn ".git already exists — skipping init"
fi

# ── 6. Update .gitignore to TRACK assets ─────────────────────────────────────
# Assets are pre-generated and should be committed so GitHub README shows them.
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
*.egg-info/
.eggs/
dist/ build/ *.so
.venv/ venv/ env/
.ipynb_checkpoints/ *.ipynb
results/*.pt
.DS_Store Thumbs.db
.vscode/ .idea/
EOF
success ".gitignore updated (assets are tracked)"

# ── 7. Stage and commit ───────────────────────────────────────────────────────
info "Staging files…"
git add .
git commit -m "feat: PINN damped harmonic oscillator in PyTorch 🌊

- Physics-Informed Neural Network solving x'' + 2γx' + ω₀²x = 0
- PyTorch autograd for exact ODE residual gradients
- Adam optimiser + ReduceLROnPlateau scheduler
- Training animation GIF + solution/loss plots
- Unit tests (pytest)
- CLI entry point with configurable hyperparameters

Config: ω₀=${OMEGA0}, γ=${GAMMA}, epochs=${EPOCHS}" 2>/dev/null \
    || warn "Nothing new to commit — files already staged"

success "Committed"

# ── 8. Create GitHub remote ───────────────────────────────────────────────────
info "Creating GitHub repository '${REPO_NAME}'…"

REMOTE_EXISTS=$(gh repo list "$GITHUB_USERNAME" --json name --jq ".[].name" 2>/dev/null | grep -x "$REPO_NAME" || true)

if [[ -n "$REMOTE_EXISTS" ]]; then
    warn "Repo '${GITHUB_USERNAME}/${REPO_NAME}' already exists — will push to it"
    REMOTE_URL="https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"
else
    gh repo create "$REPO_NAME" \
        --description "$REPO_DESCRIPTION" \
        --public \
        --source=. \
        --remote=origin \
        --push 2>/dev/null && {
        success "Repo created and pushed!"
        PUSH_DONE=1
    }
fi

# ── 9. Push ───────────────────────────────────────────────────────────────────
if [[ -z "${PUSH_DONE:-}" ]]; then
    if ! git remote get-url origin >/dev/null 2>&1; then
        git remote add origin "https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"
    fi
    git push -u origin main --force
    success "Pushed to origin/main"
fi

# ── 10. Add topics ────────────────────────────────────────────────────────────
info "Adding repo topics…"
gh repo edit "${GITHUB_USERNAME}/${REPO_NAME}" \
    --add-topic "pinn" \
    --add-topic "pytorch" \
    --add-topic "physics-informed-neural-networks" \
    --add-topic "harmonic-oscillator" \
    --add-topic "scientific-machine-learning" \
    --add-topic "deep-learning" 2>/dev/null || warn "Could not set topics (non-fatal)"

# ── Done ──────────────────────────────────────────────────────────────────────
REPO_URL="https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"

echo ""
echo -e "${BOLD}${GREEN}════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}   🎉  Repository published successfully!   ${NC}"
echo -e "${BOLD}${GREEN}════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BOLD}URL:${NC}  ${REPO_URL}"
echo ""
echo -e "  ${CYAN}Opening in browser…${NC}"
gh repo view "${GITHUB_USERNAME}/${REPO_NAME}" --web 2>/dev/null || true
