# Specs Directory

Design documents and implementation specifications for HackMatrix. As part of studying this document you may inspect the specs in the index provided.

**Repository:** https://github.com/charleseff/hack-matrix.git

## Current Focus

**Active spec:** [reward-parity.md](./reward-parity.md) - JAX reward parity with Swift RewardCalculator (5 missing components + 2 fixes)

**When starting work, read the active spec first (and any related specs needed for context) without asking.** When creating or editing specs, link to other specs where useful for cross-referencing.

## Specs Index

| Spec | Status | Description |
|------|--------|-------------|
| [training-speedups.md](./training-speedups.md) | **Deferred** | TPU training on Google TRC (10x speedup target) |
| [reward-parity.md](./reward-parity.md) | **Active** | JAX reward parity with Swift RewardCalculator (5 missing components + 2 fixes) |
| [wandb-purejaxrl.md](./wandb-purejaxrl.md) | **Complete** | Wandb integration for PureJaxRL training |
| [game-mechanics.md](./game-mechanics.md) | **Reference** | Authoritative game mechanics reference (single source of truth) |
| [training-reference.md](./training-reference.md) | **Reference** | RL training commands, monitoring, and troubleshooting |
| [jax-dummy-env.md](./jax-dummy-env.md) | **Complete** | Minimal JAX dummy environment for plug-and-play testing with Swift env |
| [env-parity-tests.md](./env-parity-tests.md) | **Complete** | Interface-based test suite for validating env implementations |
| [observation-and-attack-fixes.md](./observation-and-attack-fixes.md) | **Complete** | Add siphonCenter to obs space, ATK+ usable twice per stage |
| [test-reorganization.md](./test-reorganization.md) | **Complete** | Reorganize tests, add scheduled task testing (depends on observation-and-attack-fixes) |
| [jax-implementation.md](./jax-implementation.md) | **Complete** | Full JAX port of game logic for TPU training |
| [purejaxrl-integration.md](./purejaxrl-integration.md) | **Complete** | PureJaxRL integration with action-masked PPO for TPU training |
| [testing-and-linting.md](./testing-and-linting.md) | **Complete** | Pre-commit hooks, parallel pytest, ruff linting |
| [ci-setup.md](./ci-setup.md) | **Complete** | GitHub Actions CI for Swift and Python tests |
| [continuous-integration.md](./continuous-integration.md) | **Complete** | CI documentation and usage guide |

## Usage

Before starting a major feature or architectural change, create a spec document here.

**Status lifecycle:**
- **Draft** - Spec needs further discussion before implementation
- **Active** - Currently being implemented
- **Deferred** - Planned but not yet started (may depend on other specs)
- **Complete** - Fully implemented and verified

**When finishing a spec:**
1. Move `IMPLEMENTATION_PLAN.md` to `specs/[spec-name]-IMPLEMENTATION_PLAN.md`
2. Update the spec's status to **Complete** in this README
3. Set the next spec to **Active** in both Current Focus and Specs Index

## Firewall Notes

The dev container has an outbound firewall with an allowlist. If a spec requires network access to new domains (e.g., pip install from a new package source), you may need to update the firewall.

**Temporary (current session):**
```bash
# Resolve and add domain IPs
for ip in $(dig +short example.com | grep -E '^[0-9]'); do
    sudo ipset add allowed-domains "$ip"
done
```

**Permanent (survives container rebuild):**
Add the domain to `.devcontainer/init-firewall.sh` in the allowed domains list.

**Currently allowed:** GitHub, PyPI, Anthropic API, VS Code, npm, Google Storage (for JAX wheels).
