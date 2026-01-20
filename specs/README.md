# Specs Directory

Design documents and implementation specifications for HackMatrix.

## Current Focus

**Active spec:** [test-reorganization.md](./test-reorganization.md)

Reorganize tests, add scheduled task testing, and create implementation-level tests for hidden state.

**When starting work, read the active spec first (and any related specs needed for context) without asking.** When creating or editing specs, link to other specs where useful for cross-referencing.

## Specs Index

| Spec | Status | Description |
|------|--------|-------------|
| [game-mechanics.md](./game-mechanics.md) | **Reference** | Authoritative game mechanics reference (single source of truth) |
| [jax-dummy-env.md](./jax-dummy-env.md) | **Complete** | Minimal JAX dummy environment for plug-and-play testing with Swift env |
| [env-parity-tests.md](./env-parity-tests.md) | **Complete** | Interface-based test suite for validating env implementations |
| [test-reorganization.md](./test-reorganization.md) | **Active** | Reorganize tests, add scheduled task testing, implementation-level tests |
| [ci-setup.md](./ci-setup.md) | **Draft** | GitHub Actions CI for Swift and Python tests |
| [jax-implementation.md](./jax-implementation.md) | **Deferred** | Full JAX port of game logic (depends on test-reorganization) |

## Usage

Before starting a major feature or architectural change, create a spec document here.

**Status lifecycle:**
- **Draft** - Spec needs further discussion before implementation
- **Active** - Currently being implemented
- **Deferred** - Planned but not yet started (may depend on other specs)
- **Complete** - Fully implemented and verified

When finishing a spec, update its status to **Complete** and set the next spec to **Active** in both the Current Focus section and the Specs Index table.

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
