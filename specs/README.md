# Specs Directory

Design documents and implementation specifications for HackMatrix.

## Current Focus

**Active spec:** [jax-dummy-env.md](./jax-dummy-env.md)

Implement the minimal JAX dummy environment first. This establishes JAX patterns and enables parity testing before the full port.

## Specs Index

| Spec | Status | Description |
|------|--------|-------------|
| [jax-dummy-env.md](./jax-dummy-env.md) | **Active** | Minimal JAX dummy environment for plug-and-play testing with Swift env |
| [jax-implementation.md](./jax-implementation.md) | Deferred | Full JAX port of game logic (depends on dummy env) |

## Usage

Before starting a major feature or architectural change, create a spec document here.

**Status lifecycle:**
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
