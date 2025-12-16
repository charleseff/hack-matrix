# 868-HACK Game Architecture

## Project Conventions

- **Plan files** go in `plans/` directory

---

## Turn Structure

### Player's Turn
Player takes one of these actions:
- **Move** → Turn ends, enemy turn begins
- **Attack** → Turn ends, enemy turn begins
- **Siphon** → Turn ends, enemy turn begins
- **Execute Program** → Turn does NOT end (can chain multiple programs)
  - **Exception: Wait program** → Turn ends without requiring move/attack/siphon

### Turn Transition (when player's turn ends)
1. Turn counter increments
2. Enemy turn begins

### Enemy's Turn
1. Transmissions spawn (convert to enemies based on timer)
2. Enemies move/attack
3. Scheduled tasks execute
4. Enemy status resets (disable counters, stun flags)
