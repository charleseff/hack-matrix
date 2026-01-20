#!/bin/bash
# Filter stream-json output to readable text
# Usage: claude ... --output-format=stream-json | ./json2text.sh

while IFS= read -r line; do
    type=$(echo "$line" | jq -r '.type // empty' 2>/dev/null)

    case "$type" in
        system)
            subtype=$(echo "$line" | jq -r '.subtype // empty')
            if [ "$subtype" = "init" ]; then
                echo "=== Session started ==="
            fi
            ;;
        assistant)
            # Extract text content
            text=$(echo "$line" | jq -r '.message.content[]? | select(.type=="text") | .text // empty' 2>/dev/null)
            if [ -n "$text" ]; then
                echo "$text"
            fi

            # Count tool uses to detect parallel calls
            tool_count=$(echo "$line" | jq '[.message.content[]? | select(.type=="tool_use")] | length' 2>/dev/null)

            if [ "$tool_count" -gt 1 ] 2>/dev/null; then
                echo "  ⚡ $tool_count tools in parallel:"
            fi

            # Extract tool uses
            echo "$line" | jq -r '.message.content[]? | select(.type=="tool_use") |
                if .name == "Task" then
                    "  🤖 AGENT[\(.input.subagent_type // "unknown")]: \(.input.description // .input.prompt[0:60])"
                elif .name == "Read" then
                    "  → Read: \(.input.file_path)"
                elif .name == "Glob" then
                    "  → Glob: \(.input.pattern) in \(.input.path // ".")"
                elif .name == "Grep" then
                    "  → Grep: \"\(.input.pattern)\" in \(.input.path // ".")"
                elif .name == "Bash" then
                    "  → Bash: \(.input.command | .[0:120])\(if (.input.command | length) > 120 then "..." else "" end)"
                elif .name == "Edit" then
                    "  → Edit: \(.input.file_path)"
                elif .name == "Write" then
                    "  → Write: \(.input.file_path)"
                else
                    "  → \(.name): \(.input | tostring | .[0:150])\(if (.input | tostring | length) > 150 then "..." else "" end)"
                end' 2>/dev/null
            ;;
        user)
            # Check for agent results (Task tool results)
            agent_result=$(echo "$line" | jq -r '.tool_use_result.agent_id // empty' 2>/dev/null)
            if [ -n "$agent_result" ]; then
                echo "  ✓ Agent completed: $agent_result"
            fi
            ;;
        agent)
            # Subagent activity (if streamed)
            agent_type=$(echo "$line" | jq -r '.agent_type // empty' 2>/dev/null)
            agent_msg=$(echo "$line" | jq -r '.message // empty' 2>/dev/null)
            if [ -n "$agent_type" ]; then
                echo "    [$agent_type] $agent_msg"
            fi
            ;;
        result)
            # Final result
            result=$(echo "$line" | jq -r '.result // empty' 2>/dev/null)
            if [ -n "$result" ]; then
                echo ""
                echo "=== Result ==="
                echo "$result"
            fi
            ;;
    esac
done
