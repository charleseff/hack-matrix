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
                else
                    "  → \(.name): \(.input | tostring | .[0:80])"
                end' 2>/dev/null
            ;;
        user)
            # Check for agent results (Task tool results)
            agent_result=$(echo "$line" | jq -r '.tool_use_result.agent_id // empty' 2>/dev/null)
            if [ -n "$agent_result" ]; then
                echo "  ✓ Agent completed: $agent_result"
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
