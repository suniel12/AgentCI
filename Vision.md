# Agent CI Vision

**Agent CI is a "Cost-Aware" Wedge for AI Agent Testing.**

## Philosophy

Agent CI aims to provide a robust CI/CD pipeline specifically designed for AI agents. We recognize that AI agents are inherently stochastic and require a different approach to testing than traditional software.

Key principles:
1. **Trace-First, Not Test-First:** Every piece of data flows through a single abstraction: the Trace.
2. **Cost Awareness:** Catch cost spikes and inefficiencies before they hit production.
3. **Seamless Integration:** Integrate with existing developer workflows (pytest, GitHub Actions).

## Goals

- Ship a working, pip-installable pytest plugin for AI agent regression testing.
- Reach 500+ GitHub stars.
- Prove that developers need CI-native testing for agent tool calls and cost behavior.

## Why this matters

When you add `assert_handoff(agent_a, agent_b)` in Phase 2, you're just querying Spans within an existing Trace. No schema changes needed. The architecture is designed to scale with complexity.
