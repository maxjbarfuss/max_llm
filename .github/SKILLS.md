---
name: max-llm-agent-skills
description: >
  Dynamic capabilities and pragmatic workflows for agents working on max_llm.
  Builds on the universal AGENTS.md standard with specialized tool use patterns,
  context optimization, and multi-step reasoning strategies. Any agent with tool
  access (Claude, o1, custom models) can leverage these workflows.
version: "1.0"
updated: "2026-02-27"
---

# Agent Skills for max_llm

Practical workflows and tool use patterns for working efficiently on max_llm. This document assumes you have read [AGENTS.md](AGENTS.md) for the universal project standard.

> **Foundation**: This extends but does not replace [AGENTS.md](AGENTS.md). Always follow AGENTS.md first.

## Effective Tool Use Patterns

### 1. Multi-Step Reasoning & Planning
- Use `manage_todo_list` to track complex, multi-step tasks autonomously
- Break work into logical checkpoints; mark tasks in-progress before execution
- For search-heavy tasks, use `search_subagent` to parallelize codebase exploration
- For focused research (docs, web fetch), use `fetch_webpage` or `semantic_search` directly
- Keeps work visible if task is handed off to another agent or human

### 2. Parallel Tool Execution
- **Batch independent operations**: `read_file` (multiple file ranges), `grep_search`, `file_search` can run in parallel
- **Never parallelize**: `run_in_terminal` (sequential only), `run_notebook_cell` (kernel state dependent)
- **Efficiency threshold**: If >5 independent operations needed, consider `search_subagent` to batch them
- **Example pattern**: Scanning 10-file change set → use `get_changed_files`, then parallel reads of affected sections

### 3. Context Efficiency
- Context is a shared resource: balance user message, tool results, and reasoning
- **For large codebases**: Prefer `semantic_search` (returns filtered snippets) over `read_file` on entire files
- **For exploration**: Use `search_subagent` when initial query is uncertain; it auto-discovers relevant files efficiently
- **For verification**: After edits, run tests via `runTests` or `Build_CMakeTools` (structured output is more concise than raw logs)

### 4. Documentation & Code Sync
- After multi-phase implementation, use `get_changed_files` to scan what changed
- Use `multi_replace_string_in_file` for >1 related edits (more efficient than sequential calls)
- Always validate documentation is current before committing (SESSION.md, PLAN.md status)
- For complex diagrams, use `mermaid-diagram-validator` + `mermaid-diagram-preview` before merge

### 5. Testing & Validation
- Use `runTests` (Python) or `Build_CMakeTools` / `RunCtest_CMakeTools` (C++) for structured test execution
- Always call appropriate test tool BEFORE committing; validate full suite passes
- For coverage reports, use `runTests` with `mode="coverage"` to track progress
- Use `test_failure` tool if a test suite fails to get structured diagnostic info

### 6. Terminal & Background Process Management
- Terminal commands run sequentially (`run_in_terminal` with `isBackground=false`)
- For long-running processes (servers, watches), use `isBackground=true` with `await_terminal` / `get_terminal_output`
- Always `kill_terminal` when background process is no longer needed
- Use `terminal_last_command` / `terminal_selection` for smarter, context-aware iterations

### 7. Notebook Editing
- Use `edit_notebook_file` for adding/editing/deleting cells (preferred over terminal magic commands)
- Call `copilot_getNotebookSummary` to understand cell layout before editing
- Use `run_notebook_cell` to execute and validate (not terminal Jupyter commands)

### 8. Build System & Project Setup
- **VS Code extensions**: Use `install_extension` (during workspace setup only)
- **C++ builds**: Prefer `Build_CMakeTools` for better error messages and IDE integration
- **CMake introspection**: Use `ListBuildTargets_CMakeTools`, `ListTests_CMakeTools` to discover project structure without terminal commands
- **Project scaffolding**: Use `create_new_workspace` for complete project initialization

## Recommended Workflows

### Fast Iteration Loop
1. **Plan**: Use `manage_todo_list` to break work into steps
2. **Search**: Parallel `read_file` calls for context gathering
3. **Implement**: Edit files or create new ones (TDD: test first)
4. **Validate**: `runTests` (Python) or `Build_CMakeTools` (C++) before moving on
5. **Document**: `get_changed_files` to verify doc sync, then commit
6. **Reflect**: Update todo list, mark completed items

### Complex Multi-File Refactors
1. Use `search_subagent` to map out all affected files in one pass
2. Parallel `read_file` on identified sections
3. Use `multi_replace_string_in_file` to apply changes atomically
4. Validate with targeted test run (e.g., `runTests` on specific test file)
5. Full `make test-quick` before final commit

### Codebase Exploration (Uncertain Starting Point)
1. Call `search_subagent` with natural language query (e.g., "where is attention implemented?")
2. Review returned file/snippet locations
3. Use parallel `read_file` to examine promising candidates
4. Use `semantic_search` for follow-up queries if needed

### Git & Commit Management
- Use `get_changed_files` to review all diffs before committing
- Prepare comprehensive commit with `run_in_terminal` (single command: `git add -A && git commit -m "..."`)`
- For atomic commits across multiple files, stage them together (avoid multiple commits per message)

## Tool Preferences

| Task | Preferred Tool | Rationale |
|------|---|---|
| Find files by pattern | `file_search` | Fast glob matching |
| Search exact string/regex | `grep_search` (with `includePattern`) | Precise results |
| Semantic/fuzzy search | `semantic_search` (or `search_subagent` if uncertain) | Context-aware, handles synonyms |
| Read code sections | Parallel `read_file` calls (3-5 at once) | Efficient context gathering |
| Edit single file | `replace_string_in_file` | Precise, requires context lines |
| Edit multiple files | `multi_replace_string_in_file` | Atomic, efficient |
| Run Python tests | `runTests` (preferred) or terminal | Structured output, coverage tracking |
| Run C++ build/tests | `Build_CMakeTools` / `RunCtest_CMakeTools` | IDE integration, better error messages |
| Run terminal commands | `run_in_terminal` (sequential, not parallel) | Reliable for shell workflows |
| Plan complex work | `manage_todo_list` | Tracks progress, provides visibility |
| Explore unknown codebase | `search_subagent` | Efficient parallelization, returns file map |

## Efficiency Guidelines

- **Context usage**: Each `read_file` call costs context (proportional to file range); tool results and reasoning also consume budget
- **When context is tight**: Switch to `semantic_search` for conciseness, use `search_subagent` to batch searches, or reduce log verbosity
- **Multi-session work**: Always summarize progress in SESSION.md before ending session; start fresh with clear checkpoint if continuing Phase work
