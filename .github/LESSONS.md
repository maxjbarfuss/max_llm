# Agent Lessons

> Maintained by [@maxjbarfuss](https://github.com/maxjbarfuss). Read this at the start of every session (see [SKILLS.md](SKILLS.md#working-rule)).
>
> Each lesson captures a specific mistake observed in an agent session and the correct behavior. Agents must internalize these before starting work.

---

## L001 — Never Use MCP Git Tools

**Observed behavior**: Agent used GitKraken, GitLens, or another MCP git server for repository operations instead of the local `git` CLI.

**Correct approach**: Use only local `git` commands for all git operations (`git status`, `git add`, `git commit`, `git diff`, `git log`, `git push`). Never invoke any MCP server for git operations.

---

## L002 — Always Update SESSION.md at Session End

**Observed behavior**: Agent completed work and committed without updating `docs/SESSION.md` (clearing scratch pad, adding log entry).

**Correct approach**: Before every commit, clear the Scratch Pad section of `docs/SESSION.md` and add a row to the Running Session Log with the date, commit marker, and a one-line summary of completed work. Omitting this breaks continuity for the next session.

---

## L003 — Do Not Push to Remote Without Explicit Human Confirmation

**Observed behavior**: Agent pushed commits to the remote repository without being explicitly asked to.

**Correct approach**: All commits stay local until the human explicitly says to push or confirms the work is ready for remote. Default is local-only.

---

## L004 — Do Not Modify Protected Files Without an Explicit Request

**Observed behavior**: Agent modified `.github/SKILLS.md`, `CONTRIBUTING.md`, `.github/LESSONS.md`, or other governance files as incidental cleanup or as part of an unrelated task.

**Correct approach**: Governance files require an explicit user instruction to change. Do not touch them speculatively or as a side effect of another task.

---

## Adding a New Lesson

When @maxjbarfuss observes a repeated agent mistake, add a new entry with the next sequential number:

    ## L<NNN> — <Short descriptive title>

    **Observed behavior**: <What the agent did wrong — be specific enough that another agent recognizes the pattern.>

    **Correct approach**: <What to do instead — be prescriptive, not advisory.>
