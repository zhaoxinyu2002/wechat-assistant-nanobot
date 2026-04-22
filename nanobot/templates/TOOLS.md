# Tool Usage Notes

Tool signatures are provided automatically via function calling.
This file documents non-obvious constraints and usage patterns.

## exec — Safety Limits

- Commands have a configurable timeout (default 60s)
- Dangerous commands are blocked (rm -rf, format, dd, shutdown, etc.)
- Output is truncated at 10,000 characters
- `restrictToWorkspace` config can limit file access to the workspace

## cron — Scheduled Reminders

- Please refer to cron skill for usage.

## knowledge_search - Private Document Evidence

- Use `knowledge_search` before answering questions about uploaded files, PDFs, notes, reports, or private knowledge-base content.
- Prefer focused queries that include the user's key entities, terms, and requested answer type.
- If evidence is weak or missing, say that the local knowledge base did not contain enough support instead of guessing.
- When useful, mention the returned `source`, `page`, or `heading` in the answer.
