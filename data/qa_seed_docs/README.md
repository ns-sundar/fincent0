# Fincent Q&A Seed Documents

This folder contains a starter corpus for a future vector database powering the Q&A agent.

- Document count: **10**
- Scope: foundational financial concepts and common educational questions
- Exclusions: real-time market prices, live quotes, and trading signals
- Format: one Markdown file per document, plus `index.jsonl` metadata manifest

## Suggested ingestion flow

1. Read each `doc_*.md` file.
2. Strip YAML frontmatter (optional) and split body into chunks.
3. Generate embeddings for each chunk.
4. Store vectors with metadata from the frontmatter and `index.jsonl`.

## Notes

- Content is curated and paraphrased educational material.
- Source links are included for provenance and future refreshes.
