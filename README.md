# WeChat Private Knowledge Base RAG Assistant

A WeChat-oriented private knowledge base assistant built on top of [HKUDS/nanobot](https://github.com/HKUDS/nanobot).

This project keeps nanobot's lightweight agent architecture, channel system, provider abstraction, and tool-calling loop, then adds a local document ingestion and retrieval layer for private-data question answering.

## Method

```text
WeChat file intake
-> MinerU-first document parsing
-> local chunk storage
-> SQLite retrieval index
-> knowledge_search agent tool
-> retrieval-augmented answers with source metadata
```

The goal is not to build another generic chatbot. The goal is to let a user send documents through WeChat and later ask questions grounded in those uploaded files.

## What Changed From nanobot

This fork adds a private knowledge base pipeline around nanobot:

- WeChat file messages are saved locally and marked as ingestion candidates.
- Uploaded documents are parsed, normalized, chunked, and indexed.
- PDF parsing prefers MinerU and keeps a basic fallback path.
- Local storage is organized under `knowledge/raw`, `knowledge/parsed`, `knowledge/chunks`, and `knowledge/index`.
- A new `knowledge_search` tool lets the agent retrieve evidence before answering.
- The agent loop can automatically ingest newly uploaded files before processing the current turn.
- Knowledge-base behavior is configurable from the normal nanobot config file.

## Architecture

```text
User sends a file in WeChat
        |
        v
WeChat channel downloads the file
        |
        v
Agent receives media paths and ingest metadata
        |
        v
KnowledgeService parses and chunks the document
        |
        v
KnowledgeStore writes raw files, parsed JSON, chunks, and SQLite index
        |
        v
User asks a question
        |
        v
Agent calls knowledge_search
        |
        v
Model answers with retrieved evidence and source metadata
```

## Main Modules

```text
nanobot/knowledge/
  types.py          normalized document, section, chunk, and search result types
  parser.py         document parsing entrypoint
  mineru_parser.py  MinerU-backed PDF parsing wrapper
  chunker.py        chunking logic
  store.py          raw/parsed/chunk persistence and SQLite retrieval
  service.py        ingestion and search orchestration

nanobot/agent/tools/knowledge_search.py
  Agent tool for querying the local knowledge base.

nanobot/channels/weixin.py
  Marks downloaded WeChat files as knowledge ingestion candidates.

nanobot/agent/loop.py
  Runs automatic ingestion before building the current agent turn.
```

## Supported Documents

Current MVP support:

- `txt`
- `md`
- `docx`
- `pdf`

PDF parsing uses MinerU first. If MinerU is unavailable or fails, the parser falls back to a basic text path when possible.

## Knowledge Configuration

The project adds a top-level `knowledge` section to the normal nanobot config.

Example:

```json
{
  "knowledge": {
    "enabled": true,
    "autoIngestFromMedia": true,
    "rawDir": "knowledge/raw",
    "parsedDir": "knowledge/parsed",
    "chunksDir": "knowledge/chunks",
    "indexDir": "knowledge/index",
    "maxChunkChars": 1200,
    "chunkOverlap": 150,
    "parserPdf": "mineru",
    "mineruMode": "precision",
    "mineruApiToken": "",
    "mineruIsOcr": true
  }
}
```

Model and provider configuration follow the upstream nanobot configuration style. See the original nanobot provider documentation for API key and model setup.

## Usage

1. Configure nanobot as usual.
2. Enable the WeChat channel.
3. Configure the `knowledge` section if PDF/MinerU parsing is needed.
4. Start the nanobot gateway.
5. Send a supported document through WeChat.
6. Ask questions about the uploaded document.

For long-running usage, keep the gateway process alive. If the host machine sleeps, the bot will stop receiving and answering messages.

## Notes

This is an MVP implementation focused on a local single-user knowledge base. It intentionally uses local files and SQLite instead of a full vector database to keep deployment simple and easy to inspect.

Possible next steps:

- Add vector or hybrid retrieval.
- Improve source citation formatting.
- Add multi-user knowledge isolation.
- Add explicit knowledge management commands.
- Improve OCR status reporting and ingestion failure feedback.

## Credits

This project is based on [HKUDS/nanobot](https://github.com/HKUDS/nanobot). nanobot provides the lightweight agent runtime, provider abstraction, channel integrations, and tool-calling framework. This fork adds the WeChat private knowledge base RAG pipeline.

## License

This project follows the original MIT License. See [LICENSE](LICENSE).
