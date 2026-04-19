# WeChat Assistant Nanobot

A WeChat-based private knowledge base RAG assistant. Users can send PDF, Word, Markdown, TXT, and other documents in WeChat. The bot stores the files in a local workspace, parses them, splits them into chunks, and builds a SQLite retrieval index. After that, users can ask questions about the uploaded documents directly in WeChat.

This project is based on [HKUDS/nanobot](https://github.com/HKUDS/nanobot). nanobot provides the agent loop, model provider abstraction, tool-calling runtime, and WeChat channel. This project adds automatic WeChat file ingestion, a local knowledge base, and a `knowledge_search` retrieval tool.

## Features

- Automatically downloads WeChat file messages and sends them through the knowledge ingestion pipeline.
- Supports `pdf`, `docx`, `txt`, and `md`.
- Uses MinerU first for PDF parsing, with OCR enabled by default.
- Stores ingested data in the local workspace, without requiring a cloud vector database.
- Uses SQLite as a lightweight retrieval index, making deployment and debugging simple.
- Lets the agent call `knowledge_search` before answering questions about uploaded documents.
- Keeps nanobot's original provider configuration style, so it can work with DeepSeek, OpenAI, DashScope, and other OpenAI-compatible APIs.

## Workflow

```text
User sends a file in WeChat
  -> WeChat channel downloads the file
  -> AgentLoop detects ingestible files
  -> KnowledgeService parses the document
  -> KnowledgeChunker splits text into chunks
  -> KnowledgeStore saves raw files, parsed results, chunks, and SQLite index
  -> User asks a question
  -> Agent calls knowledge_search
  -> Model answers based on retrieved snippets
```

## Project Structure

```text
nanobot/knowledge/
  types.py          Knowledge base data structures
  parser.py         Document parsing entry point
  mineru_parser.py  MinerU PDF parser wrapper
  chunker.py        Text chunking
  store.py          Local files and SQLite index
  service.py        Ingestion and retrieval orchestration

nanobot/agent/tools/knowledge_search.py
  Local knowledge search tool used by the agent

nanobot/channels/weixin.py
  Downloads WeChat files and marks them as ingestible

nanobot/agent/loop.py
  Runs automatic file ingestion before handling user messages
```

## Requirements

- Python 3.11 or 3.12.
- A usable LLM API key.
- A MinerU API token is recommended for scanned PDFs.
- If the bot needs to run as a long-running WeChat assistant, the host machine should not sleep. For stable use, deploy it on a server.

## Installation

Install from source:

```bash
git clone https://github.com/zhaoxinyu2002/wechat-assistant-nanobot.git
cd wechat-assistant-nanobot
pip install -e ".[weixin]"
```

If you use conda, create or activate an environment first:

```bash
conda create -n nanobot python=3.11
conda activate nanobot
pip install -e ".[weixin]"
```

If you also want to run tests:

```bash
pip install -e ".[weixin,dev]"
```

## Initialize Configuration

Generate the default configuration:

```bash
nanobot onboard
```

Default config path:

```text
~/.nanobot/config.json
```

On Windows, it is usually:

```text
C:\Users\<your-username>\.nanobot\config.json
```

## Configure the Model

This project follows nanobot's provider configuration style. The example below uses the official DeepSeek API. Replace the placeholder with your own key.

```json
{
  "agents": {
    "defaults": {
      "workspace": "~/.nanobot/workspace",
      "model": "deepseek-reasoner",
      "provider": "deepseek",
      "maxTokens": 2048,
      "contextWindowTokens": 65536,
      "temperature": 0.1,
      "maxToolIterations": 40,
      "timezone": "Asia/Shanghai"
    }
  },
  "providers": {
    "deepseek": {
      "apiKey": "YOUR_DEEPSEEK_API_KEY",
      "apiBase": "https://api.deepseek.com"
    }
  }
}
```

For other OpenAI-compatible services, change the provider, model name, and `apiBase` as needed. Do not commit real API keys to GitHub.

## Configure the WeChat Channel

Enable the WeChat channel in `config.json`:

```json
{
  "channels": {
    "weixin": {
      "enabled": true,
      "allowFrom": ["*"]
    }
  }
}
```

For real use, replace `allowFrom` with your own WeChat user ID. `["*"]` allows all senders and is convenient for debugging, but it is not recommended for public or shared deployments.

Log in to the WeChat channel for the first time:

```bash
nanobot channels login weixin
```

Scan the QR code shown in the terminal. The login state is saved locally, so repeated QR login is usually not required.

## Configure the Knowledge Base

Add or update the `knowledge` section in `config.json`:

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
    "mineruBaseUrl": "https://mineru.net",
    "mineruApiToken": "YOUR_MINERU_TOKEN",
    "mineruIsOcr": true,
    "mineruTimeoutS": 300,
    "mineruPollIntervalS": 3
  }
}
```

Configuration notes:

- `enabled`: Enables or disables the knowledge base.
- `autoIngestFromMedia`: Automatically ingests files received from WeChat.
- `rawDir`: Stores original uploaded files.
- `parsedDir`: Stores parsed JSON results.
- `chunksDir`: Stores chunked JSONL files.
- `indexDir`: Stores the SQLite retrieval index.
- `parserPdf`: PDF parser. The default is `mineru`.
- `mineruMode`: `precision` uses the MinerU API; `command` uses a local command; `agent` uses the MinerU agent API.
- `mineruIsOcr`: Enables OCR. Keep this as `true` for better scanned-PDF support.

## Start the Bot

Start the gateway:

```bash
nanobot gateway
```

Common Windows + conda startup:

```powershell
conda activate nanobot
nanobot gateway
```

To keep it running in a separate PowerShell window, adjust the project path and run:

```powershell
Start-Process powershell -ArgumentList '-NoExit', '-Command', 'cd <project-path>; conda activate nanobot; nanobot gateway'
```

The bot only replies while the gateway process is running. Closing the terminal, exiting the conda environment, putting the computer to sleep, or losing network access will stop the bot from responding.

## Usage

1. Start `nanobot gateway`.
2. Send a PDF, docx, txt, or md file to the bot in WeChat.
3. Wait for ingestion to finish. The ingestion result is added to the current session context.
4. Ask questions about the uploaded file.

## Local Data

Default workspace:

```text
~/.nanobot/workspace
```

Knowledge base data is stored under the workspace:

```text
knowledge/raw       Original uploaded files
knowledge/parsed    Parsed results
knowledge/chunks    Chunked text snippets
knowledge/index     SQLite retrieval index
```

To back up the knowledge base, back up the whole workspace.

## Tests

This repository keeps the tests related to the knowledge-base extension:

```text
tests/knowledge/test_ingest_service.py
tests/tools/test_knowledge_search_tool.py
tests/agent/test_loop_knowledge_ingest.py
```

Run:

```bash
pytest tests/knowledge tests/tools/test_knowledge_search_tool.py tests/agent/test_loop_knowledge_ingest.py
```

## Open Source Notice

This project is not a chatbot framework written from scratch. It extends nanobot with a WeChat private knowledge base RAG pipeline. The original MIT License is preserved, and the upstream project is credited in this README.

## License

MIT License. See [LICENSE](LICENSE).
