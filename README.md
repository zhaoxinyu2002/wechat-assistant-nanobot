# WeChat Assistant Nanobot

A WeChat-based private knowledge-base Agentic RAG assistant. Users can upload PDF, Word, Markdown, and TXT documents in WeChat; the bot ingests them into a local knowledge base and exposes retrieval as a `knowledge_search` tool that the agent can call before answering document-grounded questions.

This project extends [HKUDS/nanobot](https://github.com/HKUDS/nanobot). Upstream nanobot provides the agent loop, model provider abstraction, tool-calling runtime, and chat-channel framework. This project adds automatic WeChat file ingestion, a local RAG knowledge pipeline, source-aware retrieval results, and reproducible retrieval benchmarks.

## Highlights

- Tool-driven Agentic RAG: document retrieval is registered as an agent tool instead of being a fixed pre-answer step.
- Automatic WeChat file ingestion for `pdf`, `docx`, `txt`, and `md` uploads.
- MinerU-based PDF parsing with OCR support for scanned or structured PDFs.
- Local-first knowledge storage with raw files, parsed JSON, chunk JSONL, SQLite metadata, and optional FAISS vector index.
- Multiple retrieval backends: hashing fallback, OpenAI-compatible embeddings, local BGE-M3 embeddings, hybrid retrieval, and optional BGE reranking.
- Source-aware evidence snippets returned with `source`, `page`, `heading`, and `score`.
- Reproducible retrieval evaluation on CMRC2018 Chinese QA and UDA-Benchmark paper evidence retrieval.

## Benchmark Highlights

| Dataset | Scale | Best setting | Recall@1 | MRR |
|---|---:|---|---:|---:|
| CMRC2018 dev subset | 150 documents / 300 queries | BGE-M3 + reranker | 0.9767 | 0.9850 |
| UDA paper evidence subset | 150 documents / 300 queries | BGE-M3 hybrid + reranker | 0.4133 | 0.4601 |

Compared with the hashing hybrid baseline, BGE-M3 plus reranking improves CMRC2018 Recall@1 from 41.00% to 97.67% and MRR from 0.5300 to 0.9850. On UDA paper evidence retrieval, the best reproduced setting improves Recall@1 from 4.00% to 41.33% and MRR from 0.0745 to 0.4601. See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for the full tables.

## Project Structure

```text
nanobot/knowledge/
  types.py          Knowledge base data structures
  parser.py         Document parsing entry point
  mineru_parser.py  MinerU PDF parser wrapper
  chunker.py        Text chunking
  embeddings.py     Hashing, OpenAI-compatible, and BGE-M3 embedding backends
  reranker.py       Optional cross-encoder reranker
  store.py          Local files and SQLite index
  service.py        Ingestion and retrieval orchestration
  eval.py           Retrieval evaluation utilities

nanobot/agent/tools/knowledge_search.py
  Source-aware local knowledge search tool used by the agent

nanobot/channels/weixin.py
  Downloads WeChat files and marks them as ingestible

nanobot/agent/loop.py
  Runs automatic file ingestion before handling user messages

eval/
  Reproducible benchmark scripts for CMRC2018 and UDA evidence retrieval

benchmark_results/
  Saved benchmark tables and JSON result files
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
    "maxFileBytes": 31457280,
    "maxChunksPerFile": 1000,
    "maxChunkChars": 1200,
    "chunkOverlap": 150,
    "chunkStrategy": "recursive",
    "chunkIncludeMetadata": true,
    "embeddingProvider": "hashing",
    "embeddingModel": "",
    "embeddingApiKey": "",
    "embeddingBaseUrl": "",
    "embeddingDim": 384,
    "embeddingBatchSize": 64,
    "vectorIndex": "faiss",
    "retrievalMode": "hybrid",
    "keywordWeight": 0.35,
    "vectorWeight": 0.65,
    "rerankerProvider": "none",
    "rerankerModel": "",
    "rerankerTopK": 20,
    "parserPdf": "mineru",
    "mineruMode": "precision",
    "mineruBaseUrl": "https://mineru.net",
    "mineruApiToken": "YOUR_MINERU_TOKEN",
    "mineruIsOcr": true,
    "mineruTimeoutS": 3600,
    "mineruPollIntervalS": 10
  }
}
```

Configuration notes:

- `enabled`: Enables or disables the knowledge base.
- `autoIngestFromMedia`: Automatically ingests files received from WeChat.
- `rawDir`: Stores original uploaded files.
- `parsedDir`: Stores parsed JSON results.
- `chunksDir`: Stores chunked JSONL files.
- `indexDir`: Stores the SQLite chunk/embedding index and optional FAISS vector index.
- `maxFileBytes`: Safety cap for automatic ingestion. Larger files are skipped with a clear error instead of running an expensive parse/embedding job in the message loop.
- `maxChunksPerFile`: Safety cap after parsing/chunking. Split very large documents or raise this intentionally if needed.
- `chunkStrategy`: Chunking strategy. `recursive` splits long sections with overlap, `section` keeps parsed sections intact, and `page` groups parsed sections by page.
- `chunkIncludeMetadata`: Prepends headings to chunk text when available, which helps retrieval and citations.
- `embeddingProvider`: Embedding backend. The built-in default is `hashing`, which is local and dependency-free. Use `openai` for OpenAI-compatible embeddings or `bge-m3` / `sentence-transformers` for local BGE embeddings.
- `embeddingModel`: Embedding model name, for example `text-embedding-3-small` or `BAAI/bge-m3`.
- `embeddingApiKey` / `embeddingBaseUrl`: Optional OpenAI-compatible embedding credentials. `OPENAI_API_KEY` is also supported.
- `vectorIndex`: Vector index backend. `faiss` is used when `faiss-cpu` is installed; otherwise vector search falls back to SQLite-stored embeddings.
- `retrievalMode`: Default retrieval mode for `knowledge_search`: `hybrid`, `vector`, or `keyword`.
- `keywordWeight` / `vectorWeight`: Score weights used by hybrid retrieval.
- `rerankerProvider`: Optional reranker. Use `bge-reranker` or `sentence-transformers` to rerank the top retrieval candidates.
- `rerankerModel`: Reranker model name, for example `BAAI/bge-reranker-base`.
- `parserPdf`: PDF parser. The default is `mineru`.
- `mineruMode`: `precision` uses the MinerU API; `command` uses a local command; `agent` uses the MinerU agent API.
- `mineruIsOcr`: Enables OCR. Keep this as `true` for better scanned-PDF support.

Optional FAISS acceleration can be installed with:

```bash
pip install ".[rag]"
```

Optional local embedding and reranker models can be installed with:

```bash
pip install ".[rag-ml]"
```

Example OpenAI-compatible embedding configuration:

```json
{
  "knowledge": {
    "embeddingProvider": "openai",
    "embeddingModel": "text-embedding-3-small",
    "embeddingApiKey": "YOUR_OPENAI_API_KEY",
    "embeddingDim": 1536
  }
}
```

Example local BGE embedding + reranker configuration:

```json
{
  "knowledge": {
    "embeddingProvider": "bge-m3",
    "embeddingModel": "BAAI/bge-m3",
    "embeddingDim": 1024,
    "rerankerProvider": "bge-reranker",
    "rerankerModel": "BAAI/bge-reranker-base",
    "rerankerTopK": 20
  }
}
```

## Evaluate Retrieval

Create a JSONL evaluation file where `relevant` values match expected source names, chunk IDs, headings, or text snippets:

```jsonl
{"query":"What does the guide say about citations?","relevant":["citations"]}
{"query":"How is hybrid retrieval configured?","relevant":["hybrid retrieval"]}
```

Run retrieval evaluation:

```bash
conda activate nanobot
nanobot knowledge eval eval.jsonl --k 1,3,5
```

The command reports `Recall@K`, `HitRate@K`, and `MRR`, which can be used to compare chunk sizes, overlap, embedding models, hybrid weights, and reranker settings.

The repository also includes reproducible benchmark scripts:

```bash
conda activate nanobot
python eval/run_cmrc_eval.py
python eval/run_uda_evidence_eval.py
```

Saved benchmark summaries are available in [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).

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
knowledge/index     SQLite chunk/embedding index and optional FAISS files
```

To back up the knowledge base, back up the whole workspace.

## Tests

This repository keeps the tests related to the knowledge-base extension:

```text
tests/knowledge/test_ingest_service.py
tests/tools/test_knowledge_search_tool.py
tests/agent/test_loop_knowledge_ingest.py
tests/channels/test_weixin_config.py
```

Run:

```bash
conda activate nanobot
pytest tests/knowledge tests/tools/test_knowledge_search_tool.py tests/agent/test_loop_knowledge_ingest.py tests/channels/test_weixin_config.py
```

## Open Source Notice

This project is not a chatbot framework written from scratch. It extends nanobot with a WeChat private knowledge base RAG pipeline. The original MIT License is preserved, and the upstream project is credited in this README.

## License

MIT License. See [LICENSE](LICENSE).
