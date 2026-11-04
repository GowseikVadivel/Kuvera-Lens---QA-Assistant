# Kuvera – Lens

Kuvera – Lens is a facts-only mutual fund information assistant. Its ingestion stages render configured HDFC scheme pages with Playwright, load webpage/PDF text with source metadata, produce citation-aware chunks, generate normalized embeddings, and publish versioned generations to Qdrant. Retrieval uses scheme-filtered Qdrant search and generates concise answers grounded in validated official evidence.

## Requirements

- Python 3.10 or newer
- pip

## Local setup

From the project directory, create and activate a virtual environment, then install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the Streamlit app:

```powershell
python -m streamlit run app.py
```

Run the tests:

```powershell
python -m unittest discover -s tests -v
```

## Configuration

For local development, add the Qdrant Cloud connection values and Groq API key to the root `.env` file:

```dotenv
QDRANT_URL=https://your-cluster-url
QDRANT_API_KEY=your-database-api-key
GROQ_API_KEY=your-groq-api-key
```

The application loads this file automatically; values already set in the process environment take precedence. `.env` is ignored by Git, so do not force-add it or share its contents. Local development uses a separate persistent Qdrant instance under `data/qdrant` when neither Qdrant variable is set. Production uses Qdrant Cloud: configure `QDRANT_URL` and `QDRANT_API_KEY` as secrets in both GitHub Actions and Streamlit deployment settings so ingestion and retrieval use the same service and active collection alias, `kuvera_lens_active`. Set `GROQ_API_KEY` in the Streamlit deployment environment for answer generation. If only one Qdrant variable is set, the app fails clearly rather than falling back to local storage. Never commit API keys or deployment secrets.

## Use the Streamlit assistant

Start the app with `python -m streamlit run app.py`. Select one of the four HDFC schemes and start a conversation. Each question and suggested follow-up uses the existing scheme-filtered retrieval and grounded answer service; conversation history is temporary session state and is sent only as bounded, untrusted context to help interpret follow-ups. A scheme change or new conversation clears that history. Answers show only citations returned by the backend, including the official source link, document title, and PDF page when available. Insufficient evidence displays the exact refusal; unavailable services produce an explicit error instead of a fabricated answer. If Groq rate-limits a request, the UI reports the provider-supplied retry window without exposing provider response details.

## Scrape scheme pages and discover official documents

Add only the exact official URLs to `config/source_links.json`:

```json
{
  "sources": [
    {
      "source_id": "large-cap-scheme",
      "scheme_id": "hdfc_large_cap_direct_growth",
      "organization": "hdfc_amc",
      "url": "https://www.hdfcfund.com/"
    }
  ]
}
```

Supported `organization` values are `hdfc_amc`, `sebi`, and `amfi`. The configured HDFC scheme pages are rendered with Playwright. The scraper extracts rendered text and discovers linked official PDFs, accepting only allowlisted HDFC hosts and scheme-specific SID, KIM, or factsheet documents. It saves raw files and metadata under `data/raw` for the document-loading phase.

Run it with:

```powershell
python -m kuvera_lens.scraper
```

Load the downloaded HTML/PDF files and their metadata with:

```powershell
python -m kuvera_lens.loaders
```

The loader verifies source metadata and content hashes, extracts HTML text or PDF page text, and fails explicitly if any manifest or document is invalid. For approved HDFC scheme pages, it selects the newest capture containing both a TER value and riskometer level, so a partial dynamic-page snapshot cannot replace a more complete capture. KIM and SID PDF types remain distinct. Scraped metadata includes scheme, title, type, source URL, publication date when found, last-checked time, and content hash. PDF extraction retains page numbers.

## Chunk loaded documents

Run the chunking phase after scraping/loading:

```powershell
python -m kuvera_lens.chunking
```

By default, the chunker reads verified documents from `data/raw` and writes versioned JSONL chunk files under `data/chunks`, plus an atomically published `active_chunks.json` manifest. Downstream phases should read only files referenced by that manifest; older document versions remain on disk and are not active. Chunks are sentence-aware, bounded to 1,200 characters with 160 characters of overlap, and retain scheme, document, URL, publication, version, section, and PDF page metadata. Same-content documents for a scheme are deduplicated, preferring a direct PDF citation URL. Unchanged files are not rewritten.

Chunk size and overlap can be changed explicitly:

```powershell
python -m kuvera_lens.chunking --max-chars 1200 --overlap-chars 160
```

This phase does not create embeddings or write to Qdrant; those are subsequent phases.

## Embed active chunks

After chunking, generate normalized `sentence-transformers/all-MiniLM-L6-v2` embeddings:

```powershell
python -m kuvera_lens.embedding
```

The embedding stage reads only `data/chunks/active_chunks.json`, writes versioned NumPy vector files under `data/embeddings/files`, and atomically publishes `data/embeddings/active_embeddings.json`. Each vector row is aligned with the corresponding chunk ID; source and citation metadata remain in the active chunk files referenced by the embedding manifest. Unchanged chunk sets are not re-embedded, and failed runs do not replace the active embedding manifest. The model weights may need to be downloaded on the first run.

Override the paths or batch size when needed:

```powershell
python -m kuvera_lens.embedding --chunks data/chunks/active_chunks.json --output data/embeddings --batch-size 32
```

This phase does not write to Qdrant or implement retrieval.

## Store active chunks in Qdrant

After generating embeddings, configure `QDRANT_URL` and `QDRANT_API_KEY`, then run:

```powershell
python -m kuvera_lens.ingestion
```

Without Qdrant Cloud environment variables, this command uses the separate local database at `data/qdrant`. With both variables set, it connects to the configured Qdrant Cloud instance. The indexer validates active chunks and embeddings, creates the keyword payload index required for `scheme_id` filters, stores text, precomputed vectors, and citation metadata in a versioned collection, verifies the complete collection, then atomically switches the `kuvera_lens_active` alias. Unchanged active generations also ensure this index exists before returning. A failed stage leaves the previous alias target active. Identical inputs are reused without duplicate points; old collections are retained for rollback. Configure the same Cloud URL and API key as secrets for both GitHub Actions and Streamlit.

Optional paths and batch size:

```powershell
python -m kuvera_lens.ingestion --chunks data/chunks/active_chunks.json --embeddings data/embeddings/active_embeddings.json --batch-size 100
```

## Retrieve evidence and generate grounded answers

`kuvera_lens.retrieval.retrieve_evidence` embeds a question with the existing Sentence Transformers model and searches the active Qdrant alias, filtering to the selected supported scheme. Known fact terms are mapped to the source vocabulary used by HDFC pages and SIDs (for example, “expense ratio” to “TER”) for retrieval only; the original question still goes to the answer pipeline. Its optional `source_type` argument accepts `"webpage"`, `"sid_pdf"`, or `"both"` (the default, preserving existing behavior). SID-only retrieval excludes KIM documents. Retrieval rejects unapproved sources and incomplete or cross-scheme provenance, and returns at most five results with cosine similarity of at least `0.35`, except for exact TER or riskometer label/value matches on the approved scheme webpage and SID asset-allocation tables. Exact field matches are prioritized over less specific results. If no valid evidence remains, it returns exactly `I couldn't verify this information from the available official sources.` without calling Groq. `answer_question` accepts the same optional argument and forwards it to retrieval.

`kuvera_lens.answering.answer_question` uses only retrieved evidence, limits generated answers to three sentences, and constructs citations from indexed source metadata (including page numbers when available). Invalid or uncited model output is replaced with the exact refusal. The initial `0.35` similarity threshold is configurable per retrieval call and should be calibrated against a reviewed question set before production use.

The chat follow-up buttons are selected dynamically for the active scheme from one scheme-filtered scan of indexed chunks. A question is suggested only when its required fact pattern appears in an approved webpage or SID chunk. Each suggestion retains its source type, and selecting it sends that question through fresh source- and scheme-filtered retrieval and the existing answer pipeline.

To import the four user-provided November 21, 2025 SIDs, place the PDFs in one directory with their expected names and run:

```powershell
python -m kuvera_lens.import_sids --input C:\path\to\sid-pdfs
python -m kuvera_lens.loaders
```

The importer verifies each PDF's scheme name and stated SID date before copying it into `data/raw`; it then writes metadata sidecars consumed by the loader. It does not retrieve the documents from the network.
