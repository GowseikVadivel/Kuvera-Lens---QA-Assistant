# Kuvera – Lens
## Implementation Guide

**Architecture:** [architecture.md](./architecture.md)  
**Requirements:** [PRD.md](./PRD.md)

## Goal

Build a Streamlit QA assistant for four HDFC mutual fund schemes. It must answer only from approved official sources, cite its evidence, keep scheme data separate, and refuse when it cannot verify an answer.

## Phase 1 — Create the project structure

Create a small Python project with clear locations for the UI, ingestion, shared logic, and tests:

```text
kuvera-lens/
├── app.py                  # Streamlit UI
├── kuvera_lens/
│   ├── config.py           # Environment and secret configuration
│   ├── schemes.py          # Supported schemes and stable IDs
│   ├── sources.py          # Approved source registry and URL checks
│   ├── loaders.py          # Load and extract HTML/PDF documents
│   ├── ingestion.py        # Chunk, embed, and index documents
│   ├── retrieval.py        # Search and evidence validation
│   └── answering.py        # Grounded answer and refusal logic
├── tests/
├── requirements.txt
└── README.md
```

Keep API keys in environment variables or deployment secrets, never in source code. Add the required libraries from the PRD: Streamlit, Requests, BeautifulSoup, PyMuPDF, sentence-transformers, Qdrant client, and the Groq client.

**Done when:** The app starts locally, configuration errors are clear, and the test runner works.

## Phase 2 — Scrape official scheme pages and discover documents

1. Define canonical IDs for the four supported schemes and configure their official HDFC scheme pages in `config/source_links.json`.
2. Render those pages with Playwright and extract the rendered page text and title.
3. Discover linked PDFs, but accept only HTTPS URLs on the explicit HDFC official-host allowlist.
4. Classify only SID, KIM, and factsheet documents that also match the selected scheme; skip generic AMC material.
5. Validate HTTP status, final host, content type, size, and PDF signature. Do not bypass blocked URLs or access controls.
6. Save page/PDF content and source metadata under `data/raw`. Reuse content hashes to avoid reprocessing unchanged documents.

**Done when:** All four configured pages can be rendered, meaningful text is extracted, relevant official PDFs are stored with provenance, and failed URLs are reported.

## Phase 3 — Load and extract documents

1. Load approved webpages and PDFs from the validated document list.
2. Extract webpage content with BeautifulSoup and PDF text with PyMuPDF.
3. Capture document metadata and retain PDF page numbers or webpage sections where available.
4. Report fetch, load, and extraction failures explicitly. Do not treat empty or malformed content as successfully loaded.
5. Compute a content hash or use a reliable document version to identify unchanged and changed documents.
6. Pass extracted documents to the next phase as text plus provenance metadata.

**Done when:** Each supported document can be loaded into a consistent representation of text and metadata, and extraction failures are visible.

## Phase 4 — Clean and chunk documents

Implemented by `python -m kuvera_lens.chunking`:

1. Normalize loaded text while preserving headings, numbers, units, and relevant context.
2. Split text at sentence boundaries, falling back to word boundaries for oversized sentences. Default chunks are at most 1,200 characters with 160 characters of overlap.
3. Keep PDF pages separate and attach metadata to every chunk:
   - Stable chunk and source IDs.
   - Canonical scheme ID.
   - Official source URL and domain.
   - Document title and version or content hash.
   - PDF page number or webpage section, when available.
4. Do not create chunks from empty text, unverified sources, or documents without a clear scheme association. Deduplicate identical content within a scheme and prefer direct PDF citation URLs.
5. Write versioned JSONL files under `data/chunks` and atomically publish `active_chunks.json`. Downstream phases must read only files listed in the active manifest; older versions are preserved and unchanged files are not rewritten.

**Done when:** Each chunk is bounded, meaningful, deterministic, and retains the provenance needed to verify and cite it. Embedding and vector storage are not part of this phase.

## Phase 5 — Generate chunk embeddings

Implemented by `python -m kuvera_lens.embedding`:

1. Read only the chunk files referenced by `data/chunks/active_chunks.json` and validate chunk IDs, source metadata, counts, and paths.
2. Generate normalized float32 vectors in batches with `sentence-transformers/all-MiniLM-L6-v2`.
3. Store vectors in versioned NumPy files under `data/embeddings/files`, preserving row order alongside the corresponding chunk IDs.
4. Publish `data/embeddings/active_embeddings.json` atomically. The manifest references the active chunk file and content hash, model profile, vector dimensions, vector file checksum, and chunk count; full citation metadata remains in the chunk records.
5. Reuse unchanged, validated vector files. Do not overwrite the active manifest if embedding or publication fails. The model weights may be downloaded on the first run.

Run this phase after chunking:

```powershell
python -m kuvera_lens.embedding
```

**Done when:** Every active chunk has one normalized vector associated with its stable chunk ID, and a failed run leaves the previous embedding manifest active. This phase does not write to Qdrant.

## Phase 6 — Store active chunks in Qdrant

Implemented by `python -m kuvera_lens.ingestion`:

1. Connect to Qdrant Cloud using `QDRANT_URL` and `QDRANT_API_KEY`. Configure the same endpoint and key as GitHub Actions and Streamlit secrets so ingestion and retrieval access the same collection.
2. Read only the active chunk and embedding manifests. Validate that every active chunk has exactly one aligned, normalized vector and complete source/scheme citation metadata.
3. Store text, precomputed vectors, and scalar metadata in a versioned Qdrant collection using deterministic UUID point IDs derived from stable chunk IDs. Retain the original chunk ID and all citation metadata in each payload.
4. Build and verify a new generation, then atomically publish it by switching the shared `kuvera_lens_active` collection alias. A staging write or verification failure leaves the previous alias target active.
5. Reuse an identical verified generation without duplicate writes. Changed/retired content is absent from the newly active collection; older collections are retained for rollback.
6. When neither Qdrant environment variable is set, use the local persistent database at `data/qdrant` for development only. Partial Cloud configuration is an error; local data is never substituted for production Cloud.

Configure the Qdrant Cloud environment variables for the target database, then run:

```powershell
python -m kuvera_lens.ingestion
```

**Done when:** All active chunks and their embeddings are present with citation metadata in the shared Qdrant collection, the `scheme_id` keyword payload index supports filtered search, and the active alias changes only after the new generation is verified.

## Phase 7 — Implement retrieval and grounded answers

For each question:

1. Validate that the selected scheme is one of the four supported schemes.
2. Embed the question and search the Qdrant `kuvera_lens_active` alias using the selected scheme ID.
3. Exclude evidence from unapproved sources, other schemes, or records missing provenance.
4. Require a minimum cosine similarity of 0.35 and check that the remaining evidence has valid citation metadata. Treat this as an initial threshold to calibrate against reviewed questions.
5. If evidence is insufficient, return exactly:

   > I couldn't verify this information from the available official sources.

   Do not call the LLM in this case.
6. Otherwise, send the question and validated evidence to Groq. Instruct it to use only that evidence, avoid advice and predictions, and answer in no more than three sentences.
7. Build citations from retrieved metadata, not URLs generated by the model. Include title and page number when available.

**Done when:** Every request returns a cited evidence-based answer, the exact refusal, or a clear service error—never a guessed answer.

The retrieval and answer functions are available as `kuvera_lens.retrieval.retrieve_evidence` and `kuvera_lens.answering.answer_question`. Focused tests cover scheme filtering, source/provenance validation, refusal behavior, citation validation, and the three-sentence limit.

## Phase 8 — Build the Streamlit UI

1. Show the four supported schemes and require a selection before accepting a question.
2. Provide a conversational question input and preserve follow-up context for the current session.
3. Retrieve evidence again for every factual follow-up; conversation history is context, not evidence.
4. Display concise answers with clickable official citations and page details when available.
5. Show verification refusals separately from temporary errors such as an unavailable vector database or LLM.
6. When the scheme changes, clear or isolate prior conversation context so answers and citations cannot cross schemes.
7. Do not request or store PAN, Aadhaar, OTPs, bank details, credentials, or unnecessary personal information.

**Done when:** Users can ask multiple questions about a selected scheme and verify the answers from the displayed sources.

## Phase 9 — Edge cases and guardrail tests

| Test case | Expected behavior |
|---|---|
| No scheme selected or unsupported scheme ID | Do not retrieve; ask the user to select a supported scheme |
| Question has no relevant evidence | Return the exact verification refusal; do not call Groq |
| Retrieved chunk belongs to a different scheme | Discard it; refuse if no valid evidence remains |
| Source domain is not allowlisted | Do not fetch or index it |
| Approved page links to an unapproved domain | Reject the discovered link |
| Approved URL redirects to an unapproved domain | Stop before fetching the redirected destination |
| Source fetch fails or returns empty/malformed content | Record an ingestion failure; do not mark it as successfully indexed |
| PDF extraction fails or has no usable text | Report the failure; do not create empty chunks |
| Document has no citation or scheme provenance | Exclude it from factual answers |
| Same document is ingested again unchanged | Do not create duplicate chunks |
| Document content changes | Update the indexed version and avoid serving stale chunks as current |
| Question asks for a buy/sell/redeem recommendation | Do not recommend; state the product is informational only |
| Question asks for returns or a prediction | Do not predict; refuse unsupported information |
| Question asks for personalized portfolio advice | Do not provide personalized advice |
| Prompt or source text asks the model to ignore rules or reveal secrets | Treat it as untrusted text; do not follow the instruction |
| Model generates unsupported numbers or an unapproved citation | Reject the output or return the verification refusal |
| Model returns more than three sentences | Reject or safely shorten only if meaning and evidence remain intact; otherwise refuse |
| Groq, embedding model, or Qdrant is unavailable | Show a clear service error; never guess or return a success-shaped fallback |
| User switches schemes mid-conversation | Do not reuse prior scheme evidence or citations |
| Sensitive personal information is entered accidentally | Do not request it or persist it in logs or application storage |

### Minimum release test set

Create reviewed questions for each supported category: scheme details, investment requirements, charges, ELSS lock-in/tax facts, and documents/statements. Include:

- Questions with known official evidence and expected citations.
- Questions for facts absent from the source corpus that must be refused.
- Similar questions for different schemes to test scheme isolation.
- Follow-up questions that depend on conversation context but still require fresh retrieval.
- Out-of-scope requests for recommendations, return predictions, and personalized advice.

Run these checks before release and whenever retrieval, prompting, chunking, or source ingestion changes.

## Phase 10 — Schedule ingestion and deploy

1. Configure GitHub Actions to run ingestion every 24 hours and report failures.
2. Configure Streamlit Community Cloud with the same `QDRANT_URL` and `QDRANT_API_KEY` as the ingestion workflow.
3. Store Groq and persistence credentials in deployment secrets.
4. Confirm that daily ingestion processes only new or changed documents and that updates are visible to the app.
5. Document how to bootstrap the index and retry a failed ingestion run.

**Done when:** The deployed app can answer from the shared index, daily refresh status is visible, and no secret is exposed in source code, logs, or the UI.
