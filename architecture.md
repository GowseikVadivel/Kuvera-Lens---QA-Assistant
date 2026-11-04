# Kuvera – Lens
## System Architecture

**Status:** Initial release  
**Source of requirements:** [PRD.md](./PRD.md)

## 1. Purpose and scope

This document describes the proposed architecture for the Kuvera – Lens initial release: a Streamlit application that answers factual questions about four supported HDFC mutual fund schemes using only retrieved evidence from approved official sources.

The architecture covers the user-facing question and answer flow, scheduled source ingestion, retrieval and answer generation, data boundaries, operational concerns, and the primary failure behaviors. It does not expand the product beyond the scope in the PRD.

## 2. Architecture principles

1. **Evidence before generation:** The language model may answer only from evidence retrieved for the current question.
2. **Scheme isolation:** Every indexed chunk and retrieval request is associated with a scheme. Retrieval is filtered to the selected scheme.
3. **Approved sources only:** A source must pass domain and URL validation before its content can enter the knowledge base.
4. **Fail closed:** Missing, weak, or invalid evidence results in the required refusal, not an ungrounded answer.
5. **Traceable answers:** Retrieved evidence retains enough metadata to provide an official link, title, and page number where available.
6. **Minimal personal data:** The application does not request or need user identity, financial-account credentials, or personal investment information.

## 3. High-level architecture

```text
                                      +------------------------------+
                                      | Approved official sources    |
                                      | HDFC AMC | SEBI | AMFI       |
                                      +---------------+--------------+
                                                      |
                                           scheduled source refresh
                                                      |
                                      +---------------v--------------+
                                      | Ingestion pipeline           |
                                      | URL/domain validation        |
                                      | HTML and PDF extraction      |
                                      | normalization, chunking      |
                                      | change detection, metadata   |
                                      +---------------+--------------+
                                                      |
                                      +---------------v--------------+
                                      | Qdrant knowledge base        |
                                      | embeddings + chunks + source |
                                      | and scheme metadata           |
                                      +---------------+--------------+
                                                      |
  +-------------------+     question  |               | evidence
  | User              +-------------->| Streamlit app +---------------+
  | scheme + chat UI  |               | retrieval and validation     |
  +-------------------+<--------------+               |              |
                           answer + citations          |              |
                                                      v              |
                                            +-------------------+     |
                                            | Groq LLM          |<----+
                                            | grounded answer   |
                                            +-------------------+
```

The Streamlit app handles the synchronous user interaction. A scheduled GitHub Actions workflow runs ingestion separately; it should not run as part of a user's chat request.

## 4. Components and responsibilities

### 4.1 Streamlit user interface

- Presents the four supported schemes and requires a selection before accepting a question.
- Maintains the current conversation in Streamlit session state.
- Sends the selected scheme, question, and relevant conversation context to the application service.
- Displays the answer, citations, and refusal message.
- Does not ask for or persist sensitive personal or account information.

Conversation state is for user experience only. It must not be treated as factual evidence or as a substitute for retrieval.

### 4.2 Application and answer service

Coordinates the per-question workflow:

1. Validate that the selected scheme is supported and the question is non-empty.
2. Build the retrieval query using the current question and only the limited context needed to resolve follow-ups.
3. Request evidence filtered to the selected scheme.
4. Validate that evidence is relevant, from an approved source, and associated with the selected scheme.
5. Return the required refusal if evidence is missing or fails validation.
6. Otherwise, request a concise answer from Groq using only the validated evidence.
7. Check the generated response against output constraints and return citations derived from retrieved source metadata.

The service owns the grounding and refusal policy; the UI must not independently invent or alter factual content.

### 4.3 Retrieval service

- Embeds the user's question with `sentence-transformers/all-MiniLM-L6-v2`.
- Queries the Qdrant active collection for semantically relevant chunks.
- Applies a mandatory selected-scheme filter and excludes unapproved or inactive source records.
- Returns the evidence text together with document identity, source URL, title, scheme, and page/section metadata where available.

Retrieval results are candidates, not automatically trusted answers. The answer service performs validation before generation.

### 4.4 Qdrant knowledge base

Stores text chunks, embeddings, and metadata produced by ingestion. Each record should include at minimum:

| Field | Purpose |
|---|---|
| `chunk_id` | Stable identity for an indexed chunk |
| `scheme_id` | Canonical identifier used for scheme filtering |
| `source_id` | Identity of the source document or webpage |
| `source_url` | Clickable official URL |
| `source_domain` | Domain used for allowlist validation |
| `document_title` | Human-readable citation title |
| `document_version` | Version, publication date, or content fingerprint |
| `page_number` | PDF page when available |
| `section` | Heading or section label when available |
| `content_hash` | Supports change detection and idempotent indexing |
| `retrieved_at` | Time content was fetched |
| `text` | Cleaned chunk content |

The active Qdrant collection contains all four schemes and every retrieval must apply a mandatory `scheme_id` filter.

The index publisher targets Qdrant Cloud when `QDRANT_URL` and `QDRANT_API_KEY` are set. It writes each complete index to a versioned collection, verifies all point IDs and generation markers, then atomically switches the shared `kuvera_lens_active` collection alias. GitHub Actions ingestion and Streamlit retrieval use these same environment variables and alias. Local development without either Cloud variable uses an isolated persistent database under `data/qdrant`; partial Cloud configuration is rejected. Each point retains its stable chunk ID, scheme ID, source identity and URL, document title/type, version/content hash, and page or section citation metadata where available.

### 4.5 Ingestion pipeline

The ingestion pipeline runs on a schedule through GitHub Actions and:

- Reads a maintained registry of approved seed URLs and domains.
- Uses Playwright to render the configured HDFC scheme pages and extract meaningful rendered text.
- Discovers linked PDFs but fetches only HTTPS URLs on the explicit official-host allowlist; validates final response hosts, status, content type, size, and PDF signature.
- Accepts scheme-specific SID, KIM, and factsheet PDFs; unrelated generic AMC documents are excluded.
- Extracts webpage content with BeautifulSoup and PDF text with PyMuPDF.
- Captures scheme name, document title/type, source URL, publication date when available, last-checked time, content hash, and PDF page numbers.
- Normalizes text, splits it into retrieval-sized chunks, and computes embeddings.
- Compares content fingerprints/versions to the prior indexed state and updates only new or changed material.
- Records ingestion outcomes so failed sources can be retried and ingestion health can be measured.

The allowlist is the authority for source trust; link discovery alone does not authorize ingestion.

The current implementation completes webpage scraping, document loading, chunking, normalized embedding generation, and versioned Qdrant indexing. Retrieval and the scheduled refresh workflow remain later implementation phases. Qdrant Cloud requires `QDRANT_URL` and `QDRANT_API_KEY` to be configured for any shared production index operation.

### 4.6 External services

- **Groq:** Generates a concise response from validated evidence. API credentials are supplied through deployment secrets and are never exposed to the browser or logs.
- **Official websites:** Provide source content. They are treated as external, potentially changed or unavailable inputs.
- **GitHub Actions:** Schedules and runs the ingestion job every 24 hours.
- **Streamlit Community Cloud:** Hosts the user-facing application.

## 5. Runtime question flow

1. The user selects a supported scheme and submits a question in the Streamlit UI.
2. The application validates the scheme and question; requests without a scheme are rejected by the UI and service.
3. The retrieval service embeds the question and searches the knowledge base with a mandatory `scheme_id` filter.
4. The answer service checks retrieved chunks for source approval, scheme consistency, and sufficient relevance.
5. If evidence is insufficient, the application returns exactly: `I couldn't verify this information from the available official sources.`
6. If evidence is sufficient, the service sends the question and validated evidence to Groq with instructions to:
   - Use only the supplied evidence.
   - Avoid advice, predictions, and unsupported facts or numbers.
   - Prefer one sentence and never exceed three sentences.
   - Indicate uncertainty rather than fill gaps.
7. The service returns a concise answer and citations constructed from evidence metadata. Citation links must point to the official source; PDF page details are included when available.
8. The UI displays the result and keeps the conversation available for follow-up questions.

For follow-up questions, prior turns may clarify references such as “that amount,” but factual claims must still be retrieved anew and validated for the currently selected scheme.

## 6. Ingestion flow

1. GitHub Actions starts the scheduled ingestion job.
2. The job loads the configured source registry and checks the prior ingestion state.
3. It fetches approved pages and discovers candidate document links.
4. Every candidate URL is normalized and checked against the approved domain and URL policy before fetching.
5. HTML and PDF content are extracted with source and page metadata.
6. Content is cleaned and fingerprinted. Unchanged documents are skipped.
7. New or changed content is chunked, embedded, and upserted into Qdrant with scheme and source metadata.
8. The job records per-source success/failure, timestamps, and document versions for monitoring and later retries.

Ingestion failures must be visible in job output or status records; a failed fetch must not be represented as a successful refresh.

## 7. Data model and identity

Use stable canonical scheme identifiers rather than display names as retrieval keys. The initial identifiers can be implementation-defined, for example:

| Display name | Canonical identifier |
|---|---|
| HDFC Large Cap Fund – Direct Growth | `hdfc_large_cap_direct_growth` |
| HDFC Flexi Cap Fund – Direct Plan Growth | `hdfc_flexi_cap_direct_growth` |
| HDFC ELSS Tax Saver Fund – Direct Plan Growth | `hdfc_elss_tax_saver_direct_growth` |
| HDFC Mid Cap Fund – Direct Growth | `hdfc_mid_cap_direct_growth` |

Source and chunk records must retain scheme association through extraction, indexing, retrieval, and citation generation. A document relevant to more than one scheme may be indexed with explicit per-scheme associations only where the source clearly supports them; otherwise, it must not be used to answer a scheme-specific question.

## 8. Trust boundaries and controls

| Boundary | Risk | Required control |
|---|---|---|
| User input to application | Unsupported requests or prompt manipulation | Validate selected scheme; treat user input as a question, not policy; enforce fixed scope and evidence checks in application code |
| Internet to ingestion | Unapproved domains, redirects, malformed links, hostile content | Validate schemes/domains and redirects before fetch; constrain ingestion to configured official sources; apply request timeouts and size limits |
| Extracted documents to index | Irrelevant or cross-scheme content | Preserve provenance and scheme identity; reject records that cannot be attributed and validated |
| Retrieved evidence to LLM | Prompt injection embedded in source text; insufficient evidence | Treat documents as untrusted quoted data; delimit evidence; instruct the model not to follow document instructions; perform deterministic source/scheme/relevance checks |
| Application to Groq | Secret exposure or unsupported model output | Keep credentials server-side in secrets; send only necessary question/context/evidence; validate output and citations before display |
| Application to browser | Leaking secrets or internal prompts | Return only user-facing answer and citation data; do not expose credentials, internal configuration, or system prompts |

The LLM is not the authorization boundary for source approval, scheme separation, or refusal. Those controls must be enforced by the application and retrieval layer.

## 9. Failure behavior

| Failure | Expected behavior |
|---|---|
| No scheme selected or unsupported scheme | Do not run retrieval; ask the user to select a supported scheme |
| No relevant or sufficient evidence | Return the exact PRD refusal message; do not call the LLM with empty evidence |
| Evidence belongs to another scheme or unapproved source | Discard it; refuse if no valid evidence remains |
| Qdrant unavailable | Show a clear temporary service error; do not generate an answer without retrieval |
| Embedding model unavailable | Show a clear service error; do not fall back to unfiltered or guessed answers |
| Groq unavailable or request fails | Show a clear generation error; do not present a fabricated success response |
| Official source unavailable during ingestion | Record the source failure; retain last successfully indexed content only according to its recorded version/status and surface staleness operationally |
| PDF or page extraction fails | Record extraction failure and do not index unverified/empty content as successful |
| Citation metadata is missing | Do not claim a precise page number; include the official source link and available title, or refuse if provenance cannot be established |

## 10. Deployment and configuration

### User-facing application

- Hosted on Streamlit Community Cloud.
- Reads Groq credentials and any deployment-specific secrets from the hosting platform's secret store.
- Uses the same approved scheme and source configuration as ingestion.
- Must not place secrets in source control, client-side state, rendered output, or user-facing logs.

### Scheduled ingestion

- Runs in GitHub Actions on a 24-hour schedule.
- Uses protected repository/workflow secrets for any required credentials.
- Writes or updates the knowledge base in the configured persistence location.
- Reports job status and per-source failures through GitHub Actions logs/status.

### Persistence consideration

The PRD selects Qdrant Cloud for the shared production index. GitHub Actions and Streamlit must use the same `QDRANT_URL`, `QDRANT_API_KEY`, and `kuvera_lens_active` alias. Local development uses an isolated local Qdrant path and must not be treated as the shared production store.

## 11. Observability and operational metrics

Capture operationally useful, non-sensitive events for:

- Ingestion run time, per-source success/failure, discovered/changed document counts, and extraction errors.
- Retrieval latency, number of valid evidence chunks, and refusal outcomes.
- Answer-generation success/failure and total response time.
- Citation metadata completeness.

Do not log secrets, full user credentials, or unnecessary personal data. User questions should not be retained in persistent logs unless a documented product need and privacy policy are established.

These events support the PRD success measures: retrieval accuracy, citation correctness, refusal accuracy, scheme separation, ingestion success, update detection, response time, and successful completion of supported questions.

## 12. Architecture decisions to confirm during implementation

The PRD specifies the main technologies and behavior but leaves these implementation details open:

- Durable/shared Qdrant Cloud access between Streamlit hosting and GitHub Actions.
- Source registry format and the specific approved URLs for each supported scheme.
- Chunk size, overlap, retrieval count, and evidence relevance thresholds.
- Exact conversation retention and session expiration behavior.
- Operational retention policy for logs and ingestion history.

These choices must preserve the requirements in this document, especially source provenance, scheme isolation, fail-closed behavior, and daily change detection.
