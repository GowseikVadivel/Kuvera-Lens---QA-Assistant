# Kuvera – Lens
## Product Requirements Document

**Version:** 1.0  
**Product category:** AI-powered mutual fund information assistant  
**Tagline:** Your facts-only mutual fund companion.

## 1. Product summary

Kuvera – Lens is a conversation-first assistant that helps users find specific information about supported mutual fund schemes. Users select a scheme, ask questions in natural language, and receive concise answers grounded in retrieved information from approved official sources. Each answer includes a source users can open to verify the information.

The assistant is informational only, not an investment adviser. If official-source evidence is insufficient, it must say:

> I couldn't verify this information from the available official sources.

It must not guess or fabricate an answer.

## 2. Problem and opportunity

Investors often need to search lengthy scheme documents, factsheets, and official webpages for details such as minimum investments, expense ratios, exit loads, lock-in periods, and riskometer information. Finding an exact answer can be difficult, especially for first-time investors and people unfamiliar with financial terminology. Unofficial sources may also provide inconsistent or outdated information.

Kuvera – Lens addresses this by letting users ask a question in plain language and receive a clear, verifiable answer from official sources.

## 3. Goals

- Make supported scheme facts easy to find through natural-language questions.
- Ground every factual answer in retrieved official-source evidence.
- Make answers concise and easy to understand.
- Provide clickable citations, preserving document title and page number where available.
- Clearly refuse to answer when the available evidence cannot verify the requested fact.
- Keep evidence and answers isolated to the scheme selected by the user.
- Refresh approved sources daily, processing only new or changed content.

## 4. Non-goals

The initial release will not provide:

- Mutual fund purchases, redemptions, or brokerage/account integrations.
- Portfolio tracking or account management.
- Personalized investment or portfolio advice.
- Buy, sell, or redeem recommendations.
- Return predictions or investment performance comparisons.
- General-purpose AI conversations.

## 5. Target users

The primary users are retail mutual fund investors, including:

- First-time investors seeking to understand scheme details.
- Existing investors researching a particular scheme.
- Users who prefer asking questions to searching lengthy documents.
- Users who want to verify information against official sources.

## 6. Initial release scope

### 6.1 Supported schemes

Users must select one of these four HDFC Mutual Fund schemes before starting a conversation:

1. HDFC Large Cap Fund – Direct Growth
2. HDFC Flexi Cap Fund – Direct Plan Growth
3. HDFC ELSS Tax Saver Fund – Direct Plan Growth
4. HDFC Mid Cap Fund – Direct Growth

### 6.2 Supported question categories

The assistant may answer factual questions about:

- **Scheme information:** objectives, fund category, benchmark, and riskometer.
- **Investment details:** minimum investment, minimum SIP, additional purchase requirements, and applicable investment conditions.
- **Charges and exit conditions:** expense ratio, exit load, and applicable charges.
- **Tax and lock-in information:** ELSS lock-in period and scheme-related tax information available in official documents.
- **Documents and statements:** factsheets, Scheme Information Documents (SID), Key Information Memorandums (KIM), and official instructions for downloading statements and capital gains reports.

## 7. Functional requirements

### FR-1: Scheme selection

The application must require the user to select one of the four supported schemes before starting a conversation.

**Acceptance criteria**

- All four supported schemes are available for selection.
- Questions cannot be submitted until a scheme is selected.
- Retrieval and answer generation use evidence for the selected scheme only.

### FR-2: Natural-language questions

The user must be able to ask supported questions in conversational language and continue asking related questions in the same conversation.

**Acceptance criteria**

- A user can submit multiple questions without restarting the conversation.
- Conversation continuity must not cause evidence from another scheme to be used.

### FR-3: Evidence-based answers

Every factual answer must be generated from relevant evidence retrieved from the indexed official-source knowledge base.

**Acceptance criteria**

- The assistant does not provide unsupported facts or numerical values.
- The assistant does not combine evidence from different schemes.
- If sufficient evidence is unavailable, the assistant returns the required refusal message rather than guessing.

### FR-4: Concise, understandable responses

Answers must be easy to understand, avoid unnecessary financial terminology, and prefer one sentence. An answer must not exceed three sentences.

### FR-5: Source citations

Every factual answer must include relevant official source information and a clickable source link. Where available, citations should preserve the document title and page number.

### FR-6: Official-source ingestion

The ingestion system must:

1. Maintain an approved list of official source URLs.
2. Scrape approved webpages.
3. Discover relevant linked official PDFs, including SID, KIM, and factsheets.
4. Validate discovered URLs and restrict ingestion to approved official domains.
5. Extract webpage and PDF content.
6. Track source metadata and document versions.
7. Check approved sources every 24 hours and process only newly discovered or changed content.

### FR-7: Supported official sources

The initial approved source organizations are:

- HDFC Asset Management Company (HDFC AMC).
- Securities and Exchange Board of India (SEBI).
- Association of Mutual Funds in India (AMFI).

Only verified content from approved official sources may be used as factual evidence.

## 8. Safety, privacy, and trust requirements

The assistant must:

- Never fabricate financial information or generate unsupported numerical values.
- Never provide buy, sell, or redeem recommendations.
- Never predict mutual fund returns.
- Never provide personalized investment or portfolio advice.
- Never use unapproved third-party sources as factual evidence.
- Never expose API keys or internal system information.
- Avoid collecting or storing PAN numbers, Aadhaar numbers, OTPs, bank account details, personal investment account credentials, or unnecessary personal contact information.
- Operate without integration into users' personal investment accounts.

## 9. User experience requirements

- Provide a conversational interface for selecting a scheme and asking questions.
- Keep answers concise and readable.
- Make source citations easy to open and verify.
- Preserve conversation continuity for related questions.
- Clearly communicate when information cannot be verified.

## 10. Technical approach

| Component | Technology |
|---|---|
| Programming language | Python |
| User interface | Streamlit |
| Web scraping | Playwright and BeautifulSoup |
| PDF extraction | PyMuPDF |
| Text processing | Python |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database | Qdrant Cloud for shared production indexing and retrieval; isolated local Qdrant storage for development |
| LLM provider | Groq |
| Source refresh | GitHub Actions |
| Hosting | Streamlit Community Cloud |

### RAG workflow

1. Collect approved official website URLs.
2. Scrape webpages and discover relevant PDFs.
3. Load documents and extract text.
4. Clean and split text into chunks.
5. Generate vector embeddings and store chunks, embeddings, and metadata in Qdrant. GitHub Actions ingestion and Streamlit retrieval must use the same Qdrant Cloud collection in production, configured with `QDRANT_URL` and `QDRANT_API_KEY`; local development data must remain separate.
6. Convert user questions into embeddings and retrieve relevant evidence.
7. Validate evidence for relevance, source approval, and selected-scheme consistency.
8. Generate a concise answer using Groq, grounded in validated evidence.
9. Display the answer with official source citations.

## 11. Success metrics

Evaluate the initial release using:

- Answer accuracy.
- Evidence retrieval accuracy.
- Source citation correctness.
- Unsupported-answer refusal accuracy.
- Scheme separation accuracy.
- Daily ingestion success rate.
- Document update detection accuracy.
- Response time.
- Successful completion of supported user questions.

## 12. Release acceptance criteria

The initial release is acceptable when:

- Users can select any of the four supported schemes and ask supported factual questions.
- Answers are grounded only in retrieved, approved official-source evidence for the selected scheme.
- Factual answers include clickable citations, with document title and page number where available.
- Responses follow the one-to-three-sentence limit and avoid unnecessary jargon.
- The exact refusal message is shown when evidence is insufficient.
- The assistant follows all advice, fabrication, source, and privacy guardrails in this PRD.
- Approved sources are checked every 24 hours and only new or changed content is processed.
- Users can ask related follow-up questions within the same conversation.

## 13. Source document

This PRD is derived from [Docs/Problem Statement.txt](./Docs/Problem%20Statement.txt).
