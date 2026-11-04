"""Streamlit views for the Kuvera Lens scheme selection and conversation flow."""

import logging
from typing import TypedDict

import httpx
import requests
import streamlit as st
from groq import GroqError, RateLimitError
from qdrant_client.http.exceptions import ApiException

from kuvera_lens.answering import AnswerResult, answer_question
from kuvera_lens.retrieval import SourceType, VERIFICATION_REFUSAL
from kuvera_lens.schemes import SCHEME_BY_ID, SUPPORTED_SCHEMES, Scheme
from kuvera_lens.suggestions import SuggestedQuestion, get_suggested_questions

LOGGER = logging.getLogger(__name__)
MAX_CONTEXT_TURNS = 8
MAX_CONTEXT_CHARACTERS = 3000
MAX_CITATION_TITLE_LENGTH = 44


def _service_error_message(exc: Exception) -> str:
    if isinstance(exc, GroqError) and type(exc).__name__ in {
        "AuthenticationError",
        "PermissionDeniedError",
    }:
        return "Groq rejected the configured credentials. Check GROQ_API_KEY."
    if isinstance(exc, RateLimitError):
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", {})
        retry_after = headers.get("retry-after") if headers is not None else None
        try:
            retry_seconds = int(float(retry_after))
        except (TypeError, ValueError):
            retry_seconds = 0
        if retry_seconds > 0:
            retry_minutes = max(1, (retry_seconds + 59) // 60)
            return (
                "The Groq service is rate-limiting requests. "
                f"Please try again in about {retry_minutes} minutes."
            )
        return (
            "The Groq service is rate-limiting requests. "
            "Please wait a moment and try again."
        )
    return "The answer service is temporarily unavailable. Please try again."


class ChatMessage(TypedDict, total=False):
    role: str
    content: str
    refused: bool
    error: bool
    citations: list[dict[str, str | int | None]]


def _theme_css() -> str:
    return """
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    :root {
        --lens-bg: #050505;
        --lens-surface: #121214;
        --lens-raised: #18181b;
        --lens-border: #27272a;
        --lens-gold: #e5b842;
        --lens-gold-light: #ffd56d;
        --lens-text: #ffffff;
        --lens-muted: #a1a1aa;
    }
    .stApp {
        background: var(--lens-bg);
        color: var(--lens-text);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: var(--lens-bg);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"], [data-testid="stDeployButton"], #MainMenu, footer {
        display: none !important;
    }
    .block-container {
        max-width: 1120px;
        padding: 1.25rem 1.5rem 2rem;
    }
    .lens-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        min-height: 48px;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--lens-border);
        margin-bottom: 26px;
    }
    .lens-wordmark {
        color: var(--lens-text);
        font-size: 17px;
        font-weight: 700;
        letter-spacing: -0.03em;
    }
    .lens-wordmark span { color: var(--lens-gold); }
    [class*="st-key-conversation-header"] {
        position: relative;
        z-index: 2;
        background: #101012;
        border: 1px solid var(--lens-border);
        border-radius: 12px;
        padding: 4px 10px;
        margin-bottom: 12px;
    }
    [class*="st-key-conversation-header"] .lens-topbar {
        min-height: 38px;
        padding: 0;
        border: 0;
        margin: 0;
    }
    [class*="st-key-new-conversation"] [data-testid="stButton"] > button {
        min-height: 32px;
        padding: 0 11px;
        border-radius: 999px;
        background: #1b1b1e !important;
        border-color: #3a3a3e !important;
        color: #f4f4f5 !important;
        font-size: 0.78rem;
    }
    .lens-overline {
        color: var(--lens-gold);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.11em;
        text-transform: uppercase;
    }
    .lens-hero h1 {
        color: var(--lens-text);
        font-size: clamp(2rem, 5vw, 3rem);
        line-height: 1.12;
        letter-spacing: -0.045em;
        margin: 0.65rem 0 0.45rem;
    }
    .lens-hero h1 span { color: var(--lens-gold); }
    .lens-subtitle { color: var(--lens-muted); font-size: 1rem; margin-bottom: 0; }
    .lens-section-label {
        color: #d1c5af;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 1.25rem 0 0.75rem;
    }
    [class*="st-key-scheme-card"] {
        background: var(--lens-surface);
        border: 1px solid var(--lens-border);
        border-radius: 9px;
        padding: 16px;
        min-height: 158px;
        margin-bottom: 12px;
    }
    [class*="st-key-scheme-card-selected"] { border-color: #d4af37; }
    [class*="st-key-scheme-card"] [data-testid="stButton"] button {
        min-height: 76px;
        width: 100%;
        justify-content: flex-start;
        text-align: left;
        white-space: normal;
        line-height: 1.4;
        font-size: 0.96rem;
        font-weight: 650;
        background: transparent;
        color: var(--lens-text);
        border: 0;
        padding: 6px 0;
        box-shadow: none;
    }
    [class*="st-key-scheme-card-selected"] [data-testid="stButton"] button {
        color: var(--lens-gold-light);
    }
    [class*="st-key-scheme-card"] [data-testid="stButton"] button:hover {
        color: var(--lens-gold-light);
        border: 0;
        background: transparent;
    }
    .lens-card-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        color: var(--lens-muted);
        font-size: 0.7rem;
    }
    .lens-tag {
        padding: 3px 7px;
        border: 1px solid var(--lens-border);
        border-radius: 4px;
        background: var(--lens-raised);
    }
    .lens-card-plan { color: var(--lens-muted); font-size: 0.77rem; }
    .lens-rupee {
        color: var(--lens-gold);
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: clamp(5rem, 14vw, 8rem);
        font-weight: 400;
        line-height: 1;
        text-align: center;
        opacity: 0.86;
        padding: 1rem 0;
    }
    .lens-trust {
        color: #e5e1e4;
        text-align: center;
        font-size: 0.82rem;
        padding: 12px 0 6px;
    }
    .lens-trust span { color: var(--lens-gold); margin-right: 7px; }
    [data-testid="stChatMessage"] {
        background: transparent;
        border: 0;
        padding: 0.35rem 0;
    }
    [data-testid="stChatMessageContent"] {
        color: var(--lens-text);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        justify-content: flex-end;
    }
    .lens-scheme-strip {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        border: 1px solid var(--lens-border);
        background: #0b0b0c;
        border-radius: 8px;
        padding: 12px 14px;
        margin: 2px 0 18px;
    }
    .lens-scheme-name { font-size: 0.9rem; font-weight: 600; }
    .lens-scheme-label {
        color: var(--lens-gold);
        font-size: 0.68rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        white-space: nowrap;
    }
    .lens-answer-card {
        background: var(--lens-surface);
        border: 1px solid var(--lens-border);
        border-radius: 8px;
        padding: 16px;
        margin: 4px 0 12px;
        line-height: 1.65;
    }
    [class*="st-key-answer-card"] {
        background: #141416;
        border: 1px solid #303034;
        border-radius: 12px;
        padding: 14px;
        margin: 4px 0 10px;
    }
    [class*="st-key-answer-card"] [data-testid="stText"],
    [class*="st-key-answer-card"] [data-testid="stText"] *,
    [class*="st-key-answer-card"] [data-testid="stText"] pre {
        color: #f4f4f5 !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.96rem;
        line-height: 1.6;
        white-space: pre-wrap;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stText"],
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stText"] * {
        color: #f4f4f5 !important;
        background: #19191c;
        border: 1px solid #303034;
        border-radius: 14px;
        padding: 9px 13px;
    }
    .lens-source-title {
        color: var(--lens-gold-light);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        margin: 8px 0;
    }
    [data-testid="stLinkButton"] {
        margin: 0 0 5px;
    }
    [data-testid="stLinkButton"] a {
        min-height: 36px;
        padding: 5px 10px;
        overflow: hidden;
        border: 1px solid #4a4026;
        border-radius: 8px;
        background: #201d16;
        color: #f4d57d !important;
        font-size: 0.76rem;
        line-height: 1.25;
        text-align: left;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    [data-testid="stLinkButton"] a:hover {
        border-color: var(--lens-gold);
        background: #292317;
        color: #ffe49a !important;
    }
    [data-testid="stBottomBlockContainer"] {
        background: rgba(8, 8, 9, 0.98) !important;
        border-top: 1px solid #27272a;
        padding: 7px 0 3px;
        box-shadow: 0 -8px 24px rgba(0, 0, 0, 0.24);
    }
    [data-testid="stBottomBlockContainer"]::after {
        display: block;
        padding: 1px 0 2px;
        color: #a1a1aa;
        content: "◇  Clear answers. Trusted sources.";
        font-size: 0.66rem;
        text-align: center;
    }
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] textarea {
        border-color: #303034 !important;
        border-radius: 15px !important;
        background-color: #171719 !important;
        color: #f4f4f5 !important;
    }
    [data-testid="stChatInput"] { min-height: 50px; }
    [data-testid="stChatInput"]:focus-within { border-color: var(--lens-gold); }
    [data-testid="stChatInput"] textarea {
        padding-top: 12px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.88rem;
    }
    [data-testid="stChatInput"] textarea::placeholder { color: #b4b4bb; }
    [data-testid="stChatInputSubmitButton"]:not(:disabled) {
        background: var(--lens-gold) !important;
        color: #050505 !important;
        border-radius: 11px !important;
        width: 34px;
        height: 34px;
        margin-right: 5px;
    }
    [data-testid="stChatInputSubmitButton"]:disabled {
        background: var(--lens-gold) !important;
        color: #050505 !important;
        border-radius: 11px !important;
        width: 34px;
        height: 34px;
        margin-right: 5px;
        opacity: 0.88;
    }
    [data-testid="stButton"] > button {
        background: var(--lens-surface) !important;
        border-color: #333338;
        border-radius: 6px;
        color: var(--lens-text) !important;
    }
    [data-testid="stButton"] > button:hover {
        background: var(--lens-raised) !important;
        border-color: var(--lens-gold);
        color: var(--lens-gold-light) !important;
    }
    [class*="st-key-start-conversation"] [data-testid="stButton"] > button[kind="primary"] {
        min-height: 50px;
        background: var(--lens-gold) !important;
        border-color: var(--lens-gold) !important;
        color: #050505 !important;
        font-weight: 700;
    }
    [class*="st-key-start-conversation"] [data-testid="stButton"] > button:disabled {
        background: #353437 !important;
        border-color: #353437 !important;
        color: #a1a1aa !important;
    }
    @media (max-width: 640px) {
        .block-container { padding: 0.5rem 0.65rem 0.75rem; }
        .lens-topbar { margin-bottom: 14px; }
        [class*="st-key-conversation-header"] { padding: 3px 8px; }
        .lens-scheme-strip {
            padding: 8px 10px;
            margin: 0 0 8px;
        }
        [data-testid="stHorizontalBlock"]:has([class*="st-key-scheme-card"]) {
            display: flex !important;
            flex-wrap: nowrap !important;
            flex-direction: row !important;
            gap: 12px;
        }
        [data-testid="stHorizontalBlock"]:has([class*="st-key-scheme-card"])
        > [data-testid="stColumn"] {
            min-width: 0 !important;
            flex: 1 1 0% !important;
            width: calc(50% - 6px) !important;
        }
        [class*="st-key-scheme-card"] { padding: 11px; min-height: 150px; }
        [class*="st-key-scheme-card"] [data-testid="stButton"] button {
            font-size: 0.84rem;
            min-height: 82px;
        }
        .lens-scheme-strip { align-items: flex-start; }
        .lens-scheme-name { font-size: 0.78rem; }
        [class*="st-key-answer-card"] { padding: 12px; }
        [data-testid="stBottomBlockContainer"] { padding: 5px 0 2px; }
        [data-testid="stChatInput"] { min-height: 46px; }
        [data-testid="stChatInput"] textarea { font-size: 0.82rem; }
        [class*="st-key-suggested-followups"] [data-testid="stHorizontalBlock"] {
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 6px;
        }
        [class*="st-key-suggested-followups"] [data-testid="stColumn"] {
            flex: 0 1 auto !important;
            width: auto !important;
            min-width: 0 !important;
        }
        [class*="st-key-suggested-followups"] [data-testid="stButton"] > button {
            min-height: 30px;
            padding: 4px 9px;
            border-radius: 999px;
            font-size: 0.72rem;
            white-space: nowrap;
        }
    }
    """


def _initialise_state() -> None:
    st.session_state.setdefault("lens_screen", "selection")
    st.session_state.setdefault("lens_scheme_id", None)
    st.session_state.setdefault("lens_messages", [])
    st.session_state.setdefault("lens_suggestions_scheme_id", None)
    st.session_state.setdefault("lens_suggestions", [])
    st.session_state.setdefault("lens_suggestion_error", False)


def _select_scheme(scheme_id: str) -> None:
    if st.session_state.get("lens_scheme_id") != scheme_id:
        st.session_state["lens_messages"] = []
        st.session_state["lens_suggestions_scheme_id"] = None
        st.session_state["lens_suggestions"] = []
        st.session_state["lens_suggestion_error"] = False
    st.session_state["lens_scheme_id"] = scheme_id


def _start_conversation() -> None:
    if st.session_state.get("lens_scheme_id") in SCHEME_BY_ID:
        st.session_state["lens_screen"] = "conversation"


def _return_to_selection() -> None:
    st.session_state["lens_screen"] = "selection"
    st.session_state["lens_scheme_id"] = None
    st.session_state["lens_messages"] = []
    st.session_state["lens_suggestions_scheme_id"] = None
    st.session_state["lens_suggestions"] = []
    st.session_state["lens_suggestion_error"] = False


def _conversation_context(messages: list[ChatMessage]) -> str | None:
    usable = [
        message
        for message in messages
        if not message.get("error")
        and message.get("role") in {"user", "assistant"}
        and isinstance(message.get("content"), str)
    ][-MAX_CONTEXT_TURNS:]
    if not usable:
        return None

    lines = [
        f"{'User' if message['role'] == 'user' else 'Assistant'}: "
        f"{message['content']}"
        for message in usable
    ]
    context = "\n".join(lines)
    return context[-MAX_CONTEXT_CHARACTERS:]


def _answer_payload(result: AnswerResult) -> ChatMessage:
    citations = [
        {
            "title": citation.title,
            "source_url": citation.source_url,
            "page_number": citation.page_number,
            "section": citation.section,
        }
        for citation in result.citations
    ]
    return {
        "role": "assistant",
        "content": result.answer,
        "refused": result.refused,
        "citations": citations,
    }


def _submit_question(
    question: str,
    *,
    source_type: SourceType | None = None,
) -> None:
    normalized_question = question.strip()
    scheme_id = st.session_state.get("lens_scheme_id")
    if not normalized_question or scheme_id not in SCHEME_BY_ID:
        return

    messages: list[ChatMessage] = st.session_state["lens_messages"]
    context = _conversation_context(messages)
    messages.append({"role": "user", "content": normalized_question})
    try:
        if source_type is None:
            result = answer_question(
                normalized_question,
                scheme_id,
                conversation_context=context,
            )
        else:
            result = answer_question(
                normalized_question,
                scheme_id,
                conversation_context=context,
                source_type=source_type,
            )
    except (
        ApiException,
        GroqError,
        httpx.HTTPError,
        requests.RequestException,
        RuntimeError,
        OSError,
        ValueError,
    ) as exc:
        LOGGER.error("Chat request failed (%s).", type(exc).__name__)
        messages.append(
            {
                "role": "assistant",
                "content": _service_error_message(exc),
                "error": True,
            }
        )
        return
    messages.append(_answer_payload(result))


def _render_header(*, conversation: bool = False) -> None:
    if not conversation:
        st.markdown(
            '<div class="lens-topbar"><div class="lens-wordmark">Kuvera '
            '<span>· Lens</span></div></div>',
            unsafe_allow_html=True,
        )
        return

    with st.container(key="conversation-header"):
        left, right = st.columns([5, 1], vertical_alignment="center")
        with left:
            st.markdown(
                '<div class="lens-topbar"><div class="lens-wordmark">Kuvera '
                '<span>· Lens</span></div></div>',
                unsafe_allow_html=True,
            )
        with right:
            if st.button("＋ New", key="new-conversation"):
                _return_to_selection()
                st.rerun()


def _scheme_plan(scheme: Scheme) -> str:
    return "Direct Plan Growth" if "Direct Plan" in scheme.display_name else "Direct Growth"


def _render_scheme_card(scheme: Scheme) -> None:
    selected = st.session_state.get("lens_scheme_id") == scheme.scheme_id
    container_key = (
        f"scheme-card-selected-{scheme.scheme_id}"
        if selected
        else f"scheme-card-{scheme.scheme_id}"
    )
    with st.container(key=container_key):
        st.markdown(
            f'<div class="lens-card-top"><span>{"● Selected" if selected else "◌ Scheme"}</span>'
            f'<span class="lens-tag">{scheme.display_name.split(" Fund")[0].replace("HDFC ", "")}</span></div>',
            unsafe_allow_html=True,
        )
        if st.button(
            scheme.display_name,
            key=f"select-{scheme.scheme_id}",
            use_container_width=True,
        ):
            _select_scheme(scheme.scheme_id)
            st.rerun()
        st.markdown(
            f'<div class="lens-card-plan">{_scheme_plan(scheme)}</div>',
            unsafe_allow_html=True,
        )


def _render_selection() -> None:
    _render_header()
    st.markdown(
        '<div class="lens-hero"><div class="lens-overline">● &nbsp; Official-source assistant</div>'
        '<h1>Kuvera <span>– Lens</span></h1>'
        '<p class="lens-subtitle">Your facts-only mutual fund companion.</p></div>'
        '<div class="lens-section-label">Select HDFC scheme</div>',
        unsafe_allow_html=True,
    )
    for row_start in (0, 2):
        columns = st.columns(2, gap="medium")
        for column, scheme in zip(columns, SUPPORTED_SCHEMES[row_start : row_start + 2]):
            with column:
                _render_scheme_card(scheme)

    st.markdown('<div class="lens-rupee" aria-hidden="true">₹</div>', unsafe_allow_html=True)
    with st.container(key="start-conversation"):
        st.button(
            "Start Conversation  →",
            key="start-chat",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.get("lens_scheme_id") not in SCHEME_BY_ID,
            on_click=_start_conversation,
        )
    st.markdown(
        '<div class="lens-trust"><span>◇</span>Clear answers. Trusted sources.</div>',
        unsafe_allow_html=True,
    )


def _render_citations(message: ChatMessage) -> None:
    citations = message.get("citations", [])
    if not citations:
        return
    st.markdown('<div class="lens-source-title">Official sources</div>', unsafe_allow_html=True)
    for index, citation in enumerate(citations, start=1):
        title = citation.get("title") or "Official document"
        title = title.split(" | ", maxsplit=1)[0].split(" · ", maxsplit=1)[0].strip()
        if len(title) > MAX_CITATION_TITLE_LENGTH:
            title = title[: MAX_CITATION_TITLE_LENGTH - 1].rstrip() + "…"
        page = citation.get("page_number")
        detail = f" · p. {page}" if page is not None else ""
        st.link_button(
            f"{index}. {title}{detail}",
            str(citation["source_url"]),
            use_container_width=True,
        )


def _render_chat_history() -> None:
    messages: list[ChatMessage] = st.session_state["lens_messages"]
    for index, message in enumerate(messages):
        role = message.get("role", "assistant")
        with st.chat_message(role):
            if message.get("error"):
                st.error(message["content"])
            elif message.get("refused") or message.get("content") == VERIFICATION_REFUSAL:
                st.warning(VERIFICATION_REFUSAL)
            elif role == "user":
                st.text(message["content"])
            else:
                with st.container(key=f"answer-card-{index}"):
                    st.text(message["content"])
                    _render_citations(message)


def _render_suggestions(scheme_id: str) -> None:
    if not st.session_state["lens_messages"]:
        return
    if st.session_state.get("lens_suggestions_scheme_id") != scheme_id:
        try:
            st.session_state["lens_suggestions"] = list(
                get_suggested_questions(scheme_id)
            )
            st.session_state["lens_suggestion_error"] = False
        except (
            ApiException,
            httpx.HTTPError,
            requests.RequestException,
            RuntimeError,
            OSError,
            ValueError,
        ) as exc:
            LOGGER.error("Suggested-question lookup failed (%s).", type(exc).__name__)
            st.session_state["lens_suggestions"] = []
            st.session_state["lens_suggestion_error"] = True
        st.session_state["lens_suggestions_scheme_id"] = scheme_id

    suggestions: list[SuggestedQuestion] = st.session_state["lens_suggestions"]
    if not suggestions:
        if st.session_state["lens_suggestion_error"]:
            st.warning("Suggested questions are temporarily unavailable.")
        return
    with st.container(key="suggested-followups"):
        st.markdown('<div class="lens-section-label">Suggested follow-ups</div>', unsafe_allow_html=True)
        columns = st.columns(len(suggestions))
        for column, suggestion in zip(columns, suggestions):
            with column:
                if st.button(
                    f"＋ {suggestion.question}",
                    key=f"suggest-{len(st.session_state['lens_messages'])}-{suggestion.question}",
                    use_container_width=True,
                ):
                    _submit_question(
                        suggestion.question,
                        source_type=suggestion.source_type,
                    )
                    st.rerun()


def _render_conversation() -> None:
    scheme_id = st.session_state.get("lens_scheme_id")
    scheme = SCHEME_BY_ID.get(scheme_id)
    if scheme is None:
        _return_to_selection()
        st.rerun()
        return

    _render_header(conversation=True)
    strip, change = st.columns([5, 1])
    with strip:
        st.markdown(
            f'<div class="lens-scheme-strip"><span class="lens-scheme-name">'
            f'▣ &nbsp; {scheme.display_name}</span>'
            f'<span class="lens-scheme-label">Selected scheme</span></div>',
            unsafe_allow_html=True,
        )
    with change:
        if st.button("Change", key="change-scheme"):
            _return_to_selection()
            st.rerun()

    if not st.session_state["lens_messages"]:
        st.markdown(
            '<div class="lens-answer-card">Ask a question about this scheme. '
            'Answers use retrieved official-source evidence and include source links '
            'when available.</div>',
            unsafe_allow_html=True,
        )
    _render_chat_history()
    _render_suggestions(scheme_id)
    question = st.chat_input(
        "Ask a question about this fund...",
        max_chars=500,
        key="chat-question",
    )
    if question:
        _submit_question(question)
        st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Kuvera – Lens",
        page_icon="🔎",
        layout="wide",
    )
    st.markdown(f"<style>{_theme_css()}</style>", unsafe_allow_html=True)
    _initialise_state()
    if st.session_state["lens_screen"] == "conversation":
        _render_conversation()
    else:
        _render_selection()
