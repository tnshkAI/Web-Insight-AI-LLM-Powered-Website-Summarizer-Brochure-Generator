"""
🌐 Website Summarizer & Brochure Generator
Streamlit front-end for an LLM-powered web-analysis pipeline.

Features
- Works with Ollama (local) or any OpenAI-compatible API (Groq, OpenAI, OpenRouter, ...)
- Mode 1: Website summarizer with selectable tone/style
- Mode 2: Company brochure generator (multi-step LLM pipeline):
      scrape links -> LLM picks relevant pages (JSON) -> scrape them -> stream brochure
- Streaming responses, cached scraping, graceful error handling, markdown download

Run locally:   streamlit run app.py
"""

import json
import re
import time
from urllib.parse import urljoin, urlparse

import openai
import requests
import streamlit as st
from openai import OpenAI

from scraper import fetch_website_contents, fetch_website_links

# --------------------------------------------------------------------------------------
# Page config & styling
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Web Insight AI",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
    }
    .hero h1 { margin: 0; font-size: 2.1rem; color: white; }
    .hero p  { margin: .4rem 0 0 0; opacity: .92; font-size: 1.02rem; }
    .pill {
        display: inline-block; padding: .15rem .65rem; margin: .5rem .35rem 0 0;
        border-radius: 999px; background: rgba(255,255,255,.18);
        font-size: .78rem; backdrop-filter: blur(4px);
    }
    div[data-testid="stMetric"] {
        background: rgba(124, 58, 237, 0.07);
        border: 1px solid rgba(124, 58, 237, 0.18);
        padding: .7rem 1rem; border-radius: 14px;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        border: none; border-radius: 12px; font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover { filter: brightness(1.1); }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
PROVIDERS = {
    "Ollama (local)": {
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.2",
        "needs_key": False,
    },
    "Groq (hosted, free tier)": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.1-8b-instant",
        "needs_key": True,
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "needs_key": True,
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.1-8b-instruct",
        "needs_key": True,
    },
    "Custom (OpenAI-compatible)": {
        "base_url": "",
        "model": "",
        "needs_key": True,
    },
}

SUMMARY_TONES = {
    "😏 Snarky (original)": (
        "You are a snarky assistant that analyzes the contents of a website, and provides a short, "
        "snarky, humorous summary, ignoring text that might be navigation related."
    ),
    "💼 Professional": (
        "You are a professional business analyst. Summarize the website clearly and concisely "
        "in a neutral, polished tone, ignoring navigation-related text."
    ),
    "🧒 Explain like I'm 5": (
        "You explain websites to a five-year-old using simple words and fun analogies. "
        "Ignore navigation-related text."
    ),
    "🏴‍☠️ Pirate": (
        "You are a pirate captain summarizing websites. Stay accurate, but speak like a pirate. "
        "Ignore navigation-related text."
    ),
    "📋 Executive bullets": (
        "You are an executive assistant. Produce a crisp bullet-point briefing: purpose, key offerings, "
        "notable news/announcements, and one takeaway. Ignore navigation-related text."
    ),
}

SUMMARY_USER_PREFIX = (
    "Here are the contents of a website.\n"
    "Provide a short summary of this website.\n"
    "If it includes news or announcements, then summarize these too.\n\n"
)

MARKDOWN_RULE = "Respond in markdown. Do not wrap the markdown in a code block - respond just with the markdown."

LINK_SYSTEM_PROMPT = """
You are given a list of links found on a company's website.
Decide which links are relevant for a brochure about the company
(About, Company, Careers/Jobs, Customers, Products, Pricing, News/Blog pages).
Exclude Terms of Service, Privacy, cookie, login, email and social-media links.
Respond ONLY with JSON in exactly this format:
{
  "links": [
    {"type": "about page", "url": "https://full.url/goes/here"},
    {"type": "careers page", "url": "https://another.full.url/careers"}
  ]
}
"""

BROCHURE_STYLES = {
    "Professional": "Write a short, polished brochure for prospective customers, investors and recruits.",
    "Humorous": "Write a short, witty and entertaining brochure for prospective customers, investors and recruits.",
}

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def get_secret(name: str, default: str = "") -> str:
    """Read from st.secrets without crashing when no secrets file exists."""
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def normalize_url(raw: str) -> str | None:
    raw = raw.strip()
    if not raw:
        return None
    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw
    parsed = urlparse(raw)
    return raw if parsed.netloc and "." in parsed.netloc else None


@st.cache_data(show_spinner=False, ttl=3600)
def cached_contents(url: str) -> str:
    return fetch_website_contents(url)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_links(url: str) -> list[str]:
    return fetch_website_links(url)


def stream_chat(client: OpenAI, model: str, system: str, user: str, temperature: float):
    """Generator yielding text chunks - plugs straight into st.write_stream."""
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def parse_json_loose(text: str) -> dict:
    """Parse JSON even if the model wraps it in prose or code fences."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def select_relevant_links(
    client: OpenAI, model: str, base_url: str, links: list[str]
) -> list[dict]:
    """Step 2 of the brochure pipeline: ask the LLM to pick relevant links (structured JSON)."""
    skip = ("#", "mailto:", "tel:", "javascript:")
    absolute = list(
        dict.fromkeys(
            urljoin(base_url, l) for l in links if l and not l.startswith(skip)
        )
    )[:100]

    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": LINK_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Links found on {base_url}:\n" + "\n".join(absolute),
            },
        ],
        temperature=0,
    )
    try:
        resp = client.chat.completions.create(
            response_format={"type": "json_object"}, **kwargs
        )
    except openai.BadRequestError:
        # Some providers/models don't support JSON mode - fall back to prompt-only JSON.
        resp = client.chat.completions.create(**kwargs)

    data = parse_json_loose(resp.choices[0].message.content)
    picked = []
    for item in data.get("links", []):
        link = item.get("url", "")
        if link.startswith(("http://", "https://")):
            picked.append({"type": item.get("type", "page"), "url": link})
    return picked


# --------------------------------------------------------------------------------------
# Pipelines (each returns: raw_context, pages_used, token_generator)
# --------------------------------------------------------------------------------------


def run_summary(client, model, url, tone, temperature):
    with st.status("Fetching website…", expanded=False) as status:
        raw = cached_contents(url)
        status.update(label="Website fetched ✅", state="complete")
    system = f"{SUMMARY_TONES[tone]}\n{MARKDOWN_RULE}"
    gen = stream_chat(client, model, system, SUMMARY_USER_PREFIX + raw, temperature)
    return raw, [{"type": "landing page", "url": url}], gen


def run_brochure(client, model, url, style, temperature, max_pages):
    pages = [{"type": "landing page", "url": url}]
    with st.status("Building brochure…", expanded=True) as status:
        st.write("🔗 Step 1/3 — collecting links from the landing page")
        links = cached_links(url)
        st.write(f"   Found **{len(links)}** links")

        st.write(
            "🧠 Step 2/3 — asking the LLM which pages matter (structured JSON output)"
        )
        try:
            picked = select_relevant_links(client, model, url, links)[:max_pages]
        except Exception as e:  # keep going with just the landing page
            st.write(
                f"   ⚠️ Link selection failed ({type(e).__name__}); using landing page only"
            )
            picked = []
        for p in picked:
            st.write(f"   • {p['type']}: {p['url']}")

        st.write("📥 Step 3/3 — scraping the selected pages")
        sections = [f"## Landing page:\n{cached_contents(url)}"]
        for p in picked:
            try:
                sections.append(f"## {p['type'].title()}:\n{cached_contents(p['url'])}")
                pages.append(p)
            except Exception:
                st.write(f"   ⚠️ Could not fetch {p['url']}")
        raw = "\n\n".join(sections)[:8_000]
        status.update(label="Context ready ✅", state="complete", expanded=False)

    system = (
        "You analyze several pages of a company's website and create a brochure about the company. "
        f"{BROCHURE_STYLES[style]} Include company culture, customers and careers/jobs if the information "
        f"is available. {MARKDOWN_RULE}"
    )
    user = f"Here are the pages of the company website ({url}). Create the brochure.\n\n{raw}"
    return raw, pages, stream_chat(client, model, system, user, temperature)


# --------------------------------------------------------------------------------------
# Result rendering
# --------------------------------------------------------------------------------------


def render_extras(tab_raw, tab_pages, result):
    with tab_raw:
        st.caption("Exactly what the LLM saw (after scraping & cleaning).")
        st.code(result["raw"], language="markdown", wrap_lines=True)
    with tab_pages:
        for p in result["pages"]:
            st.markdown(f"- **{p['type'].title()}** — [{p['url']}]({p['url']})")


def render_metrics(result):
    words = len(result["text"].split())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⏱️ Time", f"{result['elapsed']:.1f}s")
    c2.metric("📝 Words", f"{words:,}")
    c3.metric("📄 Pages used", len(result["pages"]))
    c4.metric("🤖 Model", result["model"].split("/")[-1][:18])


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ LLM Settings")

    default_provider = get_secret("LLM_PROVIDER", "Ollama (local)")
    provider_names = list(PROVIDERS)
    provider = st.selectbox(
        "Provider",
        provider_names,
        index=(
            provider_names.index(default_provider)
            if default_provider in provider_names
            else 0
        ),
    )
    preset = PROVIDERS[provider]

    base_url = st.text_input(
        "Base URL",
        value=(
            get_secret("LLM_BASE_URL", preset["base_url"])
            if provider == default_provider
            else preset["base_url"]
        ),
        key=f"base_{provider}",
    )
    model = st.text_input(
        "Model",
        value=(
            get_secret("LLM_MODEL", preset["model"])
            if provider == default_provider
            else preset["model"]
        ),
        key=f"model_{provider}",
    )
    api_key = ""
    server_key = get_secret("LLM_API_KEY", "")
    server_base = get_secret(
        "LLM_BASE_URL", PROVIDERS.get(default_provider, {}).get("base_url", "")
    )
    # The server-side key may only be used for the provider + URL the owner configured,
    # so a visitor can never redirect it to an endpoint they control.
    can_use_server_key = (
        bool(server_key) and provider == default_provider and base_url == server_base
    )
    if preset["needs_key"]:
        api_key = st.text_input(
            "API key",
            type="password",
            placeholder=(
                "Leave blank to use the app's key"
                if can_use_server_key
                else "Paste your API key"
            ),
            help="Used only for your session and never stored. Secrets are never sent to your browser.",
            key=f"key_{provider}",
        )
        if not api_key and can_use_server_key:
            api_key = server_key
            st.caption("🔒 Using the app's built-in key (rate limits may apply).")

    temperature = st.slider("Temperature", 0.0, 1.2, 0.7, 0.1)

    st.divider()
    st.subheader("📘 About")
    st.caption(
        "A two-mode web-analysis app: a tone-controllable **summarizer** and a multi-step "
        "**brochure generator** where the LLM first decides which pages to scrape, then writes "
        "the brochure from them. Responses stream live."
    )
    st.caption("Built with Streamlit · OpenAI SDK · BeautifulSoup")

# --------------------------------------------------------------------------------------
# Main UI
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>🌐 Web Insight AI</h1>
      <p>Paste any website. Get a summary or a full company brochure — powered by LLMs.</p>
      <span class="pill">Streaming</span><span class="pill">Multi-step LLM pipeline</span>
      <span class="pill">Structured JSON output</span><span class="pill">Ollama · Groq · OpenAI</span>
    </div>
    """,
    unsafe_allow_html=True,
)

mode = (
    st.segmented_control(
        "Mode", ["📝 Summarizer", "📘 Brochure Generator"], default="📝 Summarizer"
    )
    or "📝 Summarizer"
)

col_in, col_opt = st.columns([3, 2])
with col_in:
    url_input = st.text_input("Website URL", placeholder="https://anthropic.com")
with col_opt:
    if mode == "📝 Summarizer":
        tone = st.selectbox("Tone", list(SUMMARY_TONES))
    else:
        c_a, c_b = st.columns(2)
        style = c_a.selectbox("Style", list(BROCHURE_STYLES))
        max_pages = c_b.slider("Extra pages", 1, 6, 3)

st.caption("Try:")
ex_cols = st.columns(4)
examples = [
    "https://anthropic.com",
    "https://huggingface.co",
    "https://streamlit.io",
    "https://edwarddonner.com",
]
for col, ex in zip(ex_cols, examples):
    if col.button(ex.replace("https://", ""), use_container_width=True, key=f"ex_{ex}"):
        st.session_state["quick_url"] = ex

if "quick_url" in st.session_state and not url_input:
    url_input = st.session_state["quick_url"]
    st.info(f"Using example: {url_input}")

go = st.button("✨ Generate", type="primary", use_container_width=True)

# --------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------
if go:
    url = normalize_url(url_input)
    if not url:
        st.warning("Please enter a valid website URL.")
        st.stop()
    if preset["needs_key"] and not api_key:
        st.warning("This provider needs an API key — add it in the sidebar.")
        st.stop()
    if not model or not base_url:
        st.warning("Please fill in the base URL and model in the sidebar.")
        st.stop()

    client = OpenAI(base_url=base_url, api_key=api_key or "ollama", timeout=120)
    start = time.time()

    try:
        if mode == "📝 Summarizer":
            raw, pages, gen = run_summary(client, model, url, tone, temperature)
        else:
            raw, pages, gen = run_brochure(
                client, model, url, style, temperature, max_pages
            )

        tab_out, tab_raw, tab_pages = st.tabs(
            ["📄 Result", "🧾 Scraped context", "🔗 Pages used"]
        )
        with tab_out:
            text = st.write_stream(gen)

        result = {
            "mode": mode,
            "url": url,
            "text": text,
            "raw": raw,
            "pages": pages,
            "elapsed": time.time() - start,
            "model": model,
        }
        st.session_state["result"] = result

        with tab_out:
            st.divider()
            render_metrics(result)
            st.download_button(
                "⬇️ Download as Markdown",
                text,
                file_name="web_insight.md",
                mime="text/markdown",
            )
        render_extras(tab_raw, tab_pages, result)

    except openai.APIConnectionError:
        st.error(
            f"Couldn't reach the LLM at `{base_url}`. "
            + (
                "Is Ollama running? Try `ollama serve`. "
                if provider.startswith("Ollama")
                else ""
            )
            + "Note: a deployed app can't reach `localhost` — use a hosted provider there."
        )
    except openai.AuthenticationError:
        st.error("Authentication failed — check your API key.")
    except openai.NotFoundError:
        hint = f" Try `ollama pull {model}`." if provider.startswith("Ollama") else ""
        st.error(f"Model `{model}` not found on this provider.{hint}")
    except openai.RateLimitError:
        st.error("Rate limit hit — wait a moment or switch model/provider.")
    except requests.exceptions.RequestException as e:
        st.error(f"Couldn't fetch that website: {e}")
    except Exception as e:
        st.error(f"Something went wrong: {type(e).__name__}: {e}")

elif "result" in st.session_state:
    # Re-render the last result (e.g. after clicking the download button)
    result = st.session_state["result"]
    tab_out, tab_raw, tab_pages = st.tabs(
        ["📄 Result", "🧾 Scraped context", "🔗 Pages used"]
    )
    with tab_out:
        st.markdown(result["text"])
        st.divider()
        render_metrics(result)
        st.download_button(
            "⬇️ Download as Markdown",
            result["text"],
            file_name="web_insight.md",
            mime="text/markdown",
        )
    render_extras(tab_raw, tab_pages, result)
