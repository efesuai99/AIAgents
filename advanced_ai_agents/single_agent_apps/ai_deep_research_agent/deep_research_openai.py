import asyncio
import os, json
from typing import Dict, Any

import requests
import streamlit as st
from agents import Agent, Runner
from agents import set_default_openai_key
from agents.tool import function_tool


# -----------------------
# Streamlit page
# -----------------------
st.set_page_config(page_title="OpenAI Deep Research Agent", page_icon="📘", layout="wide")
st.caption("Mode: REST-only Firecrawl client")


# -----------------------
# API keys (Secrets first, then sidebar)
# -----------------------
# Preload from Secrets
st.session_state.openai_api_key = st.secrets.get("OPENAI_API_KEY", st.session_state.get("openai_api_key", ""))
st.session_state.firecrawl_api_key = st.secrets.get("FIRECRAWL_API_KEY", st.session_state.get("firecrawl_api_key", ""))

# If OpenAI key is already present (via Secrets), register it with the Agents SDK now
if st.session_state.openai_api_key:
    set_default_openai_key(st.session_state.openai_api_key)

with st.sidebar:
    st.title("API Configuration")
    openai_api_key = st.text_input("OpenAI API Key", value=st.session_state.openai_api_key, type="password")
    firecrawl_api_key = st.text_input("Firecrawl API Key", value=st.session_state.firecrawl_api_key, type="password")

    # Save entries from sidebar
    if openai_api_key:
        st.session_state.openai_api_key = openai_api_key
        set_default_openai_key(openai_api_key)

    if firecrawl_api_key:
        st.session_state.firecrawl_api_key = firecrawl_api_key


# -----------------------
# UI
# -----------------------
st.title("📘 OpenAI Deep Research Agent")
st.markdown("This app performs deep research using Firecrawl **via REST** and then enhances the report with OpenAI Agents.")

research_topic = st.text_input("Enter your research topic:", placeholder="e.g., Latest developments in AI")


# -----------------------
# Firecrawl Deep Research (REST)
# -----------------------
@function_tool
async def deep_research(query: str, max_depth: int, time_limit: int, max_urls: int) -> Dict[str, Any]:
    """
    Call Firecrawl Deep Research REST endpoint directly. No SDK.
    Returns:
      {
        "success": bool,
        "final_analysis": str,
        "sources_count": int,
        "sources": list
      }
    """
    api_key = st.session_state.get("firecrawl_api_key") or os.getenv("FIRECRAWL_API_KEY", "")
    if not api_key:
        st.error("Missing FIRECRAWL_API_KEY")
        return {"error": "Missing FIRECRAWL_API_KEY", "success": False}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "maxDepth": max_depth,
        "timeLimit": time_limit,
        "maxUrls": max_urls,
    }

    try:
        with st.spinner("Performing deep research..."):
            resp = requests.post(
                "https://api.firecrawl.dev/v1/deep-research",
                headers=headers,
                data=json.dumps(payload),
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})

        return {
            "success": True,
            "final_analysis": data.get("finalAnalysis", ""),
            "sources_count": len(data.get("sources", [])),
            "sources": data.get("sources", []),
        }

    except requests.HTTPError as e:
        msg = e.response.text if getattr(e, "response", None) is not None else str(e)
        st.error(f"Deep research HTTP error: {msg}")
        return {"error": msg, "success": False}
    except Exception as e:
        st.error(f"Deep research error: {str(e)}")
        return {"error": str(e), "success": False}


# -----------------------
# Agents
# -----------------------
research_agent = Agent(
    name="research_agent",
    instructions=(
        "You are a research assistant that performs deep web research on any topic.\n"
        "1) Call the deep_research tool with: max_depth=3, time_limit=180, max_urls=10.\n"
        "2) Organize findings into a structured report.\n"
        "3) Include citations from the provided sources.\n"
        "4) Highlight key insights."
    ),
    tools=[deep_research],
)

elaboration_agent = Agent(
    name="elaboration_agent",
    instructions=(
        "You enhance research reports:\n"
        "- Add explanations, examples, and case studies.\n"
        "- Expand key points with context and implications.\n"
        "- Maintain structure and rigor, no fluff."
    ),
)


# -----------------------
# Orchestration
# -----------------------
async def run_research_process(topic: str) -> str:
    with st.spinner("Conducting initial research..."):
        research_result = await Runner.run(research_agent, topic)
        initial_report = research_result.final_output

    with st.expander("View Initial Research Report"):
        st.markdown(initial_report)

    with st.spinner("Enhancing the report with additional information..."):
        elaboration_input = f"""
RESEARCH TOPIC: {topic}

INITIAL RESEARCH REPORT:
{initial_report}

Enhance this report with additional detail, examples, and deeper insights. Keep it factual and structured.
"""
        elaboration_result = await Runner.run(elaboration_agent, elaboration_input)
        enhanced_report = elaboration_result.final_output

    return enhanced_report


# -----------------------
# Run
# -----------------------
can_run = bool(st.session_state.openai_api_key and st.session_state.firecrawl_api_key and research_topic)
if st.button("Start Research", disabled=not can_run):
    if not st.session_state.openai_api_key or not st.session_state.firecrawl_api_key:
        st.warning("Please enter both API keys in the sidebar.")
    elif not research_topic:
        st.warning("Please enter a research topic.")
    else:
        try:
            report_placeholder = st.empty()
            enhanced_report = asyncio.run(run_research_process(research_topic))
            report_placeholder.markdown("## Enhanced Research Report")
            report_placeholder.markdown(enhanced_report)
            st.download_button(
                "Download Report",
                enhanced_report,
                file_name=f"{research_topic.replace(' ', '_')}_report.md",
                mime="text/markdown",
            )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

st.markdown("---")
st.markdown("Powered by OpenAI Agents SDK + Firecrawl REST")
