"""
Chat engine — the agent loop.

Runs synchronously (Streamlit is single-threaded). Handles:
  - Multi-turn prompt construction with context window management
  - Streaming LLM responses
  - <search_instruments> tag parsing and execution
  - <m2m_request> tag parsing
  - <update_view> tag parsing
  - Gemma thought/channel token stripping
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

WORKSPACE_DIR = Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────────────────────
# Thought Parsing (ported from chat_panel.py)
# ──────────────────────────────────────────────────────────────

def parse_thoughts(text: str) -> Tuple[str, str]:
    """
    Strips Gemma 4 channel/thought tokens from accumulated text.
    Returns: (thoughts_text, response_text)
    """
    start_pattern = r"<\|?channel\|?>\s*(?:thought|reasoning)\b\s*"
    match_start = re.search(start_pattern, text, re.IGNORECASE)

    tags = [
        "<|channel>", "<channel|>", "<|channel>text",
        "<|channel>response", "<|thought|>", "<|think|>",
        "<think>", "</think>",
    ]

    if not match_start:
        cleaned = text
        for tag in tags:
            cleaned = cleaned.replace(tag, "")
        return "", cleaned

    start_idx = match_start.start()
    start_end_idx = match_start.end()
    prefix = text[:start_idx]
    remaining = text[start_end_idx:]

    close_pattern = r"<channel\|>|<\|?channel\|?>\s*(?:text|response)\b\s*"
    match_close = re.search(close_pattern, remaining, re.IGNORECASE)

    if match_close:
        thoughts = remaining[: match_close.start()].strip()
        response = remaining[match_close.end():]
        for tag in tags:
            thoughts = thoughts.replace(tag, "")
            response = response.replace(tag, "")
        return thoughts.strip(), prefix + response
    else:
        cleaned = remaining
        for tag in tags:
            cleaned = cleaned.replace(tag, "")
        return "", prefix + cleaned


# ──────────────────────────────────────────────────────────────
# Search Instruments Parsing
# ──────────────────────────────────────────────────────────────

def parse_search_instruments(text: str) -> Optional[Dict[str, str]]:
    """Parse <search_instruments .../> tag from LLM output.

    Accepts any combination of: subsite, node, sensor, method.
    All attributes are optional (omit to match all).
    Returns dict of provided attributes or None if no tag found.
    """
    pattern = r'<search_instruments\s+([^>]*)/\s*>'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    attrs_str = match.group(1)
    result = {}
    for key in ["subsite", "node", "sensor", "method"]:
        attr_match = re.search(rf'{key}\s*=\s*["\']([^"\']+)["\']', attrs_str)
        if attr_match:
            result[key] = attr_match.group(1)
    # Even if no attributes were provided (match all), return empty dict
    return result


def execute_search_instruments(params: Dict[str, str]) -> str:
    """Execute an instrument search and return formatted results.

    Calls the rca_instrumentation module directly. If the search results in
    3 or fewer unique sensors, it automatically fetches live metadata from
    the OOI M2M API to provide the exact available date ranges.
    """
    import sys
    import requests
    backend_dir = str(WORKSPACE_DIR / "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    from rag.rca_instrumentation import search_instrumentation
    from core.config import settings

    results = search_instrumentation(
        subsite=params.get("subsite"),
        node=params.get("node"),
        sensor=params.get("sensor"),
        method=params.get("method"),
    )

    if not results:
        return "No instruments found matching the search criteria."

    # Group by unique sensor (subsite, node, sensor)
    unique_sensors = {}
    for r in results:
        key = (r['subsite'], r['node'], r['sensor'])
        if key not in unique_sensors:
            unique_sensors[key] = []
        unique_sensors[key].append(r)

    # Fetch live metadata if scope is narrow
    live_metadata = {}
    fetched_live = False
    if len(unique_sensors) <= 3:
        fetched_live = True
        auth = (settings.OOI_USERNAME, settings.OOI_TOKEN)
        for (subsite, node, sensor) in unique_sensors.keys():
            url = f"https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv/{subsite}/{node}/{sensor}/metadata"
            try:
                resp = requests.get(url, auth=auth, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    # Map (method, stream) to (beginTime, endTime)
                    times_map = {}
                    for t in data.get("times", []):
                        method = t.get("method")
                        stream = t.get("stream")
                        if method and stream:
                            times_map[(method, stream)] = (t.get("beginTime"), t.get("endTime"))
                    live_metadata[(subsite, node, sensor)] = times_map
                else:
                    live_metadata[(subsite, node, sensor)] = "API_ERROR"
            except Exception as e:
                logger.warning(f"Failed to fetch metadata for {subsite}/{node}/{sensor}: {e}")
                live_metadata[(subsite, node, sensor)] = "API_ERROR"

    # Cloud Zarr fast-path availability — checked only for narrow (live) searches
    # so it stays bounded. Degrades to no-flag if s3fs/network/store unavailable.
    zarr_client = None
    zarr_seen = {}
    if fetched_live:
        from services.zarr_client import ZarrClient
        zarr_client = ZarrClient()

    def _zarr_available(subsite, node, sensor, method, stream) -> bool:
        if zarr_client is None:
            return False
        key = (subsite, node, sensor, method, stream)
        if key not in zarr_seen:
            zarr_seen[key] = zarr_client.dataset_exists(*key)
        return zarr_seen[key]

    # Format output
    output_lines = [f"Found {len(results)} matching instrument endpoint(s):\n"]

    for r in results:
        subsite = r['subsite']
        node = r['node']
        sensor = r['sensor']
        method = r['method']
        stream = r['stream']

        base_line = f"  - {subsite} / {node} / {sensor} | method: {method} | stream: {stream}"

        if fetched_live:
            meta = live_metadata.get((subsite, node, sensor))
            if meta == "API_ERROR":
                output_lines.append(f"{base_line} | [NO METADATA AVAILABLE - API ERROR]")
            elif isinstance(meta, dict):
                bounds = meta.get((method, stream))
                if bounds:
                    begin_dt, end_dt = bounds
                    line = f"{base_line} | 🟢 AVAILABLE: {begin_dt} to {end_dt}"
                    if _zarr_available(subsite, node, sensor, method, stream):
                        line += " | ⚡ CLOUD ZARR AVAILABLE (fast path — no wait)"
                    output_lines.append(line)
                else:
                    output_lines.append(f"{base_line} | 🔴 UNAVAILABLE (Deprecated on OOI)")
            else:
                output_lines.append(base_line)
        else:
            output_lines.append(base_line)

    if not fetched_live:
        output_lines.append(
            "\n⚠️ Displaying local database results only. Narrow your search (e.g., specify a subsite, node, and sensor) "
            "to automatically fetch live data availability dates."
        )

    return "\n".join(output_lines)


# ──────────────────────────────────────────────────────────────
# M2M Request Parsing
# ──────────────────────────────────────────────────────────────

def parse_m2m_request(text: str) -> Optional[Dict[str, str]]:
    """Parse <m2m_request .../> or <md2m_request .../> tag from LLM output. Returns dict or None."""
    pattern = r'<(?:m2m|md2m)_request\s+([^>]*)/\s*>'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    attrs_str = match.group(1)
    result = {}
    for key in ["subsite", "node", "sensor", "method", "stream", "begin_dt", "end_dt"]:
        attr_match = re.search(rf'{key}\s*=\s*["\']([^"\']+)["\']', attrs_str)
        if attr_match:
            result[key] = attr_match.group(1)
    if len(result) == 7:
        return result
    return None


def parse_zarr_request(text: str) -> Optional[Dict[str, str]]:
    """Parse <zarr_request .../> tag (cloud fast path). Same 7 attrs as m2m."""
    match = re.search(r'<zarr_request\s+([^>]*)/\s*>', text, re.DOTALL)
    if not match:
        return None
    attrs_str = match.group(1)
    result = {}
    for key in ["subsite", "node", "sensor", "method", "stream", "begin_dt", "end_dt"]:
        attr_match = re.search(rf'{key}\s*=\s*["\']([^"\']+)["\']', attrs_str)
        if attr_match:
            result[key] = attr_match.group(1)
    if len(result) == 7:
        return result
    return None


# ──────────────────────────────────────────────────────────────
# Update View Parsing
# ──────────────────────────────────────────────────────────────

def parse_update_view(text: str) -> Optional[Dict[str, str]]:
    """Parse <update_view .../> tag from LLM output."""
    match = re.search(r'<update_view\s+((?:[^>"\']|"[^"]*"|\'[^\']*\')*?)\s*/?>',text, re.DOTALL)
    if not match:
        return None

    tag_content = match.group(1)
    result = {}

    plot_match = re.search(r'plot=["\']([^"\']+)["\']', tag_content)
    dataset_match = re.search(r'dataset=["\']([^"\']+)["\']', tag_content)
    flow_match = re.search(r'flowchart=["\']([^"\']+)["\']', tag_content)

    if plot_match:
        result["plot"] = plot_match.group(1)
    if dataset_match:
        result["dataset"] = dataset_match.group(1)
    if flow_match:
        result["flowchart"] = flow_match.group(1)

    return result if result else None

# ──────────────────────────────────────────────────────────────
# Generate Plot Parsing
# ──────────────────────────────────────────────────────────────

def parse_plot_requests(text: str) -> List[Dict[str, str]]:
    """Parse all <generate_plot .../> tags from LLM output."""
    results = []
    for match in re.finditer(r'<generate_plot\s+([^>]*)/\s*>', text, re.DOTALL):
        attrs_str = match.group(1)
        result = {}
        for key in ["dataset", "x_col", "y_col", "title", "color_col", "color", "plot_type", "invert_y", "labels"]:
            attr_match = re.search(rf'{key}\s*=\s*["\']([^"\']+)["\']', attrs_str)
            if attr_match:
                result[key] = attr_match.group(1)
        if "dataset" in result and "x_col" in result and "y_col" in result:
            results.append(result)
    return results

def parse_map_requests(text: str) -> List[Dict[str, str]]:
    """Parse all <render_map .../> tags from LLM output."""
    results = []
    for match in re.finditer(r'<render_map\s+([^>]*)/\s*>', text, re.DOTALL):
        attrs_str = match.group(1)
        result = {}
        for key in ["lat", "lon", "title"]:
            attr_match = re.search(rf'{key}\s*=\s*["\']([^"\']+)["\']', attrs_str)
            if attr_match:
                result[key] = attr_match.group(1)
        if "lat" in result and "lon" in result:
            results.append(result)
    return results
# ──────────────────────────────────────────────────────────────
# Agent Loop
# ──────────────────────────────────────────────────────────────

def build_prompt(turns: List[Dict[str, str]], system_prompt: str) -> str:
    """Build a Gemma-formatted prompt from conversation turns."""
    prompt_parts = [
        f"<start_of_turn>user\n[System instructions]\n{system_prompt}\n\n"
    ]

    for i, turn in enumerate(turns):
        role = turn["role"]
        content = turn["content"]
        if i == 0:
            prompt_parts.append(f"{content}<end_of_turn>\n")
        else:
            prompt_parts.append(f"<start_of_turn>{role}\n{content}<end_of_turn>\n")

    prompt_parts.append("<start_of_turn>model\n")
    return "".join(prompt_parts)


def agent_loop(
    message: str,
    project_id: str,
    llm_manager,
    session_messages: Optional[List[Dict[str, Any]]] = None,
) -> Generator[dict, None, None]:
    """
    The core agent loop. Yields dicts describing events.
    """
    from core.llm_manager import get_system_prompt
    from state.project_manager import project_manager

    purpose = project_manager.get_project_purpose(project_id)
    system_prompt = get_system_prompt(purpose)

    turns = []
    if session_messages:
        for msg in session_messages:
            role = msg.get("role")
            content = msg.get("content", "") or msg.get("text", "")
            if not content:
                continue
            
            # Map roles for Gemma
            if role in ("copilot", "assistant"):
                role = "model"
            elif role == "user":
                role = "user"
            else:
                continue
                
            turns.append({"role": role, "content": content})
            
        # Ensure the current message is at the end if it's not already there
        if not turns or turns[-1]["content"] != message:
            turns.append({"role": "user", "content": message})
    else:
        turns = [{"role": "user", "content": message}]

    max_loops = 5

    for loop_idx in range(max_loops):
        # ── Build prompt with context window management ──
        while True:
            formatted_prompt = build_prompt(turns, system_prompt)

            try:
                tokens = llm_manager.tokenize(formatted_prompt)
                token_count = len(tokens)
            except Exception:
                token_count = len(formatted_prompt) // 4

            if token_count <= 80000 or len(turns) <= 3:
                break


            logger.warning(f"Prompt tokens ({token_count}) approaching limit. Trimming oldest turn pair.")
            turns.pop(1)
            turns.pop(1)

        logger.info(f"Agent loop {loop_idx}, tokens: {token_count}")

        # ── Stream LLM response ──
        stream = llm_manager.generate_response_stream(formatted_prompt, is_raw=True)

        accumulated_text = ""
        for chunk in stream:
            yield {"type": "token", "text": chunk}
            accumulated_text += chunk

        # ── Parse <search_instruments> ──
        search_params = parse_search_instruments(accumulated_text)
        if search_params is not None:
            yield {"type": "system", "text": "Searching instrument database..."}
            search_result = execute_search_instruments(search_params)
            yield {"type": "search_results", "text": search_result}

            # Feed results back into the conversation and continue the loop
            turns.append({"role": "model", "content": accumulated_text})
            turns.append({
                "role": "user",
                "content": f"Observation (Instrument Search Results):\n{search_result}"
            })
            continue

        # ── Parse <zarr_request> (cloud fast path) ──
        if "/>" in accumulated_text and re.search(r'<zarr_request', accumulated_text):
            zarr_params = parse_zarr_request(accumulated_text)
            if zarr_params:
                yield {"type": "zarr_request", "params": zarr_params, "accumulated": accumulated_text}
                return

        # ── Parse <m2m_request> ──
        m2m_params = None
        if "/>" in accumulated_text and re.search(r'<(?:m2m|md2m)_request', accumulated_text):
            m2m_params = parse_m2m_request(accumulated_text)

        if m2m_params:
            yield {"type": "m2m_request", "params": m2m_params, "accumulated": accumulated_text}
            # The caller must handle approval and send back a continuation.
            # For now, we break — the caller re-invokes the loop with the observation.
            return

        # ── Parse <update_view> ──
        view_update = parse_update_view(accumulated_text)
        if view_update:
            yield {"type": "update_view", "params": view_update}

        # ── Parse <generate_plot> ──
        plot_requests = parse_plot_requests(accumulated_text)
        for req in plot_requests:
            yield {"type": "generate_plot", "params": req}

        # ── Parse <render_map> ──
        map_requests = parse_map_requests(accumulated_text)
        for req in map_requests:
            yield {"type": "render_map", "params": req}

        # No action tags — done
        break

    yield {"type": "done"}


def agent_loop_with_m2m_continuation(
    message: str,
    project_id: str,
    llm_manager,
    m2m_observation: str,
    prior_accumulated: str,
    session_messages: Optional[List[Dict[str, Any]]] = None,
) -> Generator[dict, None, None]:
    """
    Continue the agent loop after an M2M approval/rejection.
    """
    from core.llm_manager import get_system_prompt
    from state.project_manager import project_manager

    purpose = project_manager.get_project_purpose(project_id)
    system_prompt = get_system_prompt(purpose)

    turns = []
    if session_messages:
        for msg in session_messages:
            role = msg.get("role")
            content = msg.get("content", "") or msg.get("text", "")
            if not content:
                continue
            if role in ("copilot", "assistant"):
                role = "model"
            elif role == "user":
                role = "user"
            else:
                continue
            turns.append({"role": role, "content": content})
            
        # The user's original message might already be the last message.
        # But we need to ensure the observation is inserted properly.
        # Actually, in Streamlit, M2M handling injects after the assistant.
        # So we append the m2m_observation as the latest user turn.
        turns.append({"role": "user", "content": f"Observation:\n{m2m_observation}"})
    else:
        turns = [
            {"role": "user", "content": message},
            {"role": "model", "content": prior_accumulated},
            {"role": "user", "content": f"Observation:\n{m2m_observation}"},
        ]

    max_loops = 4
    for loop_idx in range(max_loops):
        while True:
            formatted_prompt = build_prompt(turns, system_prompt)
            try:
                tokens = llm_manager.tokenize(formatted_prompt)
                token_count = len(tokens)
            except Exception:
                token_count = len(formatted_prompt) // 4

            if token_count <= 80000 or len(turns) <= 3:
                break
            turns.pop(1)
            turns.pop(1)

        stream = llm_manager.generate_response_stream(formatted_prompt, is_raw=True)
        accumulated_text = ""
        for chunk in stream:
            yield {"type": "token", "text": chunk}
            accumulated_text += chunk

        # ── Parse <search_instruments> ──
        search_params = parse_search_instruments(accumulated_text)
        if search_params is not None:
            yield {"type": "system", "text": "Searching instrument database..."}
            search_result = execute_search_instruments(search_params)
            yield {"type": "search_results", "text": search_result}

            turns.append({"role": "model", "content": accumulated_text})
            turns.append({
                "role": "user",
                "content": f"Observation (Instrument Search Results):\n{search_result}"
            })
            continue

        # ── Parse <update_view> ──
        view_update = parse_update_view(accumulated_text)
        if view_update:
            yield {"type": "update_view", "params": view_update}

        # ── Parse <generate_plot> ──
        plot_requests = parse_plot_requests(accumulated_text)
        for req in plot_requests:
            yield {"type": "generate_plot", "params": req}

        # No action tags — done
        break

    yield {"type": "done"}
