import re
import uuid
import logging
from typing import Generator, Dict, Any, Tuple, List

from streamlit_utils.chat_engine import agent_loop, parse_thoughts

logger = logging.getLogger(__name__)

class ChatController:
    """
    Controller handling business logic for the Chat view.
    Orchestrates the LLM manager, parsing logic, and M2M data requests.
    """

    def __init__(self, llm_manager, project_manager):
        self.llm_manager = llm_manager
        self.project_manager = project_manager

    def clean_for_display(self, text: str) -> Tuple[str, List[Tuple[str, str]], List[Dict]]:
        """Strip internal tags and format thoughts for display."""
        thoughts, response = parse_thoughts(text)

        # Extract plot paths
        plots = re.findall(r'\[System: Plot saved to (.*?) \| CSV: (.*?)\]', response)
        old_plots = re.findall(r'\[System: Plot saved to ([^|]*?)\]', response)
        for p in old_plots:
            plots.append((p.strip(), ""))
            
        maps_extracted = re.findall(r'\[System: Map rendered at lat=(.*?), lon=(.*?), title=(.*?)\]', response)
        maps = [{"lat": float(lat), "lon": float(lon), "title": title} for lat, lon, title in maps_extracted]

        # Strip all tags and system messages
        for tag in [r'<m2m_request\s+[^>]*/\s*>', r'<zarr_request\s+[^>]*/\s*>',
                   r'<update_view\s+[^/>]+/?>',
                   r'<search_instruments\s+[^>]*/\s*>', r'<render_map\s+[^>]*/\s*>',
                   r'\[System:.*?\]']:
            response = re.sub(tag, '', response)

        response = response.strip()
        formatted = f"<div class=\"thinking-block\">{thoughts}</div>\n\n{response}" if thoughts else response
            
        return formatted, plots, maps

    def process_message_stream(
        self, prompt: str, project_id: str, session_messages: list
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Executes the agent loop and yields normalized events for the view to render.
        """
        accumulated = ""
        m2m_request_params = None
        m2m_accumulated = ""
        
        for event in agent_loop(
            message=prompt, 
            project_id=project_id, 
            llm_manager=self.llm_manager, 
            session_messages=session_messages
        ):
            if event["type"] == "token":
                accumulated += event["text"]
                display, plots, maps = self.clean_for_display(accumulated)
                yield {
                    "type": "update",
                    "display_text": display,
                    "plots": plots,
                    "maps": maps,
                    "raw_accumulated": accumulated
                }
            elif event["type"] == "m2m_tag_start":
                m2m_accumulated = ""
                yield {"type": "m2m_start"}
            elif event["type"] == "m2m_tag_token":
                m2m_accumulated += event["text"]
            elif event["type"] == "m2m_tag_end":
                import xml.etree.ElementTree as ET
                try:
                    root = ET.fromstring(m2m_accumulated.strip())
                    params = {child.tag: child.text for child in root}
                    m2m_request_params = params
                    yield {"type": "m2m_ready", "params": params}
                except Exception as e:
                    logger.error(f"Error parsing m2m tags: {e}")
            elif event["type"] == "m2m_response":
                # Re-yield the raw text that the LLM generates after M2M
                accumulated += event.get("text", "")
                display, plots, maps = self.clean_for_display(accumulated)
                yield {
                    "type": "update",
                    "display_text": display,
                    "plots": plots,
                    "maps": maps,
                    "raw_accumulated": accumulated
                }
            elif event["type"] == "done":
                yield {"type": "done", "raw_accumulated": accumulated, "m2m_params": m2m_request_params}
