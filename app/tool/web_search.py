import asyncio
from typing import List

from googlesearch import search as google_search
from baidusearch.baidusearch import search as baidu_search

from app.tool.base import BaseTool
from app.config import config


class WebSearch(BaseTool):
    name: str = "web_search"
    description: str = """Perform a web search and return a list of relevant links.
Use this tool when you need to find information on the web, get up-to-date data, or research specific topics.
The tool returns a list of URLs that match the search query.
"""
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "(required) The search query to submit to the search engine.",
            },
            "num_results": {
                "type": "integer",
                "description": "(optional) The number of search results to return. Default is 10.",
                "default": 10,
            },
        },
        "required": ["query"],
    }
    _search_engine: dict = {
        "google": google_search,
        "baidu": baidu_search,
    }

    async def execute(self, query: str, num_results: int = 10) -> List[str]:
        """
        Execute a Web search and return a list of URLs.

        Args:
            query (str): The search query to submit to the search engine.
            num_results (int, optional): The number of search results to return. Default is 10.

        Returns:
            List[str]: A list of URLs matching the search query.
        """
        # Run the search in a thread pool to prevent blocking
        loop = asyncio.get_event_loop()
        search_engine = self.get_search_engine()
        links = await loop.run_in_executor(
            None, lambda: list(search_engine(query, num_results=num_results))
        )

        return links
    
    def get_search_engine(self):
        """Determines the search engine to use based on the configuration."""
        if config.search_config is None:
            return google_search
        else:
            engine = config.search_config.engine.lower()
            return self._search_engine.get(engine, google_search)
