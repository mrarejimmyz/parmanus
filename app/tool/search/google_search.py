from googlesearch import search

from app.tool.search.base import SearchItem, WebSearchEngine


class GoogleSearchEngine(WebSearchEngine):
    def perform_search(self, query, num_results=10, *args, **kwargs):
        """
        Google search engine.

        Returns results formatted according to SearchItem model.
        """
        raw_results = search(query, num_results=num_results, advanced=True)

        results = []
        for i, item in enumerate(raw_results):
            if isinstance(item, str):
                # If it's just a URL
                results.append(
                    {"title": f"Google Result {i+1}", "url": item, "description": ""}
                )
            else:
                results.append(
                    SearchItem(
                        title=item.title, url=item.url, description=item.description
                    )
                )

        return results
