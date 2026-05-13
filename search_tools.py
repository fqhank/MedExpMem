"""
SearchTools: Standalone class for Wikipedia and PubMed search functionality.

This module provides a simple interface to search Wikipedia and PubMed databases
using LangChain's community tools. Results are returned as raw text.
"""

import logging
from typing import Optional

from langchain_community.tools import WikipediaQueryRun, PubmedQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, PubMedAPIWrapper


class SearchTools:
    """
    A utility class for searching Wikipedia and PubMed databases.

    This class provides methods to search Wikipedia articles and PubMed research papers
    using LangChain's community tools. Results are returned as raw text content.

    Attributes:
        verbose (bool): Enable verbose logging for debugging purposes.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the SearchTools instance.

        Args:
            verbose (bool): If True, enable verbose logging. Defaults to False.
        """
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

        if verbose:
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(
                    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                )
                self.logger.addHandler(handler)

    def search_wikipedia(
        self,
        query: str,
        top_k_results: int = 3,
        doc_content_chars_max: int = 10000
    ) -> str:
        """
        Search Wikipedia for articles matching the query.

        Args:
            query (str): The search query string. Must be non-empty.
            top_k_results (int): Maximum number of results to return. Defaults to 3.
            doc_content_chars_max (int): Maximum characters per document. Defaults to 10000.

        Returns:
            str: Raw search results from Wikipedia as a formatted string.
        """
        
        # Validate input
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if self.verbose:
            self.logger.info(f"Searching Wikipedia for: {query}")
            self.logger.info(f"Parameters: top_k={top_k_results}, max_chars={doc_content_chars_max}")

        try:
            # Create Wikipedia API wrapper with specified parameters
            api_wrapper = WikipediaAPIWrapper(
                top_k_results=top_k_results,
                doc_content_chars_max=doc_content_chars_max
            )

            # Create and run the Wikipedia query tool
            tool = WikipediaQueryRun(api_wrapper=api_wrapper)
            result = tool.run(query)

            if self.verbose:
                self.logger.info(f"Wikipedia search completed, result length: {len(result)}")

            return result

        except Exception as e:
            error_msg = f"Wikipedia search failed for query '{query}': {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg) from e

    def search_pubmed(
        self,
        query: str,
        top_k_results: int = 3,
        doc_content_chars_max: int = 10000
    ) -> str:
        """
        Search PubMed for research papers matching the query.

        Args:
            query (str): The search query string. Must be non-empty.
            top_k_results (int): Maximum number of results to return. Defaults to 3.
            doc_content_chars_max (int): Maximum characters per document. Defaults to 10000.

        Returns:
            str: Raw search results from PubMed as a formatted string containing
                 paper titles, authors, and abstracts.
        """
        # Validate input
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if self.verbose:
            self.logger.info(f"Searching PubMed for: {query}")
            self.logger.info(f"Parameters: top_k={top_k_results}, max_chars={doc_content_chars_max}")

        try:
            # Create PubMed API wrapper with specified parameters
            api_wrapper = PubMedAPIWrapper(
                top_k_results=top_k_results,
                doc_content_chars_max=doc_content_chars_max
            )

            # Create and run the PubMed query tool
            tool = PubmedQueryRun(api_wrapper=api_wrapper)
            result = tool.run(query)

            if self.verbose:
                self.logger.info(f"PubMed search completed, result length: {len(result)}")

            print('materials get')
            return result

        except Exception as e:
            error_msg = f"PubMed search failed for query '{query}': {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg) from e
