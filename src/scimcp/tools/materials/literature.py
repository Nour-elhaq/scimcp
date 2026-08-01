"""arXiv literature search for materials science papers.

Searches arXiv for papers by keywords, authors, or categories.
Uses the arXiv API (no API key required).
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any


def search_arxiv(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
    sort_order: str = "descending",
    category: str = "",
) -> dict[str, Any]:
    """Search arXiv for scientific papers.

    Args:
        query: Search query (e.g. 'MXene DFT band structure').
        max_results: Maximum number of results (default 10).
        sort_by: Sort by 'relevance', 'lastUpdatedDate', or 'submittedDate'.
        sort_order: 'ascending' or 'descending'.
        category: arXiv category filter (e.g. 'cond-mat.mtrl-sci').

    Returns:
        Dictionary with list of papers and metadata.
    """
    # Build query
    search_query = query
    if category:
        search_query = f"cat:{category} AND {query}"

    # URL encode
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }

    url = f"http://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"

    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            xml_data = response.read().decode("utf-8")
    except Exception as e:
        return {"error": str(e), "papers": []}

    # Parse XML
    root = ET.fromstring(xml_data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    papers = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns)
        summary = entry.find("atom:summary", ns)
        published = entry.find("atom:published", ns)
        authors = entry.findall("atom:author", ns)
        links = entry.findall("atom:link", ns)

        paper = {
            "title": title.text.strip().replace("\n", " ") if title is not None else "",
            "abstract": summary.text.strip().replace("\n", " ")[:500] if summary is not None else "",
            "published": published.text if published is not None else "",
            "authors": [
                a.find("atom:name", ns).text
                for a in authors
                if a.find("atom:name", ns) is not None
            ],
            "arxiv_url": "",
            "pdf_url": "",
        }

        for link in links:
            if link.get("type") == "text/html":
                paper["arxiv_url"] = link.get("href", "")
            elif link.get("title") == "pdf":
                paper["pdf_url"] = link.get("href", "")

        papers.append(paper)

    return {
        "query": query,
        "total_results": len(papers),
        "papers": papers,
    }


def search_materials_science(
    topic: str,
    max_results: int = 10,
) -> dict[str, Any]:
    """Search for materials science papers on arXiv.

    Automatically filters to materials science categories.

    Args:
        topic: Search topic (e.g. 'perovskite solar cell', 'MXene battery').
        max_results: Maximum number of results.

    Returns:
        Search results filtered to materials science.
    """
    categories = [
        "cond-mat.mtrl-sci",  # Materials Science
        "cond-mat.mes-hall",  # Mesoscale and Nanoscale Physics
        "physics.chem-ph",    # Chemical Physics
    ]

    all_papers = []
    for cat in categories:
        result = search_arxiv(
            query=topic,
            max_results=max_results,
            category=cat,
        )
        all_papers.extend(result.get("papers", []))

    # Deduplicate by title
    seen_titles = set()
    unique_papers = []
    for paper in all_papers:
        if paper["title"] not in seen_titles:
            seen_titles.add(paper["title"])
            unique_papers.append(paper)

    return {
        "query": topic,
        "total_results": len(unique_papers[:max_results]),
        "papers": unique_papers[:max_results],
    }


def search_by_author(
    author: str,
    max_results: int = 10,
) -> dict[str, Any]:
    """Search arXiv papers by author name.

    Args:
        author: Author name (e.g. 'Naguib').
        max_results: Maximum number of results.

    Returns:
        Papers by the specified author.
    """
    query = f"au:{author}"
    return search_arxiv(query=query, max_results=max_results)


def get_paper_details(arxiv_id: str) -> dict[str, Any] | None:
    """Get detailed info for a specific arXiv paper.

    Args:
        arxiv_id: arXiv ID (e.g. '2301.12345').

    Returns:
        Paper details or None if not found.
    """
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"

    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            xml_data = response.read().decode("utf-8")
    except Exception:
        return None

    root = ET.fromstring(xml_data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    entry = root.find("atom:entry", ns)
    if entry is None:
        return None

    title = entry.find("atom:title", ns)
    summary = entry.find("atom:summary", ns)
    published = entry.find("atom:published", ns)
    authors = entry.findall("atom:author", ns)
    links = entry.findall("atom:link", ns)
    categories = entry.findall("atom:category", ns)

    paper = {
        "arxiv_id": arxiv_id,
        "title": title.text.strip().replace("\n", " ") if title is not None else "",
        "abstract": summary.text.strip().replace("\n", " ") if summary is not None else "",
        "published": published.text if published is not None else "",
        "authors": [
            a.find("atom:name", ns).text
            for a in authors
            if a.find("atom:name", ns) is not None
        ],
        "categories": [c.get("term", "") for c in categories],
        "arxiv_url": "",
        "pdf_url": "",
    }

    for link in links:
        if link.get("type") == "text/html":
            paper["arxiv_url"] = link.get("href", "")
        elif link.get("title") == "pdf":
            paper["pdf_url"] = link.get("href", "")

    return paper
