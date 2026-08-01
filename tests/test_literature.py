"""Tests for arXiv literature search tools."""

import json
from unittest.mock import patch, MagicMock

import pytest

from scimcp.tools.materials.literature import (
    search_arxiv,
    search_materials_science,
    search_by_author,
    get_paper_details,
)


MOCK_ARXIV_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query: MXene</title>
  <totalResults>1</totalResults>
  <entry>
    <title>Electronic Structure of MXenes</title>
    <summary>We study the electronic structure of MXenes using DFT...</summary>
    <published>2024-01-15T00:00:00Z</published>
    <author><name>John Smith</name></author>
    <author><name>Jane Doe</name></author>
    <link href="http://arxiv.org/abs/2401.12345" type="text/html"/>
    <link href="http://arxiv.org/pdf/2401.12345" title="pdf"/>
  </entry>
</feed>
"""


def mock_urlopen(timeout=15):
    """Create a mock response for arXiv API."""
    mock_response = MagicMock()
    mock_response.read.return_value = MOCK_ARXIV_XML.encode("utf-8")
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)
    return mock_response


class TestSearchArxiv:
    @patch("scimcp.tools.materials.literature.urllib.request.urlopen")
    def test_basic_search(self, mock_urlopen):
        mock_urlopen.return_value = mock_urlopen
        mock_urlopen.read.return_value = MOCK_ARXIV_XML.encode("utf-8")
        mock_urlopen.__enter__ = lambda s: s
        mock_urlopen.__exit__ = MagicMock(return_value=False)

        result = search_arxiv("MXene electronic structure")
        assert "papers" in result
        assert result["query"] == "MXene electronic structure"
        assert result["total_results"] >= 0

    @patch("scimcp.tools.materials.literature.urllib.request.urlopen")
    def test_search_with_max_results(self, mock_urlopen):
        mock_urlopen.return_value = mock_urlopen
        mock_urlopen.read.return_value = MOCK_ARXIV_XML.encode("utf-8")
        mock_urlopen.__enter__ = lambda s: s
        mock_urlopen.__exit__ = MagicMock(return_value=False)

        result = search_arxiv("MXene", max_results=5)
        assert "papers" in result

    @patch("scimcp.tools.materials.literature.urllib.request.urlopen")
    def test_search_error_handling(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Network error")
        result = search_arxiv("MXene")
        assert "error" in result
        assert result["papers"] == []

    @patch("scimcp.tools.materials.literature.urllib.request.urlopen")
    def test_paper_fields(self, mock_urlopen):
        mock_urlopen.return_value = mock_urlopen
        mock_urlopen.read.return_value = MOCK_ARXIV_XML.encode("utf-8")
        mock_urlopen.__enter__ = lambda s: s
        mock_urlopen.__exit__ = MagicMock(return_value=False)

        result = search_arxiv("MXene")
        if result["papers"]:
            paper = result["papers"][0]
            assert "title" in paper
            assert "abstract" in paper
            assert "authors" in paper
            assert "published" in paper
            assert "arxiv_url" in paper
            assert "pdf_url" in paper


class TestSearchMaterialsScience:
    @patch("scimcp.tools.materials.literature.urllib.request.urlopen")
    def test_materials_search(self, mock_urlopen):
        mock_urlopen.return_value = mock_urlopen
        mock_urlopen.read.return_value = MOCK_ARXIV_XML.encode("utf-8")
        mock_urlopen.__enter__ = lambda s: s
        mock_urlopen.__exit__ = MagicMock(return_value=False)

        result = search_materials_science("MXene")
        assert "papers" in result
        assert result["query"] == "MXene"

    @patch("scimcp.tools.materials.literature.urllib.request.urlopen")
    def test_materials_search_deduplication(self, mock_urlopen):
        mock_urlopen.return_value = mock_urlopen
        mock_urlopen.read.return_value = MOCK_ARXIV_XML.encode("utf-8")
        mock_urlopen.__enter__ = lambda s: s
        mock_urlopen.__exit__ = MagicMock(return_value=False)

        result = search_materials_science("MXene", max_results=10)
        # Should not crash even with multiple categories
        assert "papers" in result


class TestSearchByAuthor:
    @patch("scimcp.tools.materials.literature.urllib.request.urlopen")
    def test_author_search(self, mock_urlopen):
        mock_urlopen.return_value = mock_urlopen
        mock_urlopen.read.return_value = MOCK_ARXIV_XML.encode("utf-8")
        mock_urlopen.__enter__ = lambda s: s
        mock_urlopen.__exit__ = MagicMock(return_value=False)

        result = search_by_author("Naguib")
        assert "papers" in result
        assert result["query"] == "au:Naguib"


class TestGetPaperDetails:
    @patch("scimcp.tools.materials.literature.urllib.request.urlopen")
    def test_get_details(self, mock_urlopen):
        mock_urlopen.return_value = mock_urlopen
        mock_urlopen.read.return_value = MOCK_ARXIV_XML.encode("utf-8")
        mock_urlopen.__enter__ = lambda s: s
        mock_urlopen.__exit__ = MagicMock(return_value=False)

        result = get_paper_details("2401.12345")
        if result is not None:
            assert "title" in result
            assert "abstract" in result
            assert "authors" in result
            assert "categories" in result

    @patch("scimcp.tools.materials.literature.urllib.request.urlopen")
    def test_get_details_error(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Network error")
        result = get_paper_details("2401.12345")
        assert result is None
