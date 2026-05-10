"""PubMed API client using NCBI E-utilities."""
import os
import time
import xml.etree.ElementTree as ET
from typing import List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from agent.state import Paper

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

NCBI_TOOL = "clinical_scout"
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "kuntal.beehiveai@gmail.com")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

# Rate limit: 3 req/s without key, 10/s with key
_RATE_DELAY = 0.11 if NCBI_API_KEY else 0.34


def _session() -> requests.Session:
    """Return a requests Session with retry logic."""
    sess = requests.Session()
    retry = Retry(total=3, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503])
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    return sess


def _base_params() -> dict:
    """Return common NCBI E-utilities parameters."""
    params = {"tool": NCBI_TOOL, "email": NCBI_EMAIL}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    return params


def search_pubmed(query: str, max_results: int = 20) -> List[str]:
    """Return list of PMIDs matching query."""
    params = {
        **_base_params(),
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    resp = _session().get(ESEARCH_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    pmids = data.get("esearchresult", {}).get("idlist", [])
    time.sleep(_RATE_DELAY)
    return pmids


def fetch_abstracts(pmids: List[str]) -> List[Paper]:
    """Fetch paper metadata and abstracts for a list of PMIDs."""
    papers: List[Paper] = []
    batch_size = 20

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i: i + batch_size]
        params = {
            **_base_params(),
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
            "rettype": "abstract",
        }
        resp = _session().get(EFETCH_URL, params=params, timeout=30)
        resp.raise_for_status()
        papers.extend(_parse_xml(resp.text))
        time.sleep(_RATE_DELAY)

    return [p for p in papers if p["abstract"].strip()]


def _parse_xml(xml_text: str) -> List[Paper]:
    """Parse PubMed XML response into a list of Paper objects."""
    papers: List[Paper] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return papers

    for article in root.findall(".//PubmedArticle"):
        try:
            papers.append(_parse_article(article))
        except (KeyError, AttributeError, AssertionError):
            continue

    return papers


def _parse_article(article: ET.Element) -> Paper:
    """Extract fields from a single PubmedArticle XML element."""
    medline = article.find("MedlineCitation")
    assert medline is not None

    pmid_el = medline.find("PMID")
    pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else ""

    art = medline.find("Article")
    assert art is not None

    title = _element_text(art.find("ArticleTitle"))
    abstract = _extract_abstract(art)
    authors = _extract_authors(art)

    journal = art.find("Journal")
    iso_el = journal.find("ISOAbbreviation") if journal is not None else None
    journal_name = iso_el.text.strip() if iso_el is not None and iso_el.text else ""

    year = _extract_year(medline, art)

    return Paper(
        pmid=pmid,
        title=title,
        abstract=abstract,
        authors=authors,
        year=year,
        journal=journal_name,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        sigma_score=None,
    )


def _element_text(el: Optional[ET.Element]) -> str:
    """Join all text nodes within an element."""
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def _extract_abstract(art: ET.Element) -> str:
    """Return abstract text, handling structured abstracts with section labels."""
    abstract_el = art.find("Abstract")
    if abstract_el is None:
        return ""
    parts = []
    for text_el in abstract_el.findall("AbstractText"):
        label = text_el.get("Label", "")
        text = "".join(text_el.itertext()).strip()
        if label:
            parts.append(f"{label}: {text}")
        elif text:
            parts.append(text)
    return " ".join(parts)


def _extract_authors(art: ET.Element) -> str:
    """Return first-three authors string, appending 'et al.' when needed."""
    author_list = art.find("AuthorList")
    if author_list is None:
        return "Unknown"
    authors = []
    for author in author_list.findall("Author"):
        last = author.findtext("LastName", "")
        fore = author.findtext("ForeName", "")
        name = f"{last} {fore}".strip() if fore else last
        if name:
            authors.append(name)
    if len(authors) > 3:
        return ", ".join(authors[:3]) + " et al."
    return ", ".join(authors)


def _extract_year(medline: ET.Element, art: ET.Element) -> str:
    """Extract publication year from PubDate, ArticleDate, or PubMedPubDate."""
    journal = art.find("Journal")
    if journal is not None:
        pub_date = journal.find(".//PubDate")
        if pub_date is not None:
            year = pub_date.findtext("Year", "")
            if year:
                return year
            medline_date = pub_date.findtext("MedlineDate", "")
            if medline_date:
                return medline_date[:4]

    for ad in art.findall("ArticleDate"):
        year = ad.findtext("Year", "")
        if year:
            return year

    pubmed_data = medline.getparent() if hasattr(medline, "getparent") else None
    if pubmed_data is not None:
        for pub_status in pubmed_data.findall(".//PubMedPubDate"):
            if pub_status.get("PubStatus") == "pubmed":
                year = pub_status.findtext("Year", "")
                if year:
                    return year

    return "n.d."
