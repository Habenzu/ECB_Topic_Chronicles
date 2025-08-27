from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from loguru import logger
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from dateutil import parser as dparse
import pandas as pd
from pathlib import Path
import requests
import time
import random
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from tqdm import tqdm
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union
import requests
from bs4 import BeautifulSoup, Tag
import json
from pathlib import Path
from tqdm import tqdm
import polars as pl
import ast


FILE_DIR = Path(__file__).resolve().parent
SCRIPT_NAME = Path(__file__).name

# =============================================================================
# 1. Scrape Publications Date, Authors and URLs to the HTMLs
# - Page to scrape: https://www.ecb.europa.eu/press/pubbydate/html/index.en.html?
# - Applies filters like topic or year to scrape those
# - per filter value a csv is created which has the information 
#   date,category,title,url,authors,filter_type,filter_value
# =============================================================================

BASE_URL = "https://www.ecb.europa.eu"
TOPICS = ["Accountability","Asset purchase programme (APP)","Bank failure","Bank profitability","Bank resolution","Banking regulation","Banking sector","Banking supervision","Banking union","Banknotes and coins","Basel III","Benchmark rates","Bitcoin","Brexit","Capital key","Capital markets union","Central bank digital currencies (CBDC)","Central bank independence","Central banking","Central counterparties (CCPs)","Climate change","Collateral","Communication","Coronavirus","Crypto-assets","Currencies","Cyber resilience","Deposit facility rate","Digital euro","Digitalisation","Distributed ledger technology (DLT)","Diversity and inclusion","Economic and Monetary Union (EMU)","Economic development","Emergency liquidity assistance (ELA)","Euro","Euro area","Euro overnight index average (EONIA)","Euro short-term rate (€STR)","European integration","Excess reserves","Exchange rates","Financial assets","Financial crisis","Financial integration","Financial market infrastructures","Financial markets","Financial stability","Fintech","Fiscal policy","Forward guidance","Governance","Haircuts","History of the euro","Inflation","Innovation","Instant payments","Interest rates","International relations","International role of the euro","Key ECB interest rates","Labour market","Legal framework","Liquidity","Liquidity lines","Macroprudential policy","Main refinancing operations (MRO) rate","Marginal lending facility rate","Microprudential policy","Minimum reserve requirements","Monetary policy","Money","Non-performing loans","Outright Monetary Transactions (OMTs)","Pandemic emergency longer-term refinancing operations (PELTROs)","Pandemic emergency purchase programme (PEPP)","Payment systems","Policies","Price stability","Profits","Protectionism","Repo lines","Resilience","Risks","Rules and procedures","Russia-Ukraine war","Russian war against Ukraine","Sanctions","Securities","Statistics and data","Strategy review","Stress tests","Structural reforms","Swap lines","TARGET Instant Payment Settlement (TIPS)","TARGET2","Targeted longer-term refinancing operations (TLTROs)","Technology","Trade","Transmission Protection Instrument (TPI)","Two-tier system","Uncertainties",]
YEARS = [str(x) for x in list(range(1992, 2025+1))]

FILTERS = {
    'topic': TOPICS, 
    'year' : YEARS}

def _headless_chrome():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1200,2000")
    return webdriver.Chrome(options=opts)

def scroll_until_loaded(driver, timeout=10, max_idle_rounds=3, hard_cap_rounds=200):
    prev_count = len(driver.find_elements(By.CSS_SELECTOR, "dl > dt"))
    idle = 0
    rounds = 0
    while True:
        rounds += 1
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        try:
            WebDriverWait(driver, timeout).until(
                lambda d: len(d.find_elements(By.CSS_SELECTOR, "dl > dt")) > prev_count
            )
            prev_count = len(driver.find_elements(By.CSS_SELECTOR, "dl > dt"))
            idle = 0
        except:
            idle += 1
        if idle >= max_idle_rounds or rounds >= hard_cap_rounds:
            break

def parse_page(html: str):
    soup = BeautifulSoup(html, "html.parser")

    # Get the OUTER (top-level) <dl>. We take the first <dl> in the main content.
    dl = soup.select_one("dl")
    if dl is None:
        return []

    # Iterate top-level siblings: pair each <dt> with its immediate <dd>
    items = []
    node = dl.find("dt")
    while node:
        if node.name == "dt":
            dt_tag = node
            dd_tag = dt_tag.find_next_sibling("dd")
            if dd_tag and dd_tag.parent == dl:
                # Extract fields
                date_raw = dt_tag.get_text(strip=True)
                try:
                    date_iso = dparse.parse(date_raw, dayfirst=False).date().isoformat()
                except Exception:
                    date_iso = date_raw  # fallback to raw if parsing fails

                cat = dd_tag.select_one("div.category")
                category = (cat.get_text(strip=True) if cat else None)

                a = dd_tag.select_one("div.title a")
                title = a.get_text(strip=True) if a else None
                href = a["href"] if a and a.has_attr("href") else None
                url = urljoin(BASE_URL, href) if href else None

                # Authors (if present)
                author_nodes = dd_tag.select("div.authors li")
                authors = [li.get_text(strip=True) for li in author_nodes] if author_nodes else []

                items.append(
                    {
                        "date": date_iso,
                        "category": category,
                        "title": title,
                        "url": url,
                        "authors": authors,
                    }
                )
            node = dd_tag.find_next_sibling() if dd_tag else node.find_next_sibling()
        else:
            node = node.find_next_sibling()
    return items

def get_url(filter_type, value: str): 
    index_html = "https://www.ecb.europa.eu/press/pubbydate/html/index.en.html?"
    return f"{index_html}{filter_type}={value.replace(' ', '%20')}"

def scrape_ecb_pub(filter_type, filter_value, save_csv: str = None,) -> pd.DataFrame:
    driver = _headless_chrome()
    url = get_url(filter_type, filter_value)
    try:
        driver.get(url)
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, "dl > dt")))
        scroll_until_loaded(driver)
        data = parse_page(driver.page_source)
        df = pd.DataFrame(data, columns=["date", "category", "title", "url", "authors"])
        df['filter_type'] = filter_type
        df['filter_value'] = filter_value
        if save_csv:
            df.to_csv(save_csv, index=False)
        logger.success(f"Successfully scraped: {url}, rows: {df.shape}")
        return df
    except Exception as e: 
        logger.warning(f"Failed {filter_type} {filter_value}: {url}: {e}")
    finally:
        driver.quit()

# =============================================================================
# 2. Downloading all the HTMLs from the URLs scraped in 1.)
# =============================================================================

def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/124.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en,de;q=0.9",
    })
    retry = Retry(
        total=6,                # total retry attempts
        connect=3,              # connection retries
        read=3,                 # read retries
        status=6,               # status code retries
        backoff_factor=0.8,     # exponential backoff (respects Retry-After)
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"HEAD", "GET", "OPTIONS"},
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

def download_htmls(links:pd.DataFrame, data_dir:str, filter_type:str): 
    html_dir = data_dir / 'htmls'
    html_dir.mkdir(parents=True, exist_ok=True)
    overview_path = data_dir / f"{filter_type}_html_downloaded_overview.csv"

    if overview_path.exists(): 
        all_links_downloaded = pd.read_csv(overview_path)
        return all_links_downloaded
    else:
        session = make_session()

        rows, indices = [], []
        for index, _ in tqdm(links.iterrows(), total=links.shape[0], desc = "Download HTMLs"):
            row = _.to_dict()
            url = _.get("url")
            out_path = html_dir / f"{_.get('url_hash')}.html"

            html_downloaded = False
            err = None

            try:
                time.sleep(random.uniform(0.05, 0.25))
                r = session.get(url, timeout=(5, 20), allow_redirects=True)
                if r.status_code == 200:
                    out_path.write_text(r.text, encoding=r.encoding or "utf-8", errors="ignore")
                    html_downloaded = True
                else:
                    err = f"HTTP {r.status_code}"
            except requests.RequestException as e:
                err = str(e)[:300]

            row["html_downloaded"] = html_downloaded
            row["download_error"] = err

            indices.append(index)
            rows.append(row)


        all_links_downloaded = pd.DataFrame.from_records(rows, index = indices)
        all_links_downloaded.to_csv(overview_path)
        return all_links_downloaded

# =============================================================================
# 3. Parsing the downloaded HTMLs to extract the text
# =============================================================================

@dataclass
class IgnoreRules:
    block_selectors: List[str] = field(default_factory=lambda: [
        "script", "style", "noscript", "iframe", "svg",
        "details", "summary", ".accordion", ".collapsible", ".collapse",
        "[data-accordion]", "[data-collapsible]",
        ".footnote", ".footnotes", ".endnote", ".endnotes", ".fn", ".fn-list",
        ".references", ".reference-list", ".ref-list", ".bibliography", ".citations",
        "aside[role='note']", "aside.footnotes", "aside.endnotes",
        ".resources", "#resources",
        "figure", ".figure", "figcaption", "img", "picture", "[role='figure']", "[role='img']",
        ".ecb-media",
        "header", "footer", "nav", "aside",
        ".ecb-related", ".ecb-related-links", ".related", ".downloads",
        ".attachments", ".attachment", ".ecb-attachments", ".share", ".social",
        ".metadata", ".ecb-toolbar", ".toc", "#toc", ".table-of-contents", ".breadcrumbs",
        "table", ".table", ".data-table", ".datatable", ".ecb-table", ".table-wrapper",
        "[role='table']", "[role='grid']",
        ".chart", ".chart-container", ".chart-title",
    ])
    tablelike_keywords: Tuple[str, ...] = (
        "table", "data-table", "datatable", "ecb-table", "table-wrapper", "rt-table", "grid",
    )
    header_remove_regex: re.Pattern = re.compile(
        r"\b(Resources|Related|Downloads|Further reading|Attachments|See also|References|Footnotes?|Endnotes?|Notes?)\b",
        re.I,
    )
    caption_regex: re.Pattern = re.compile(
        r"^\s*((fig(?:ure)?|chart|table|box|exhibit|diagram|graph)\s*\d+|panel\s*[A-Z])\s*[:\.\-]\s+",
        re.I,
    )
    source_or_note_regex: re.Pattern = re.compile(r"^\s*(source|sources|note|notes)\s*[:\-]\s+", re.I)
    meta_prefixes: Tuple[str, ...] = (
        "Prepared by", "Published as part of", "JEL", "Keywords", "Key words",
        "Acknowledgements", "Acknowledgments", "Disclaimer",
    )
    drop_attr_keywords: Tuple[str, ...] = (
        "collapse", "collapsible", "accordion",
        "footnote", "endnote", "reference", "citation",
        "caption", "figure-title", "chart-title", "table-title",
        "source", "note",
    )
    inline_footnote_selectors: List[str] = field(default_factory=lambda: [
        "sup", "a.footnote-ref", "a[role='doc-noteref']",
    ])
    anchor_href_contains: Tuple[str, ...] = ("#footnote", "#fn", "#note")
    anchor_class_contains: Tuple[str, ...] = ("footnote", "endnote", "reference", "citation")
    anchor_id_contains: Tuple[str, ...] = ("footnote", "endnote", "reference", "citation")
    inline_inline_tags: Set[str] = field(default_factory=lambda: {"span", "small", "em", "i"})
    non_content_parents: Set[str] = field(default_factory=lambda: {"figure", "aside", "nav", "footer", "details", "summary", "picture"})
    non_content_roles_regex: re.Pattern = re.compile(r"^(figure|img|presentation|region)$", re.I)

DEFAULT_RULES = IgnoreRules()

# ---------------------------- IO ---------------------------------------------

def load_html(path: Union[str, Path]) -> str:
    """Load HTML from a local file path."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return p.read_text(encoding="utf-8", errors="ignore")

# ------------------------ helpers / heuristics --------------------------------

def _attrs_text(tag: Tag) -> str:
    try:
        classes = " ".join(tag.get("class") or [])
        return f"{tag.get('id') or ''} {classes}".strip().lower()
    except Exception:
        return ""

def _is_hidden_by_style_or_parent(tag: Tag) -> bool:
    def hidden(t: Tag) -> bool:
        if not isinstance(t, Tag):
            return False
        attrs = t.attrs or {}
        if "hidden" in attrs or attrs.get("aria-hidden") in ("true", "1"):
            return True
        style = (attrs.get("style") or "").replace(" ", "").lower()
        return "display:none" in style or "visibility:hidden" in style
    cur = tag
    while isinstance(cur, Tag):
        if hidden(cur):
            return True
        cur = cur.parent
    return False

def _is_table_like(tag: Tag, rules: IgnoreRules = DEFAULT_RULES) -> bool:
    if not isinstance(tag, Tag):
        return False
    if (tag.name or "").lower() in {"table", "thead", "tbody", "tr", "td", "th", "caption"}:
        return True
    attrs = tag.attrs or {}
    if (attrs.get("role") or "").lower() in {"table", "grid", "rowgroup", "row"}:
        return True
    if any(k in _attrs_text(tag) for k in rules.tablelike_keywords):
        return True
    style = (attrs.get("style") or "").replace(" ", "").lower()
    return "display:table" in style

# --------------------------- extraction ---------------------------------------

def _pick_main_container(soup: BeautifulSoup) -> Tag:
    candidates = [
        "article[itemprop='articleBody']",
        "article",
        "main",
        "div[role='main']",
        ".ecb-pressContent",
        ".ecb-content",
        "#content",
        "#mainContent",
        "section.ecb-content",
        "section#content",
        "div.content",
    ]
    best, best_count = None, -1
    for sel in candidates:
        for el in soup.select(sel):
            cnt = len(el.find_all("p"))
            if cnt > best_count:
                best, best_count = el, cnt
    return best or soup.body or soup

def _remove_unwanted(container: Tag, rules: IgnoreRules = DEFAULT_RULES) -> None:
    """Remove junk blocks (incl. MathJax/MathML) before text extraction."""
    # --- hard purge MathJax/MathML (covers v3 <mjx-*> and legacy .MathJax / KaTeX) ---
    # tag-based
    for el in list(container.find_all(["mjx-container", "mjx-math", "mjx-assistive-mml", "math"])):
        el.decompose()
    # any tag whose name starts with 'mjx-' (extra safety)
    for el in list(container.find_all(lambda t: isinstance(t, Tag) and (t.name or "").lower().startswith("mjx-"))):
        el.decompose()
    # class-based (legacy MathJax/KaTeX)
    for el in list(container.select(".MathJax, [class*='MathJax'], .katex, .katex-display")):
        el.decompose()

    # --- existing behaviour (unchanged) ---
    for sel in rules.block_selectors:
        for el in container.select(sel):
            el.decompose()

    for h in container.find_all(["h2", "h3", "h4", "h5"]):
        if rules.header_remove_regex.search(h.get_text(" ", strip=True)):
            for sib in list(h.next_siblings):
                if isinstance(sib, Tag) and sib.name in {"h2", "h3", "h4", "h5"}:
                    break
                if isinstance(sib, Tag):
                    sib.decompose()
            h.decompose()

    for el in list(container.find_all(True)):
        # remove opacity:0 as "hidden" (MathJax assistive markup sometimes uses this)
        attrs = el.attrs or {}
        style = (attrs.get("style") or "").replace(" ", "").lower()

        if _is_table_like(el, rules):
            el.decompose()
            continue

        attr_text = _attrs_text(el)
        role = (attrs.get("role") or "").lower()
        if ("hidden" in attrs) or (attrs.get("aria-hidden") in ("true", "1")) \
           or ("display:none" in style) or ("visibility:hidden" in style) or ("opacity:0" in style) \
           or (role in {"img", "presentation"}) \
           or any(k in attr_text for k in (*rules.drop_attr_keywords, "mathjax", "mjx", "katex")):
            el.decompose()

def _strip_inline_refs(p: Tag, rules: IgnoreRules = DEFAULT_RULES) -> None:
    for sel in rules.inline_footnote_selectors:
        for el in p.select(sel):
            el.decompose()
    for a in list(p.find_all("a")):
        href = (a.get("href") or "").lower()
        cls = " ".join(a.get("class") or []).lower()
        aid = (a.get("id") or "").lower()
        if any(k in href for k in rules.anchor_href_contains) or \
           any(k in cls for k in rules.anchor_class_contains) or \
           any(k in aid for k in rules.anchor_id_contains):
            a.decompose()
    for el in list(p.find_all(True)):
        if el.name in rules.inline_inline_tags:
            attrs_text = _attrs_text(el)
            if any(k in attrs_text for k in rules.anchor_class_contains):
                el.decompose()

def _looks_like_caption_or_note(text: str, p: Tag, rules: IgnoreRules = DEFAULT_RULES) -> bool:
    if rules.caption_regex.search(text) or rules.source_or_note_regex.search(text):
        return True
    attrs_text = _attrs_text(p)
    if any(k in attrs_text for k in ("caption", "figure-title", "chart-title", "table-title", "source", "note")):
        return True
    return False

def extract_text(container: Tag, rules: IgnoreRules = DEFAULT_RULES) -> str:
    paragraphs: List[str] = []
    for p in container.find_all(["p", "li"]):
        if p.find_parent(list(rules.non_content_parents)):
            continue
        anc = p.find_parent(attrs={"role": rules.non_content_roles_regex})
        if anc:
            continue

        parent = p.parent
        skip = False
        while isinstance(parent, Tag):
            if _is_table_like(parent, rules):
                skip = True
                break
            parent = parent.parent
        if skip or _is_hidden_by_style_or_parent(p):
            continue

        _strip_inline_refs(p, rules)

        txt = p.get_text(" ", strip=True)
        if not txt:
            continue
        # case-insensitive meta prefix check
        if any(txt.lower().startswith(prefix.lower()) for prefix in rules.meta_prefixes):
            continue
        if _looks_like_caption_or_note(txt, p, rules):
            continue

        txt = re.sub(r"(?<=\w)\s*\[(\d{1,3}|[a-z])\](?=[\s\.,;:])?", "", txt, flags=re.I)
        txt = re.sub(r"(?<=\w)\s*\((\d{1,3})\)(?=[\s\.,;:])?", "", txt)

        txt = re.sub(r"\s+", " ", txt).strip()
        
        
        if txt:
            paragraphs.append(txt)
    return "\n\n".join(paragraphs)

def parse_ecb_article(source: Union[str, Path], rules: IgnoreRules = DEFAULT_RULES) -> Dict[str, str]:
    """Returns ONLY the cleaned text. (Title/authors/date are not extracted.)"""
    html = load_html(source)
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")
    container = _pick_main_container(soup)
    _remove_unwanted(container, rules)
    text = extract_text(container, rules)
    return {"text": text}

def _to_df(records, parquet_path = None):     
    # Create a Polars DataFrame
    df = pl.DataFrame(records)

    df = df.with_columns([
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        pl.col(["category", "title", "filter_type", "url", "text"]).cast(pl.Utf8),
        pl.col(["url_hash", "index"]).cast(pl.UInt64, strict=False),
        pl.col("html_downloaded").cast(pl.Boolean, strict=False), 
        pl.col("path_to_html").map_elements(
            lambda p: None if p is None else str(p),
            return_dtype=pl.Utf8
        ),

        pl.when(pl.col("download_error").cast(pl.String).str.strip_chars().is_in(["", "NaN", "nan", "None"]))
        .then(None).otherwise(pl.col("download_error").cast(pl.String)).alias("download_error"),
        pl.when(pl.col("error").cast(pl.String).str.strip_chars().is_in(["", "NaN", "nan", "None"]))
        .then(None).otherwise(pl.col("error").cast(pl.String)).alias("error"),

        pl.col("filter_value").cast(pl.List(pl.String)),
    ])

    authors_list = []
    for i in df[['url_hash', 'authors']].iter_rows(named= True):
        temp = {}
        temp['url_hash'] = i['url_hash'] 
        authors = None
        authors = ast.literal_eval(i['authors'])
        if len(authors) < 1: 
            authors = None
        
        temp['authors'] = authors
        authors_list.append(temp)

    df = df.drop('authors')
    df = df.join(pl.DataFrame(authors_list, schema= pl.Schema({'url_hash':pl.UInt64, 'authors': pl.List(pl.String)})), on = 'url_hash', how="left")

    df = df.with_columns(
        pl.col("path_to_html").map_elements(
            lambda p: None if p is None else str(p),
            return_dtype=pl.Utf8
        )
    )

    if parquet_path is not None: 
        df.write_parquet(parquet_path)
    return df

def parse_all(all_urls_downloaded, html_dir, out_path): 
    custom_rules = replace(
        DEFAULT_RULES,
        block_selectors=[
            *DEFAULT_RULES.block_selectors,
            ".ecb-authors", ".ecb-author", ".byline", ".author",
            ".ecb-publishDate", ".ecb-pressPublishedDate", ".ecb-article-meta",
            "time[itemprop='datePublished']", "time[datetime]",
            ".ecb-publicationDate",
            "title", ".title",
            "[class*='MathJax']", "mjx-container",
            "[class*='address-box']",
        ],
        meta_prefixes=(
            *DEFAULT_RULES.meta_prefixes,
            "For media queries",
            "Check out The ECB Blog", 
            "For topics relating to", 
        )
    )
    if not out_path.exists(): 
        records = []
        for idx, row in tqdm(all_urls_downloaded.iterrows(), total=all_urls_downloaded.shape[0], desc = "Parsing HTMLs"):
            url = row.get("url", "")
            url_hash = row.get("url_hash")
            path_to_html = html_dir / f"{url_hash}.html"

            # detect PDFs via URL or optional content_type column
            is_pdf = (isinstance(url, str) and url.lower().endswith(".pdf"))

            rec = row.to_dict()
            rec['index'] = int(idx)
            rec['path_to_html'] = path_to_html
            rec['text'] = ""
            rec["error"] = ""

            if not is_pdf:
                try:
                    art = parse_ecb_article(path_to_html, rules=custom_rules)
                    rec["text"] = art.get("text", "") or ""
                except Exception as e:
                    rec["error"] = str(e)
                    logger.warning(f"{url}, {e}")
            else:
                rec["text"] = ""

            records.append(rec)

        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

        logger.info(f"Wrote {len(records)} records to {out_path}")
        res_df = _to_df(records, out_path)
    else: 
        logger.warning(f"Already exists, loading from {out_path}")
        res_df = pl.read_parquet(out_path)
    
    return res_df
    

if __name__ == "__main__": 
    
    # =============== PARAMETERS =============== 
    filter_type = 'year'
    filter_values = FILTERS[filter_type]
    data_dir = FILE_DIR.parent / Path(f"data/blogs_articles/{filter_type}")
    data_dir.mkdir(parents=True, exist_ok=True)
    # =============== 1. Publication Scraping =============== 
    pub_dir = data_dir / 'publications'
    pub_dir.mkdir(parents=True, exist_ok=True)

    for val in tqdm(filter_values,  total=len(filter_values), desc = f"Scraping from All news & Publications page, filter_type: {filter_type}"):
        save_path = pub_dir / f"ecb_published_{val}.csv"
        if not save_path.exists():
            # logger.warning(f"For filter_value {val} of type {filter_type} does not exist yet, scraping the publications list from the url.")
            df = scrape_ecb_pub(filter_type=filter_type, filter_value=val, save_csv=save_path)  
    logger.success('Finished scraping the publications from https://www.ecb.europa.eu/press/pubbydate/html/index.en.html')

    all_urls = pd.concat([pd.read_csv(pub_dir / f"ecb_published_{val}.csv") for val in filter_values])
    all_urls = (
        all_urls
        .groupby(["date", "category", "title", "authors","filter_type","url"], as_index=False)
        .agg({"filter_value": lambda x: list(set(x))})
    )
    all_urls['url_hash'] = all_urls['url'].apply(lambda x: str(hash(x) & ((1<<64)-1)))

    # =============== 2. Downloading HTMLs ===============
    all_urls_downloaded = download_htmls(all_urls, data_dir, filter_type)
    # =============== 3. Parsing the HTMLs to extract the text ===============
    output_path = data_dir.parent /f"parsed_ecb_articles_by_{filter_type}.parquet"
    parse_all(all_urls_downloaded, html_dir= data_dir/ 'htmls', out_path= output_path)
    pass
    
