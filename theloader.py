#!/usr/bin/env python3
"""
theloader - download the high-resolution images of a website, grouped by
the site's own categories.

Built for texture sites whose items live at URLs like
    https://example.com/textures/123
and which belong to categories such as "Concrete", "Stone", ...

Pipeline:
  1. discover item pages   (ID range, sitemap.xml, or crawling the site)
  2. read every item page  (title + category + every image candidate)
  3. for every candidate, find the largest available version
     (srcset, lazy-load attributes, download links, size suffixes, ...)
  4. read only the image header to get width x height, skip small ones
  5. save into  <out>/<category>/<id>_<title>/...  and write manifest.csv

Run `python theloader.py --help` for all options.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, Tag
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None  # textures can be huge (8K+), that's fine

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif", ".avif", ".exr", ".tga")
LAZY_ATTRS = (
    "data-src", "data-original", "data-lazy-src", "data-lazy", "data-url",
    "data-full", "data-full-src", "data-large", "data-large-src", "data-zoom",
    "data-zoom-image", "data-zoom-src", "data-hires", "data-highres", "data-image",
    "data-bg", "data-background", "data-src-retina", "data-orig-file", "data-large-file",
)
SRCSET_ATTRS = ("srcset", "data-srcset", "data-lazy-srcset")
# query parameters that usually *shrink* an image on CDNs / resizers
RESIZE_PARAMS = {"w", "h", "width", "height", "size", "resize", "fit", "quality", "q",
                 "dpr", "crop", "thumb", "thumbnail", "maxwidth", "maxheight", "scale"}
# path fragments that usually mean "small version"
SMALL_HINTS = re.compile(r"(thumb|thumbnail|small|preview|_sm\b|-sm\b|mini|icon|avatar|logo)", re.I)
SIZE_SUFFIX = re.compile(r"[-_@](\d{2,4})x(\d{2,4})(?=\.[a-z0-9]+$)", re.I)
CSS_URL = re.compile(r"url\(\s*['\"]?([^'\")]+)['\"]?\s*\)", re.I)

print_lock = threading.Lock()


def log(*a):
    with print_lock:
        print(*a, flush=True)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

class Http:
    def __init__(self, delay: float, timeout: float, retries: int, user_agent: str, cookies: str | None):
        self.delay = delay
        self.timeout = timeout
        self.retries = retries
        self.local = threading.local()
        self.headers = {"User-Agent": user_agent, "Accept-Language": "en,fa;q=0.8"}
        self.cookies = cookies

    @property
    def session(self) -> requests.Session:
        s = getattr(self.local, "s", None)
        if s is None:
            s = requests.Session()
            s.headers.update(self.headers)
            if self.cookies:
                s.headers["Cookie"] = self.cookies
            self.local.s = s
        return s

    def get(self, url: str, stream: bool = False, referer: str | None = None) -> requests.Response | None:
        headers = {"Referer": referer} if referer else {}
        for attempt in range(self.retries + 1):
            try:
                if self.delay:
                    time.sleep(self.delay)
                r = self.session.get(url, timeout=self.timeout, stream=stream, headers=headers)
                if r.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    r.close()
                    time.sleep(2 ** attempt * 2)
                    continue
                return r
            except requests.RequestException as e:
                if attempt >= self.retries:
                    log(f"  ! request failed: {url} ({e.__class__.__name__})")
                    return None
                time.sleep(2 ** attempt * 2)
        return None


class Renderer:
    """Optional headless browser for JavaScript-heavy pages (needs playwright)."""

    def __init__(self):
        from playwright.sync_api import sync_playwright  # noqa: lazy import
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch()
        self._lock = threading.Lock()

    def html(self, url: str) -> str | None:
        with self._lock:  # playwright sync API is not thread-safe
            page = self._browser.new_page()
            try:
                page.goto(url, wait_until="networkidle", timeout=60000)
                # scroll to trigger lazy-loading
                page.evaluate("""async () => {
                    for (let y = 0; y < document.body.scrollHeight; y += 800) {
                        window.scrollTo(0, y); await new Promise(r => setTimeout(r, 150));
                    }}""")
                return page.content()
            except Exception as e:
                log(f"  ! render failed: {url} ({e})")
                return None
            finally:
                page.close()

    def close(self):
        self._browser.close()
        self._pw.stop()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def safe_name(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip(" .")
    return text[:max_len].strip() or "untitled"


def looks_like_image(url: str) -> bool:
    return urlparse(url).path.lower().endswith(IMAGE_EXT)


def parse_srcset(value: str) -> list[tuple[str, float]]:
    out = []
    for part in re.split(r",\s+(?=\S)", value.strip()):
        bits = part.strip().split()
        if not bits:
            continue
        score = 0.0
        if len(bits) > 1:
            d = bits[1].lower()
            try:
                score = float(d[:-1]) * (1 if d.endswith("w") else 1000)
            except ValueError:
                pass
        out.append((bits[0], score))
    return out


def upgrade_variants(url: str) -> list[str]:
    """Guess bigger versions of an image URL (tried first, the original last)."""
    variants = []
    p = urlparse(url)
    # 1) remove size suffix: photo-300x200.jpg -> photo.jpg
    new_path = SIZE_SUFFIX.sub("", p.path)
    if new_path != p.path:
        variants.append(urlunparse(p._replace(path=new_path)))
    # 2) remove resize query params: image.jpg?w=400 -> image.jpg
    if p.query:
        kept = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if k.lower() not in RESIZE_PARAMS]
        if len(kept) != len(parse_qsl(p.query, keep_blank_values=True)):
            q = urlencode(kept)
            variants.append(urlunparse(p._replace(query=q)))
            if new_path != p.path:
                variants.append(urlunparse(p._replace(path=new_path, query=q)))
    # 3) thumb folders: /thumbs/x.jpg -> /x.jpg, /small/ -> /large/ , /original/
    for a, bs in (("/thumbs/", ["/"]), ("/thumb/", ["/"]), ("/thumbnails/", ["/"]),
                  ("/small/", ["/original/", "/large/", "/full/"]),
                  ("/medium/", ["/original/", "/large/", "/full/"]),
                  ("/preview/", ["/original/", "/full/"])):
        if a in p.path:
            for b in bs:
                variants.append(urlunparse(p._replace(path=p.path.replace(a, b, 1))))
    for a, b in (("_thumb.", "."), ("-thumb.", "."), ("_small.", "."), ("-small.", "."),
                 ("_preview.", "."), ("-preview.", "."), ("_sm.", "."), ("_medium.", ".")):
        if a in p.path.lower():
            i = p.path.lower().rfind(a)
            variants.append(urlunparse(p._replace(path=p.path[:i] + b + p.path[i + len(a):])))
    variants.append(url)
    seen, out = set(), []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# --------------------------------------------------------------------------- #
# Page parsing
# --------------------------------------------------------------------------- #

@dataclass
class Candidate:
    url: str
    hint: float = 0.0        # srcset width etc. (bigger is better)
    source: str = ""         # where we found it


@dataclass
class Item:
    page_url: str
    item_id: str
    title: str = ""
    category: str = ""
    candidates: list[Candidate] = field(default_factory=list)


def extract_category(soup: BeautifulSoup, page_url: str, args) -> str:
    return find_category(soup, page_url, args)[0]


NAV_CLASS = re.compile(r"(^|[-_ ])(nav|navbar|menu|header|footer|mega)([-_ ]|$)", re.I)


def _in_navigation(el) -> bool:
    """True if the element sits in the site menu / header / footer (same on every page)."""
    for p in [el] + list(el.parents):
        if getattr(p, "name", None) in ("nav", "header", "footer"):
            return True
        if getattr(p, "attrs", None):
            cls = " ".join(p.get("class", [])) + " " + (p.get("id") or "")
            if NAV_CLASS.search(cls) or p.get("role") == "navigation":
                return True
    return False


def _norm_label(t: str) -> str:
    t = (t or "").replace("\u200c", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", t).strip().strip(":：-–|").strip().lower()


def _clean_value(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").replace("\xa0", " ")).strip().strip(":：-–|،,").strip()


def _value_after(node, labels: set[str]) -> str | None:
    """The first non-empty text that follows a label like 'Category:'."""
    cur = node
    for _ in range(4):
        for sib in cur.next_siblings:
            if isinstance(sib, Tag):
                if sib.name in ("script", "style"):
                    continue
                link = sib.find("a") if sib.name != "a" else None
                txt = _clean_value(link.get_text(" ", strip=True) if link and link.get_text(strip=True)
                                   else sib.get_text(" ", strip=True))
            else:
                txt = _clean_value(str(sib))
            if txt and _norm_label(txt) not in labels:
                return txt if len(txt) <= 60 else None
        cur = cur.parent
        if cur is None or cur.name in ("body", "html", "[document]"):
            break
    return None


def find_category(soup: BeautifulSoup, page_url: str, args) -> tuple[str, str]:
    """Return (category, how it was found)."""
    # 1) user-supplied CSS selector wins
    if args.category_selector:
        els = soup.select(args.category_selector)
        texts = [e.get_text(" ", strip=True) for e in els if e.get_text(strip=True)]
        if texts:
            return (" - ".join(texts) if args.category_join else texts[0]), f"selector {args.category_selector!r}"

    # 2) a labelled field on the page:  "Category: Stone", <dt>Category</dt><dd>Stone</dd>, ...
    labels = {_norm_label(l) for l in args.category_label.split(",") if l.strip()}
    for node in soup.find_all(string=True):
        parent = node.parent
        if parent is None or parent.name in ("script", "style", "title", "option", "noscript"):
            continue
        t = _norm_label(str(node))
        if not t:
            continue
        value = None
        if t in labels:
            if _in_navigation(parent):
                continue
            value = _value_after(node, labels)
        else:
            m = re.match(r"^\s*([^:：]{1,30})[:：]\s*(.+)$", str(node).replace("\u200c", " "))
            if m and _norm_label(m.group(1)) in labels and not _in_navigation(parent):
                value = _clean_value(m.group(2))
                if len(value) > 60:
                    value = None
        if value:
            return value, f"label {(t if t in labels else _norm_label(t.split(':')[0]))!r} on the page"

    # 3) an element whose class / id says "category" (outside the menu)
    for el in soup.select('[class*="category" i], [id*="category" i], [itemprop="category"]'):
        if el.name in ("body", "html") or _in_navigation(el):
            continue
        link = el.find("a")
        txt = _clean_value((link or el).get_text(" ", strip=True))
        if txt and len(txt) <= 60 and _norm_label(txt) not in labels:
            return txt, "element with class/id 'category'"

    # 4) JSON-LD BreadcrumbList
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(s.string or "")
        except Exception:
            continue
        stack = data if isinstance(data, list) else [data]
        while stack:
            d = stack.pop()
            if isinstance(d, dict):
                if d.get("@type") == "BreadcrumbList":
                    names = []
                    for el in d.get("itemListElement", []):
                        it = el.get("item")
                        nm = el.get("name") or (it.get("name") if isinstance(it, dict) else None)
                        if nm:
                            names.append(nm)
                    cat = _cat_from_crumbs(names)
                    if cat:
                        return cat, "JSON-LD breadcrumb"
                stack.extend(v for v in d.values() if isinstance(v, (dict, list)))
            elif isinstance(d, list):
                stack.extend(d)

    # 5) HTML breadcrumb
    crumb = soup.select_one('[aria-label*="breadcrumb" i], .breadcrumb, .breadcrumbs, '
                            '#breadcrumb, #breadcrumbs, [class*="breadcrumb"], [itemtype*="BreadcrumbList"]')
    if crumb:
        names = [a.get_text(" ", strip=True) for a in crumb.find_all(["a", "span", "li"]) if a.get_text(strip=True)]
        uniq = []
        for n in names:
            if not uniq or uniq[-1] != n:
                uniq.append(n)
        cat = _cat_from_crumbs(uniq)
        if cat:
            return cat, "breadcrumb"

    # 6) a category-looking link outside the site menu
    pat = re.compile(args.category_link_pattern, re.I)
    for a in soup.find_all("a", href=True):
        if pat.search(urlparse(urljoin(page_url, a["href"])).path) and a.get_text(strip=True) \
                and not _in_navigation(a):
            return a.get_text(" ", strip=True), f"category link {a['href']}"

    # 7) meta tag article:section
    m = soup.find("meta", attrs={"property": "article:section"})
    if m and m.get("content"):
        return m["content"], "meta article:section"
    return "uncategorized", "nothing found"


def _cat_from_crumbs(names: list[str]) -> str:
    names = [n for n in names if n and n.lower() not in ("home", "خانه", "صفحه اصلی", "»", ">", "/")]
    if len(names) >= 2:
        return names[-2]   # last crumb is usually the item itself
    return names[0] if names else ""


def extract_title(soup: BeautifulSoup) -> str:
    for sel in ("h1",):
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            return el.get_text(" ", strip=True)
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return og["content"]
    return soup.title.get_text(strip=True) if soup.title else ""


def extract_candidates(soup: BeautifulSoup, page_url: str, args) -> list[Candidate]:
    root = soup
    if args.image_scope:
        scoped = soup.select(args.image_scope)
        if scoped:
            root = BeautifulSoup("".join(str(s) for s in scoped), "html.parser")
    cands: list[Candidate] = []

    def add(u, hint=0.0, src=""):
        if not u or u.startswith("data:"):
            return
        cands.append(Candidate(urljoin(page_url, u.strip()), hint, src))

    # links that point straight at image files (often the full-res download)
    for a in root.find_all("a", href=True):
        if looks_like_image(a["href"]):
            add(a["href"], 1e9, "link")
        for attr in ("data-download", "data-href", "data-full"):
            if a.get(attr):
                add(a[attr], 1e9, "link-data")
    for el in root.find_all(["img", "source"]):
        for attr in SRCSET_ATTRS:
            if el.get(attr):
                for u, w in parse_srcset(el[attr]):
                    add(u, w, attr)
        for attr in LAZY_ATTRS:
            if el.get(attr):
                add(el[attr], 1e6, attr)
        if el.name == "img" and el.get("src"):
            add(el["src"], 0, "src")
    # inline background images
    for el in root.find_all(style=True):
        for u in CSS_URL.findall(el["style"]):
            add(u, 0, "css")
    for el in root.find_all(attrs={"data-bg": True}):
        add(el["data-bg"], 0, "data-bg")
    # social / structured data (whole page, these describe the item itself)
    for prop in ("og:image", "og:image:url", "og:image:secure_url", "twitter:image"):
        for m in soup.find_all("meta", attrs={"property": prop}) + soup.find_all("meta", attrs={"name": prop}):
            add(m.get("content"), 0, prop)
    link = soup.find("link", rel="image_src")
    if link:
        add(link.get("href"), 0, "image_src")
    for s in soup.find_all("script", type="application/ld+json"):
        for u in re.findall(r'"(?:image|contentUrl|url)"\s*:\s*"([^"]+)"', s.string or ""):
            if looks_like_image(u):
                add(u.replace("\\/", "/"), 0, "json-ld")
    if args.scan_scripts:
        # image URLs hidden inside inline JavaScript (galleries, viewers)
        for s in soup.find_all("script"):
            for u in re.findall(r'["\']((?:https?:)?//[^"\']+?\.(?:jpe?g|png|webp|tiff?))["\']'
                                r'|["\'](/[^"\']+?\.(?:jpe?g|png|webp|tiff?))["\']', s.string or "", re.I):
                add((u[0] or u[1]).replace("\\/", "/"), 0, "script")

    exclude = re.compile(args.exclude, re.I) if args.exclude else None
    out, seen = [], set()
    for c in sorted(cands, key=lambda c: -c.hint):
        if c.url in seen:
            continue
        if exclude and exclude.search(c.url):
            continue
        if urlparse(c.url).path.lower().endswith((".svg", ".ico")):
            continue
        seen.add(c.url)
        out.append(c)
    return out


# --------------------------------------------------------------------------- #
# Image probing / downloading
# --------------------------------------------------------------------------- #

def probe_size(http: Http, url: str, referer: str) -> tuple[int, int, str] | None:
    """Read only the beginning of the file to learn its dimensions."""
    r = http.get(url, stream=True, referer=referer)
    if r is None:
        return None
    try:
        if r.status_code != 200:
            return None
        ctype = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if ctype and not ctype.startswith("image/") and ctype not in ("application/octet-stream", "binary/octet-stream"):
            return None
        parser = ImageFile.Parser()
        read = 0
        for chunk in r.iter_content(16384):
            read += len(chunk)
            try:
                parser.feed(chunk)
            except Exception:
                return None
            if parser.image is not None:
                w, h = parser.image.size
                return w, h, r.url
            if read > 4 * 1024 * 1024:
                break
        # header not found in first chunks: fall back to full decode
        return None
    finally:
        r.close()


def download(http: Http, url: str, referer: str) -> bytes | None:
    r = http.get(url, referer=referer)
    if r is None or r.status_code != 200:
        return None
    return r.content


def best_version(http: Http, cand: Candidate, referer: str, try_upgrades: bool) -> tuple[str, int, int] | None:
    variants = upgrade_variants(cand.url) if try_upgrades else [cand.url]
    best = None
    for v in variants:
        res = probe_size(http, v, referer)
        if res is None:
            continue
        w, h, final = res
        if best is None or w * h > best[1] * best[2]:
            best = (final, w, h)
        if v != cand.url:
            # an upgrade worked; no need to try the smaller variants
            break
    return best


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

def item_id_from_url(url: str, item_re: re.Pattern) -> str | None:
    m = item_re.search(urlparse(url).path)
    if not m:
        return None
    return m.group("id") if "id" in m.groupdict() else m.group(1) if m.groups() else m.group(0).strip("/")


def discover_ids(args) -> list[str]:
    out = []
    for part in args.ids.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(str(i) for i in range(int(a), int(b) + 1))
        elif part:
            out.append(part)
    return [args.item_url.format(base=args.base.rstrip("/"), id=i) for i in out]


def discover_sitemap(http: Http, args, item_re) -> list[str]:
    start = args.sitemap if args.sitemap.startswith("http") else urljoin(args.base, "/sitemap.xml")
    todo, seen, pages = [start], set(), []
    # robots.txt may list sitemaps
    r = http.get(urljoin(args.base, "/robots.txt"))
    if r is not None and r.status_code == 200:
        todo += re.findall(r"(?im)^sitemap:\s*(\S+)", r.text)
    while todo:
        sm = todo.pop()
        if sm in seen:
            continue
        seen.add(sm)
        r = http.get(sm)
        if r is None or r.status_code != 200:
            continue
        try:
            root = ET.fromstring(r.content)
        except ET.ParseError:
            continue
        for loc in root.iter():
            if loc.tag.endswith("loc") and loc.text:
                u = loc.text.strip()
                if u.endswith(".xml") or "sitemap" in u.lower() and not item_re.search(urlparse(u).path):
                    todo.append(u)
                elif item_re.search(urlparse(u).path):
                    pages.append(u)
    log(f"sitemap: found {len(pages)} item pages")
    return list(dict.fromkeys(pages))


def discover_crawl(http: Http, args, item_re, renderer) -> list[str]:
    base = urlparse(args.base)
    starts = [urljoin(args.base, s) for s in (args.start or ["/"])]
    q, seen, items = deque(starts), set(starts), []
    pages = 0
    skip = re.compile(r"\.(jpe?g|png|webp|gif|svg|ico|css|js|pdf|zip|rar|7z|mp4|mp3|woff2?|ttf|exr|tiff?)$", re.I)
    while q and pages < args.max_pages:
        url = q.popleft()
        html = fetch_html(http, url, renderer)
        pages += 1
        if html is None:
            continue
        soup = BeautifulSoup(html, "html.parser")
        new_items = 0
        for a in soup.find_all("a", href=True):
            u = urljoin(url, a["href"]).split("#")[0]
            p = urlparse(u)
            if p.netloc != base.netloc or skip.search(p.path) or u in seen:
                continue
            seen.add(u)
            if item_re.search(p.path):
                items.append(u)
                new_items += 1
            elif not args.follow or re.search(args.follow, p.path + ("?" + p.query if p.query else "")):
                q.append(u)
        if pages % 10 == 0 or new_items:
            log(f"crawl: {pages} pages visited, {len(items)} item pages found (queue {len(q)})")
        if args.limit and len(items) >= args.limit:
            break
        if not items and pages in (30, 150):
            show_url_shapes(seen, base.netloc)
    if not items:
        show_url_shapes(seen, base.netloc)
    return list(dict.fromkeys(items))


def show_url_shapes(urls, netloc: str, top: int = 20):
    """Print the most common address shapes, to help find the right --item-pattern."""
    from collections import Counter
    shapes, example = Counter(), {}
    for u in urls:
        p = urlparse(u)
        if p.netloc != netloc:
            continue
        shape = re.sub(r"\d+", "<N>", p.path) + ("?" + re.sub(r"=\d+", "=<N>", p.query) if p.query else "")
        shapes[shape] += 1
        example.setdefault(shape, u)
    log("\n  No page matched --item-pattern yet. Most common address shapes on the site:")
    for shape, n in shapes.most_common(top):
        log(f"    {n:>5} x  {shape:<45} e.g. {example[shape]}")
    log("  -> Send the line that looks like a texture page, or set --item-pattern / --ids accordingly.\n")


def fetch_html(http: Http, url: str, renderer) -> str | None:
    if renderer:
        return renderer.html(url)
    r = http.get(url)
    if r is None or r.status_code != 200 or "html" not in r.headers.get("Content-Type", "html"):
        return None
    r.encoding = r.encoding if r.encoding and r.encoding.lower() != "iso-8859-1" else r.apparent_encoding
    return r.text


# --------------------------------------------------------------------------- #
# Main work
# --------------------------------------------------------------------------- #

class Manifest:
    FIELDS = ["item_id", "category", "title", "page_url", "image_url", "width", "height", "bytes", "sha1", "file"]

    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()
        self.done_pages: set[str] = set()
        self.hashes: set[str] = set()
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.done_pages.add(row["page_url"])
                    self.hashes.add(row["sha1"])
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=self.FIELDS)
        if new:
            self.w.writeheader()

    def add(self, row: dict):
        with self.lock:
            self.w.writerow(row)
            self.f.flush()

    def seen_hash(self, h: str) -> bool:
        with self.lock:
            if h in self.hashes:
                return True
            self.hashes.add(h)
            return False


def process_item(url: str, http: Http, args, item_re, manifest: Manifest | None, renderer) -> dict:
    stats = {"saved": 0, "small": 0, "dup": 0, "failed": 0}
    html = fetch_html(http, url, renderer)
    if html is None:
        log(f"✗ {url}  (page not available)")
        stats["failed"] += 1
        return stats
    soup = BeautifulSoup(html, "html.parser")
    category, cat_how = find_category(soup, url, args)
    item = Item(page_url=url, item_id=item_id_from_url(url, item_re) or "",
                title=extract_title(soup), category=category)
    if args.only_category and not category_matches(item.category, args.only_category):
        return stats
    item.candidates = extract_candidates(soup, url, args)

    found = []
    seen_final = set()
    for c in item.candidates:
        res = best_version(http, c, url, not args.no_upgrade)
        if res is None:
            continue
        final, w, h = res
        if final in seen_final:
            continue
        seen_final.add(final)
        found.append((final, w, h, c))

    min_side = lambda w, h: min(w, h) if args.size_mode == "min" else max(w, h)  # noqa: E731
    big = [f for f in found if min_side(f[1], f[2]) >= args.min_size]
    stats["small"] += len(found) - len(big)
    big.sort(key=lambda f: -f[1] * f[2])
    if args.best_only:
        big = big[:1]

    if args.probe:
        log(f"\n=== {url}")
        log(f"  id       : {item.item_id}")
        log(f"  title    : {item.title}")
        log(f"  category : {item.category}   (found by: {cat_how})")
        log(f"  {len(item.candidates)} image candidates on page:")
        for final, w, h, c in sorted(found, key=lambda f: -f[1] * f[2]):
            ok = "KEEP " if (final, w, h, c) in big else "small"
            log(f"   [{ok}] {w:>5}x{h:<5} {c.source:<14} {final}")
        unreachable = [c.url for c in item.candidates if c.url not in {f[3].url for f in found}]
        for u in unreachable:
            log(f"   [ -- ] not an image / unreachable: {u}")
        return stats

    cat_dir = safe_name(item.category)
    name = safe_name(f"{item.item_id}_{item.title}" if item.title else item.item_id or "item")
    folder = os.path.join(args.out, cat_dir, name) if args.item_folders else os.path.join(args.out, cat_dir)
    for i, (final, w, h, c) in enumerate(big):
        data = download(http, final, url)
        if not data:
            stats["failed"] += 1
            continue
        sha1 = hashlib.sha1(data).hexdigest()
        if manifest and manifest.seen_hash(sha1):
            stats["dup"] += 1
            continue
        try:
            with Image.open(io.BytesIO(data)) as im:
                w, h = im.size
                fmt = (im.format or "").lower()
        except Exception:
            fmt = ""
        ext = os.path.splitext(urlparse(final).path)[1].lower() or ("." + fmt if fmt else ".jpg")
        if ext == ".jpeg":
            ext = ".jpg"
        orig = safe_name(os.path.splitext(os.path.basename(urlparse(final).path))[0], 60)
        fname = f"{orig}{ext}" if args.item_folders else (f"{name}{ext}" if len(big) == 1 else f"{name}_{i + 1}{ext}")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, fname)
        k = 2
        while os.path.exists(path):
            path = os.path.join(folder, f"{os.path.splitext(fname)[0]}_{k}{ext}")
            k += 1
        with open(path, "wb") as f:
            f.write(data)
        stats["saved"] += 1
        if manifest:
            manifest.add({"item_id": item.item_id, "category": item.category, "title": item.title,
                          "page_url": url, "image_url": final, "width": w, "height": h,
                          "bytes": len(data), "sha1": sha1, "file": os.path.relpath(path, args.out)})
    log(f"✓ [{item.category}] {item.item_id} {item.title[:40]!r}: "
        f"{stats['saved']} saved, {stats['small']} too small, {stats['dup']} duplicate")
    return stats



# --------------------------------------------------------------------------- #
# Scan: which IDs exist, and which category lives in which ID range
# --------------------------------------------------------------------------- #

def category_matches(category: str, wanted: str) -> bool:
    names = [w.strip().lower() for w in wanted.split(",") if w.strip()]
    return (category or "").strip().lower() in names


def scan_page(url: str, http: Http, args, item_re, renderer) -> dict:
    row = {"id": item_id_from_url(url, item_re) or "", "url": url, "status": "", "category": "", "title": ""}
    if renderer:
        html = renderer.html(url)
        row["status"] = "active" if html else "missing"
    else:
        r = http.get(url)
        if r is None:
            row["status"] = "error"
            return row
        if r.status_code != 200:
            row["status"] = "missing" if r.status_code in (404, 410) else f"http {r.status_code}"
            return row
        if "html" not in r.headers.get("Content-Type", "html"):
            row["status"] = "not html"
            return row
        # a "missing" texture often redirects to the home page or a list page
        if r.url != url and not item_re.search(urlparse(r.url).path):
            row["status"] = "redirect"
            return row
        r.encoding = r.encoding if r.encoding and r.encoding.lower() != "iso-8859-1" else r.apparent_encoding
        html = r.text
        row["status"] = "active"
    if html:
        soup = BeautifulSoup(html, "html.parser")
        row["title"] = extract_title(soup)
        row["category"] = extract_category(soup, url, args)
    return row


def _id_key(row):
    return (0, int(row["id"]), "") if str(row["id"]).isdigit() else (1, 0, str(row["id"]))


def _ranges(nums: list[int]) -> list[tuple[int, int]]:
    out = []
    for n in sorted(nums):
        if out and n == out[-1][1] + 1:
            out[-1] = (out[-1][0], n)
        else:
            out.append((n, n))
    return out


def _fmt_ranges(rs: list[tuple[int, int]], max_show: int = 12) -> str:
    parts = [f"{a}" if a == b else f"{a}-{b}" for a, b in rs]
    if len(parts) > max_show:
        return ", ".join(parts[:max_show]) + f", ... (+{len(parts) - max_show} more)"
    return ", ".join(parts)


def scan_report(rows: list[dict]) -> str:
    active = sorted([r for r in rows if r["status"] == "active"], key=_id_key)
    lines = [f"Checked IDs : {len(rows)}",
             f"Active IDs  : {len(active)}",
             f"Not active  : {len(rows) - len(active)}"]
    num_active = [int(r["id"]) for r in active if str(r["id"]).isdigit()]
    if num_active:
        lines.append(f"Lowest / highest active ID: {min(num_active)} / {max(num_active)}")
        lines.append(f"Active ID ranges: {_fmt_ranges(_ranges(num_active))}")
    other = {}
    for r in rows:
        if r["status"] != "active":
            other[r["status"]] = other.get(r["status"], 0) + 1
    if other:
        lines.append("Not active because: " + ", ".join(f"{k}: {v}" for k, v in sorted(other.items())))

    # category blocks: runs of consecutive *active* IDs with the same category
    # (missing IDs in between don't break a block)
    blocks: dict[str, list[tuple[int, int]]] = {}
    counts: dict[str, int] = {}
    prev_cat = None
    for r in active:
        cat = r["category"] or "uncategorized"
        counts[cat] = counts.get(cat, 0) + 1
        if not str(r["id"]).isdigit():
            continue
        n = int(r["id"])
        if cat == prev_cat:
            a, _ = blocks[cat][-1]
            blocks[cat][-1] = (a, n)
        else:
            blocks.setdefault(cat, []).append((n, n))
        prev_cat = cat
    lines.append("")
    lines.append(f"Categories ({len(counts)}):")
    w = max([len(c) for c in counts] + [8])
    for cat, n in sorted(counts.items(), key=lambda kv: (blocks.get(kv[0], [(0, 0)])[0][0], kv[0])):
        bl = blocks.get(cat, [])
        span = f"   (spread over {bl[0][0]}-{bl[-1][1]})" if len(bl) > 3 else ""
        lines.append(f"  {cat:<{w}}  {n:>6} textures   IDs: {_fmt_ranges(bl, 6)}{span}")
    return "\n".join(lines)


def run_scan(urls: list[str] | None, http: Http, args, item_re, renderer) -> int:
    os.makedirs(args.out, exist_ok=True)
    rows: list[dict] = []
    workers = 1 if renderer else max(1, args.workers)
    done = [0]

    def run(batch):
        with ThreadPoolExecutor(workers) as ex:
            for row in ex.map(lambda u: scan_page(u, http, args, item_re, renderer), batch):
                rows.append(row)
                done[0] += 1
                mark = "✓" if row["status"] == "active" else "·"
                if row["status"] == "active" or done[0] % 50 == 0:
                    log(f"{mark} {row['id']:>6}  {row['status']:<9} {row['category'][:25]:<25} {row['title'][:40]}")

    if urls is not None:
        log(f"scanning {len(urls)} pages ...\n")
        run(urls)
    else:
        # open-ended: walk upwards from --scan-start until --scan-gap IDs in a row are missing
        log(f"scanning from ID {args.scan_start} until {args.scan_gap} IDs in a row are missing ...\n")
        n, last_active, batch_size = args.scan_start, args.scan_start - 1, max(20, workers * 10)
        while n <= args.scan_max and n - last_active <= args.scan_gap:
            batch_ids = list(range(n, n + batch_size))
            start = len(rows)
            run([args.item_url.format(base=args.base.rstrip("/"), id=i) for i in batch_ids])
            for r in rows[start:]:
                if r["status"] == "active" and str(r["id"]).isdigit():
                    last_active = max(last_active, int(r["id"]))
            n += batch_size

    rows.sort(key=_id_key)
    csv_path = os.path.join(args.out, "scan.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "status", "category", "title", "url"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    report = scan_report(rows)
    with open(os.path.join(args.out, "scan_summary.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    log("\n" + "=" * 70 + "\n" + report + "\n" + "=" * 70)
    log(f"\nSaved: {csv_path}  and  scan_summary.txt")
    log("Next: download from this list, e.g.\n"
        f"  python theloader.py {args.base} --from-scan \"{csv_path}\" --only-category \"Stone\"")
    return 0


def load_scan(path: str, only_category: str | None) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "active"]
    if only_category:
        rows = [r for r in rows if category_matches(r["category"], only_category)]
    log(f"from scan: {len(rows)} active pages" + (f" in '{only_category}'" if only_category else ""))
    return [r["url"] for r in rows]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download high-resolution images of a site, grouped by category.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # look at 2 pages and show what would be downloaded (nothing is saved)
  python theloader.py https://example.com --ids 1,2 --probe

  # download textures 1..500
  python theloader.py https://example.com --ids 1-500

  # find every /textures/N page automatically by crawling the site
  python theloader.py https://example.com --crawl
""")
    p.add_argument("base", help="site address, e.g. https://example.com")
    g = p.add_argument_group("which pages")
    g.add_argument("--ids", help="item IDs: '1-500' or '3,7,10-20'")
    g.add_argument("--item-url", default="{base}/textures/{id}",
                   help="item URL template for --ids (default: {base}/textures/{id})")
    g.add_argument("--item-pattern", default=r"/textures?/(?P<id>\d+)(?:[/-][^/?#]*)?/?$",
                   help="regex that recognises an item page path (default: /textures/<N>, also /texture/<N> and /textures/<N>-slug)")
    g.add_argument("--sitemap", nargs="?", const="auto", help="read item pages from sitemap.xml")
    g.add_argument("--crawl", action="store_true", help="crawl the site to find item pages")
    g.add_argument("--start", action="append", help="crawl start path(s), e.g. /textures (default /)")
    g.add_argument("--follow", help="only follow non-item links whose path matches this regex while crawling")
    g.add_argument("--max-pages", type=int, default=5000, help="crawl limit (default 5000)")
    g.add_argument("--urls", help="text file with one item URL per line")
    g = p.add_argument_group("scan (check which IDs exist before downloading)")
    g.add_argument("--scan", action="store_true",
                   help="only check which item pages exist and their categories; writes scan.csv + scan_summary.txt")
    g.add_argument("--scan-start", type=int, default=1, help="first ID when scanning without --ids (default 1)")
    g.add_argument("--scan-gap", type=int, default=500,
                   help="without --ids: stop after this many missing IDs in a row (default 500)")
    g.add_argument("--scan-max", type=int, default=1000000, help="never scan past this ID (default 1000000)")
    g.add_argument("--from-scan", metavar="SCAN_CSV", help="download the active pages listed in a scan.csv")
    g.add_argument("--only-category", metavar="NAMES",
                   help="only these categories, comma separated, e.g. 'Stone,Concrete'")
    g = p.add_argument_group("what to keep")
    g.add_argument("--min-size", type=int, default=600, help="minimum pixels (default 600)")
    g.add_argument("--size-mode", choices=["min", "max"], default="min",
                   help="'min': the SHORT side must be >= min-size (default); 'max': the long side")
    g.add_argument("--best-only", action="store_true", help="keep only the single largest image per item")
    g.add_argument("--exclude", default=r"logo|favicon|sprite|avatar|icon|banner|placeholder",
                   help="regex of image URLs to ignore")
    g.add_argument("--image-scope", help="CSS selector: only look for images inside this part of the page")
    g.add_argument("--scan-scripts", action="store_true", help="also find image URLs inside inline JavaScript")
    g.add_argument("--no-upgrade", action="store_true", help="don't try to guess bigger versions of URLs")
    g = p.add_argument_group("categories")
    g.add_argument("--category-selector", help="CSS selector of the category text, e.g. '.texture-category a'")
    g.add_argument("--category-label", default="category,categories,cat,دسته,دسته بندی,دسته‌بندی,دسته بندی ها,دسته‌ها",
                   help="words that label the category field on the page, comma separated "
                        "(default: category, دسته‌بندی, ...)")
    g.add_argument("--category-join", action="store_true",
                   help="if the selector matches several elements, join them (e.g. 'Stone - Marble')")
    g.add_argument("--category-link-pattern", default=r"/(category|categories|cat|textures?/category|collection)s?/",
                   help="regex of category link URLs used as a fallback")
    g = p.add_argument_group("output & network")
    g.add_argument("--out", default="downloads", help="output folder (default ./downloads)")
    g.add_argument("--item-folders", action="store_true",
                   help="one sub-folder per item: <out>/<category>/<id>_<title>/<file>. "
                        "Default: all images of a category in one folder: <out>/<category>/<id>_<title>.jpg")
    g.add_argument("--flat", action="store_true", help=argparse.SUPPRESS)  # old option, now the default
    g.add_argument("--probe", action="store_true", help="only analyse the pages and print the result")
    g.add_argument("--workers", type=int, default=4, help="parallel pages (default 4)")
    g.add_argument("--delay", type=float, default=0.2, help="seconds to wait before each request (default 0.2)")
    g.add_argument("--timeout", type=float, default=60)
    g.add_argument("--retries", type=int, default=3)
    g.add_argument("--render", action="store_true", help="use a headless browser (for JavaScript sites; needs playwright)")
    g.add_argument("--cookies", help="Cookie header value, for pages that need login")
    g.add_argument("--user-agent", default="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 theloader/1.0")
    g.add_argument("--limit", type=int, help="process at most N item pages (good for testing)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.base.startswith(("http://", "https://")):
        args.base = "https://" + args.base
    item_re = re.compile(args.item_pattern)
    http = Http(args.delay, args.timeout, args.retries, args.user_agent, args.cookies)
    renderer = Renderer() if args.render else None

    try:
        if args.scan and not (args.ids or args.urls or args.sitemap or args.crawl):
            return run_scan(None, http, args, item_re, renderer)
        urls: list[str] = []
        if args.from_scan:
            urls += load_scan(args.from_scan, args.only_category)
        if args.ids:
            urls += discover_ids(args)
        if args.urls:
            with open(args.urls, encoding="utf-8") as f:
                urls += [l.strip() for l in f if l.strip() and not l.startswith("#")]
        if args.sitemap:
            urls += discover_sitemap(http, args, item_re)
        if args.crawl:
            urls += discover_crawl(http, args, item_re, renderer)
        if not (args.ids or args.urls or args.sitemap or args.crawl or args.from_scan):
            log("No pages selected. Trying sitemap.xml first, then crawling ...")
            urls = discover_sitemap(http, args, item_re) or discover_crawl(http, args, item_re, renderer)
        urls = list(dict.fromkeys(urls))
        if args.limit:
            urls = urls[: args.limit]
        if not urls:
            log("No item pages found. Check --item-pattern / --ids / --start.")
            return 1
        if args.scan:
            return run_scan(urls, http, args, item_re, renderer)

        manifest = None
        if not args.probe:
            os.makedirs(args.out, exist_ok=True)
            manifest = Manifest(os.path.join(args.out, "manifest.csv"))
            before = len(urls)
            urls = [u for u in urls if u not in manifest.done_pages]
            if before != len(urls):
                log(f"resume: skipping {before - len(urls)} pages already in manifest.csv")
        log(f"{len(urls)} item pages to process\n")

        total = {"saved": 0, "small": 0, "dup": 0, "failed": 0}
        workers = 1 if (args.probe or renderer) else max(1, args.workers)
        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(process_item, u, http, args, item_re, manifest, renderer) for u in urls]
            for fu in as_completed(futs):
                try:
                    for k, v in fu.result().items():
                        total[k] += v
                except Exception as e:  # never let one page kill the run
                    log(f"  ! error: {e}")
                    total["failed"] += 1
        if not args.probe:
            log(f"\nDone. {total['saved']} images saved to '{args.out}', "
                f"{total['small']} too small, {total['dup']} duplicates, {total['failed']} failed.")
        return 0
    finally:
        if renderer:
            renderer.close()


if __name__ == "__main__":
    sys.exit(main())
