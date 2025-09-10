#!/usr/bin/python

# Copyright (c) 2024, Paolo Tassotti <paolo.tassotti@sourcesense.com>
# GNU General Public License v3.0+ (see LICENSES/GPL-3.0-or-later.txt or https://www.gnu.org/licenses/gpl-3.0.txt)

import re

from bs4 import BeautifulSoup


def clean_wikipedia_text(soup):

  for tag in soup.select(
    'span.reference, div.reflist, h3, h4, h5, h6, span.mw-editsection, sup.reference, '
    '.toc, .interlanguage-link, #siteNotice, .vector-header, .vector-menu, .mw-head, '
    '.mw-wiki-logo, .siteNotice, .jump-to-link, .hatnote, .ambox, .metadata, .shortdescription, .infobox, '
    '.vector-header-container, .vector-sitenotice-container, .vector-main-menu-container, #mw-navigation, '
    '.vector-sticky-pinned-container, .vector-column-start, .vector-column-end, .vector-page-toolbar, '
    '.vector-body-before-content, .mw-indicators, '
    '.mw-jump-link'
  ):
    tag.decompose()


  # Extract all visible text
  content_div = soup.select_one('.mw-parser-output')
  if content_div:
    text = content_div.get_text(separator=" ", strip=True)
  else:
    text = soup.get_text(separator=" ", strip=True)

  # Remove language links / headers typical of Wikipedia
  text = re.sub(r"(\d+\s+languages.*?)(\n|$)", "", text, flags=re.IGNORECASE)

  # Remove [1], [2], [3], ... citations
  text = re.sub(r"(\[\s*\d+\s*\])+", "", text)

    
  # Remove [citation needed]
  text = re.sub(r"\[\s*citation needed\s*\]", "", text, flags=re.IGNORECASE)


  sections_to_remove = [
    r'See also:.*',
    r'External links.*',
    r'References.*',
    r'Categories:.*',
    r'Hidden categories:.*',
    r'Authority control.*'
  ]

  pattern = '|'.join(sections_to_remove)
  text = re.sub(pattern, '', text, flags=re.IGNORECASE|re.DOTALL)


  # Collapse extra spaces
  text = re.sub(r"\s+", " ", text)
  text = re.sub(r"\n{2,}", "\n", text)
    
  # adjuct punctuation
  # Remove spaces before punctuation (e.g. "coin toss .")
  text = re.sub(r"\s+([.,;:!?])", r"\1", text)

  # Fix spaces before punctuation and inside parentheses
  text = re.sub(r'\s*([.,;:!?])', r'\1', text)
  text = re.sub(r'\(\s+', '(', text)
  text = re.sub(r'\s+\)', ')', text)


  # Remove "show v t e" sections and lists of items in all caps / no punctuation
  text = re.sub(r'show v t e.*?(?=(\n|$))', '', text, flags=re.IGNORECASE|re.DOTALL)

  # Remove "Retrieved from" lines
  text = re.sub(r'Retrieved from .*', '', text, flags=re.IGNORECASE)

  # Remove "Categories:" and "Hidden categories:" sections
  text = re.sub(r'Categories:.*', '', text, flags=re.IGNORECASE)
  text = re.sub(r'Hidden categories:.*', '', text, flags=re.IGNORECASE)

  # Remove "Authority control" sections
  text = re.sub(r'Authority control.*', '', text, flags=re.IGNORECASE)


  return text.strip()


def extract_text_from_html(file_path):

  with open(file_path, 'r', encoding='utf-8') as f:
    html = f.read()

  soup = BeautifulSoup(html, 'html.parser')

  # Remove unwanted elements
  for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'table', 'figure']):
    tag.decompose()


  # detect wikipedia
  is_wikipedia = "wikipedia.org" in html or "mw-parser-output" in html

  if is_wikipedia:
    body_text = clean_wikipedia_text(soup)
  else:
    body_text = soup.get_text(separator="\n", strip=True)


  # Get page title
  title = soup.title.string if soup.title else ''


  # collapse multiple newlines
  body_text = re.sub(r"\n{2,}", "\n", body_text)


  # Extract links
  links = [a['href'] for a in soup.find_all('a', href=True)]

  # Extract tables info
  table_data = []
  for row in soup.find_all('tr'):
    cells = row.find_all(['td', 'th'])
    table_data.append([cell.get_text(strip=True) for cell in cells])

  return {
    'title': title,
    'body_text': body_text,
    'links': links,
    'tables': table_data
  }

