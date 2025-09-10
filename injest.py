#!/usr/bin/python

# Copyright (c) 2025, Paolo Tassotti <paolo.tassotti@gmail.com>
# GNU General Public License v3.0+ (see LICENSES/GPL-3.0-or-later.txt or https://www.gnu.org/licenses/gpl-3.0.txt)


import os
from pathlib import Path


from html_to_text import extract_text_from_html
from pdf_to_text import extract_text_from_pdf



# dispatcher by file extension
extractors = {
    ".html": lambda path: extract_text_from_html(path),
    ".htm":  lambda path: extract_text_from_html(path),
    ".pdf":  lambda path: extract_text_from_pdf(path),
    #".txt":  lambda path: extract_text_from_txt(path),
    # ".docx": lambda path: extract_text_from_docx(path),
    # ".xlsx": lambda path: extract_text_from_xlsx(path),
}


# traverse directory recursively and process files with appropriate extractor
def process_knowledge_dir(directory):

  processed_docs = []

  for root, _, files in os.walk(directory):

    for filename in files:

      ext = Path(filename).suffix.lower()
      extractor = extractors.get(ext)
      if extractor:
        file_path = os.path.join(root, filename)
        print(f"Processing: {file_path}")
        try:
          doc = extractor(file_path)
          processed_docs.append(doc)
        except Exception as e:
          print(f"Failed to process {file_path}: {e}")

  return processed_docs

