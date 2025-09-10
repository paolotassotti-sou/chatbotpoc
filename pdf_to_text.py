#!/usr/bin/python

# Copyright (c) 2024, Paolo Tassotti <paolo.tassotti@sourcesense.com>
# GNU General Public License v3.0+ (see LICENSES/GPL-3.0-or-later.txt or https://www.gnu.org/licenses/gpl-3.0.txt)

from pathlib import Path
from PyPDF2 import PdfReader


def extract_text_from_pdf(file_path):
  reader = PdfReader(file_path)
  text = "\n".join([page.extract_text() or "" for page in reader.pages])
  return {"title": Path(file_path).stem, "body_text": text, "links": [], "tables": []}

