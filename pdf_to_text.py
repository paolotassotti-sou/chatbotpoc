#!/usr/bin/python

# Copyright (c) 2024, Paolo Tassotti <paolo.tassotti@sourcesense.com>
# GNU General Public License v3.0+ (see LICENSES/GPL-3.0-or-later.txt or https://www.gnu.org/licenses/gpl-3.0.txt)

from pathlib import Path
import fitz  # PyMuPDF


def extract_text_from_pdf(file_path):

  doc = fitz.open(file_path)
  raw_lines = []

  # Extract text line by line
  for page in doc:
    text = page.get_text("text")
    raw_lines.extend(line.strip() for line in text.split("\n") if line.strip())

  cleaned_lines = []
  current_bullet = ""

  for line in raw_lines:
    if line.startswith("•"):  # new bullet
      if current_bullet:
        # normalize internal newlines
        cleaned_lines.append(current_bullet.replace("\n", " ").strip())
      current_bullet = line
    else:
      if current_bullet:
        current_bullet += " " + line
      else:
        # stray line before any bullet
        cleaned_lines.append(line.replace("\n", " ").strip())

  # Append the last bullet if any
  if current_bullet:
    cleaned_lines.append(current_bullet.replace("\n", " ").strip())

  # Optionally, remove double spaces
  cleaned_lines = [l.replace("  ", " ") for l in cleaned_lines]

  print(cleaned_lines)

  return {
    "title": Path(file_path).stem,
    "full_text": "\n\n".join(cleaned_lines),
    "links": [],
    "tables": []
  }

