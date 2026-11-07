"""Docling-based structural extraction for PrivGraphAI.

PROPOSED version. Changes from the original (all backwards compatible):
  1. Formula enrichment is switched on, so formula elements get their LaTeX text
     (the original produced formula elements with empty content).
  2. Picture images are generated, so figure elements get an image_path and the
     vision model (Moondream) actually runs on them.
  3. Text that sits inside a figure (axis ticks, bar values, category labels) is
     removed from the loose text elements and attached to its figure instead,
     grouped into left-to-right columns so that a bar's value stays next to its label.
  4. Page headers/footers (page numbers etc.) are dropped.
  5. Formula LaTeX is tidied (`H _ { r a w }` becomes `H_{raw}`).
  6. Round 2: figure text is written as "label = value" pairs ("G1 = 62.4; G2 = 71.8") when a
     column holds one number and a label, which a small model reads far more reliably than
     a flat list of numbers. Assumes vertical bar-chart-like layouts; other text is kept as is.
  7. Round 3 (after the research-paper test in the web UI): long paragraphs are split into
     overlapping, sentence-aware chunks of about 900 characters, and large tables are split into one
     element per row ("Table 2 - row 7: Model = G-Retriever; Pruning = PCST; ..."). A whole paragraph or
     a 30-row table as ONE element embeds as a blur, is cut by the evidence budget, and had only its first
     1,200 characters sent to graph extraction. Chunk ids look like text-549_1, text-549_2; row ids like
     table-3_12.
All new behaviour can be switched off with constructor flags.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

logger = logging.getLogger("privgraphai.docling_parser")

ElementType = Literal["text", "table", "figure", "formula"]

# Some PDF fonts map ligatures to wrong Unicode characters in the text layer.
_LIGATURE_FIXES = str.maketrans(
    {
        "\u019f": "ti",  # Ɵ  (ApplicaƟon -> Application)
        "\u014c": "ft",  # Ō  (soŌware -> software)
        "\u015b": "tt",  # ś  (Kośayam -> Kottayam)
        "\u01a9": "tt",  # Ʃ  (aƩached -> attached)
        "\uf0b7": "- ",  # private-use bullet
    }
)

_PAGE_FURNITURE_LABELS = ("page_footer", "page_header")
_MAX_TEXT_CHARS = 900        # longer paragraphs are split into chunks of about this size
_TEXT_OVERLAP = 120          # characters carried over between consecutive chunks
_TABLE_WHOLE_MAX_CHARS = 1500  # tables up to this size stay one element; bigger ones become one element per row
_NUMBER = re.compile(r"^[+-]?\d+(?:\.\d+)?%?$")


def _tidy_latex(s: str) -> str:
    """Collapse the letter-by-letter spacing in model-generated LaTeX."""
    s = s.strip()

    def brace(m):
        toks = m.group(1).split()
        if toks and all(len(t) == 1 for t in toks):
            return "{" + "".join(toks) + "}"
        return "{" + " ".join(toks) + "}"

    s = re.sub(r"\{([^{}]*)\}", brace, s)
    s = re.sub(r"\s*_\s*", "_", s)
    return re.sub(r"\s+", " ", s)


def _split_text(text: str, max_chars: int = _MAX_TEXT_CHARS, overlap: int = _TEXT_OVERLAP) -> list[str]:
    """Split text into sentence-aware chunks of at most about `max_chars`, with a small overlap."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    # a single very long "sentence" (a list, a table dump) is hard-split so no chunk explodes
    pieces: list[str] = []
    for sent in sentences:
        while len(sent) > max_chars:
            pieces.append(sent[:max_chars])
            sent = sent[max_chars:]
        if sent:
            pieces.append(sent)

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for piece in pieces:
        if current and size + len(piece) + 1 > max_chars:
            chunks.append(" ".join(current))
            # carry the last sentence(s) over, up to `overlap` characters
            carry: list[str] = []
            carried = 0
            for prev in reversed(current):
                if carried + len(prev) > overlap and carry:
                    break
                if len(prev) > overlap:
                    break
                carry.insert(0, prev)
                carried += len(prev) + 1
            current, size = carry, carried
        current.append(piece)
        size += len(piece) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def _column_name(col) -> str:
    if isinstance(col, tuple):
        parts: list[str] = []
        for part in col:
            part = str(part).strip()
            if part and not part.lower().startswith("unnamed") and part not in parts:
                parts.append(part)
        return " / ".join(parts)
    return str(col).strip()


@dataclass
class MultimodalElement:
    element_id: str
    element_type: ElementType
    page_number: Optional[int]
    content: str
    image_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class DoclingParser:
    def __init__(
        self,
        image_export_dir: str | Path = "./_extracted_images",
        force_ocr: bool = False,
        enrich_formulas: bool = True,
        export_picture_images: bool = True,
        drop_page_furniture: bool = True,
        figure_column_tolerance: float = 0.06,
        max_text_chars: int = _MAX_TEXT_CHARS,
        text_overlap: int = _TEXT_OVERLAP,
        split_large_tables: bool = True,
    ):
        self.image_export_dir = Path(image_export_dir)
        self.image_export_dir.mkdir(parents=True, exist_ok=True)
        self.force_ocr = force_ocr
        self.enrich_formulas = enrich_formulas
        self.export_picture_images = export_picture_images
        self.drop_page_furniture = drop_page_furniture
        # Texts whose centres are within this fraction of the figure width are
        # treated as one column (e.g. a bar's value and its category label).
        self.figure_column_tolerance = figure_column_tolerance
        self.max_text_chars = max_text_chars          # 0 switches paragraph splitting off
        self.text_overlap = text_overlap
        self.split_large_tables = split_large_tables
        self._converter: Any = None

    # ------------------------------------------------------------------ setup
    def _get_converter(self):
        if self._converter is None:
            try:
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.document_converter import DocumentConverter, PdfFormatOption
            except ImportError as exc:
                raise ImportError(
                    "docling is not installed. Install it with `pip install docling`."
                ) from exc

            pipeline_options = PdfPipelineOptions()
            if self.force_ocr:
                # Ignore the PDF's embedded text layer and OCR every page.
                pipeline_options.do_ocr = True
                pipeline_options.ocr_options.force_full_page_ocr = True
            if self.enrich_formulas:
                # Needs the CodeFormula model: download it once while online.
                pipeline_options.do_formula_enrichment = True
            if self.export_picture_images:
                pipeline_options.generate_picture_images = True
                pipeline_options.images_scale = 2.0

            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
        return self._converter

    # ------------------------------------------------------------------ parse
    def parse(self, source: str | Path) -> list[MultimodalElement]:
        converter = self._get_converter()
        logger.info("Parsing document with Docling: %s", source)
        result = converter.convert(str(source))
        doc = result.document

        figure_texts, inside_figure = self._figure_text_map(doc)

        elements: list[MultimodalElement] = []
        elements.extend(self._extract_text_and_formulas(doc, skip=inside_figure))
        elements.extend(self._extract_tables(doc))
        elements.extend(self._extract_figures(doc, figure_texts))

        logger.info(
            "Extracted %d elements (%d text, %d table, %d figure, %d formula)",
            len(elements),
            sum(e.element_type == "text" for e in elements),
            sum(e.element_type == "table" for e in elements),
            sum(e.element_type == "figure" for e in elements),
            sum(e.element_type == "formula" for e in elements),
        )
        return elements

    # ------------------------------------------------------- text / formulas
    def _extract_text_and_formulas(self, doc, skip: Optional[set] = None) -> list[MultimodalElement]:
        skip = skip or set()
        elements: list[MultimodalElement] = []
        for i, item in enumerate(getattr(doc, "texts", [])):
            if i in skip:
                continue  # belongs to a figure; attached to that figure instead
            label = str(getattr(item, "label", "text")).lower()
            if self.drop_page_furniture and any(l in label for l in _PAGE_FURNITURE_LABELS):
                continue
            element_type: ElementType = "formula" if "formula" in label else "text"
            content = (getattr(item, "text", "") or "").translate(_LIGATURE_FIXES)
            if element_type == "formula":
                content = _tidy_latex(content)
            page = self._first_page_number(item)
            chunks = (
                _split_text(content, self.max_text_chars, self.text_overlap)
                if element_type == "text" and self.max_text_chars and len(content) > self.max_text_chars
                else [content]
            )
            if len(chunks) == 1:
                elements.append(
                    MultimodalElement(
                        element_id=f"text-{i}",
                        element_type=element_type,
                        page_number=page,
                        content=content,
                        metadata={"label": label},
                    )
                )
            else:
                for k, chunk in enumerate(chunks, start=1):
                    elements.append(
                        MultimodalElement(
                            element_id=f"text-{i}_{k}",
                            element_type="text",
                            page_number=page,
                            content=chunk,
                            metadata={"label": label, "parent": f"text-{i}", "chunk": k, "of": len(chunks)},
                        )
                    )
        return elements

    # ---------------------------------------------------------------- tables
    def _extract_tables(self, doc) -> list[MultimodalElement]:
        elements: list[MultimodalElement] = []
        for i, table in enumerate(getattr(doc, "tables", [])):
            try:
                content = table.export_to_markdown(doc=doc)
            except Exception:
                content = str(table)
            page = self._first_page_number(table)

            rows = self._table_rows(table, doc) if self.split_large_tables and len(content) > _TABLE_WHOLE_MAX_CHARS else None
            if not rows:
                elements.append(
                    MultimodalElement(element_id=f"table-{i}", element_type="table", page_number=page, content=content)
                )
                continue

            caption = ""
            caption_fn = getattr(table, "caption_text", None)
            if callable(caption_fn):
                try:
                    caption = (caption_fn(doc) or "").strip()
                except Exception:
                    caption = ""
            caption = caption or f"Table {i + 1}"
            columns, body = rows
            elements.append(
                MultimodalElement(
                    element_id=f"table-{i}",
                    element_type="table",
                    page_number=page,
                    content=f"{caption}. Columns: {', '.join(c for c in columns if c)}. "
                            f"The table has {len(body)} rows, listed as separate elements.",
                    metadata={"table_overview": True},
                )
            )
            for j, values in enumerate(body, start=1):
                cells = "; ".join(f"{col} = {val}" for col, val in zip(columns, values) if col and val)
                if not cells:
                    continue
                elements.append(
                    MultimodalElement(
                        element_id=f"table-{i}_{j}",
                        element_type="table",
                        page_number=page,
                        content=f"{caption} - row {j}: {cells}",
                        metadata={"table_row": True, "table": f"table-{i}", "row": j},
                    )
                )
        return elements

    @staticmethod
    def _table_rows(table, doc):
        """(column names, list of row value lists) from Docling's dataframe export, or None."""
        try:
            try:
                df = table.export_to_dataframe(doc=doc)
            except TypeError:
                df = table.export_to_dataframe()
        except Exception as exc:
            logger.warning("Could not export a table as rows (%s); keeping it as one element", exc)
            return None
        if df is None or len(df) == 0:
            return None
        columns = [_column_name(c) for c in df.columns]
        body = [[str(v).strip() if v is not None and str(v).lower() != "nan" else "" for v in row]
                for row in df.itertuples(index=False, name=None)]
        return columns, body

    # --------------------------------------------------------------- figures
    def _extract_figures(self, doc, figure_texts: Optional[dict] = None) -> list[MultimodalElement]:
        figure_texts = figure_texts or {}
        elements: list[MultimodalElement] = []
        for i, picture in enumerate(getattr(doc, "pictures", [])):
            image_path = None
            try:
                pil_image = picture.get_image(doc)
                if pil_image is not None:
                    image_path = str(self.image_export_dir / f"figure-{i}.png")
                    pil_image.save(image_path)
            except Exception as exc:
                logger.warning("Could not export image for figure-%d: %s", i, exc)

            caption = ""
            caption_fn = getattr(picture, "caption_text", None)
            if callable(caption_fn):
                try:
                    caption = caption_fn(doc) or ""
                except Exception:
                    caption = ""

            metadata = {}
            if figure_texts.get(i):
                metadata["figure_text"] = figure_texts[i]

            elements.append(
                MultimodalElement(
                    element_id=f"figure-{i}",
                    element_type="figure",
                    page_number=self._first_page_number(picture),
                    content=caption,
                    image_path=image_path,
                    metadata=metadata,
                )
            )
        return elements

    # ---------------------------------------------- text inside figure bboxes
    def _figure_text_map(self, doc):
        """Return ({picture_index: 'col1 | col2 | ...'}, {text indices inside figures})."""
        pictures = list(getattr(doc, "pictures", []))
        texts = list(getattr(doc, "texts", []))
        figure_texts: dict[int, str] = {}
        inside: set[int] = set()
        if not pictures:
            return figure_texts, inside

        pic_geo = []
        for pic in pictures:
            prov = getattr(pic, "prov", None)
            pic_geo.append((prov[0].page_no, prov[0].bbox) if prov else (None, None))

        members: dict[int, list] = {i: [] for i in range(len(pictures))}
        for j, item in enumerate(texts):
            prov = getattr(item, "prov", None)
            txt = (getattr(item, "text", "") or "").strip()
            if not prov or not txt:
                continue
            page, bbox = prov[0].page_no, prov[0].bbox
            xc, yc = (bbox.l + bbox.r) / 2.0, (bbox.t + bbox.b) / 2.0
            for pi, (ppage, pbox) in enumerate(pic_geo):
                if pbox is None or ppage != page:
                    continue
                lo, hi = sorted((pbox.t, pbox.b))
                if pbox.l <= xc <= pbox.r and lo <= yc <= hi:
                    origin = str(getattr(bbox, "coord_origin", "")).upper()
                    members[pi].append((txt, xc, yc, "BOTTOMLEFT" in origin))
                    inside.add(j)
                    break

        for pi, items in members.items():
            if not items:
                continue
            pbox = pic_geo[pi][1]
            figure_texts[pi] = self._columns(items, max(pbox.r - pbox.l, 1.0))
        return figure_texts, inside

    def _columns(self, items, width: float) -> str:
        """Group texts into left-to-right columns, then describe them as label = value pairs."""
        tol = self.figure_column_tolerance * width
        cols: list[dict] = []
        for txt, xc, yc, bottom_left in sorted(items, key=lambda it: it[1]):
            if cols and abs(xc - cols[-1]["x"]) <= tol:
                col = cols[-1]
                col["items"].append((txt, yc, bottom_left))
                col["xs"].append(xc)
                col["x"] = sum(col["xs"]) / len(col["xs"])
            else:
                cols.append({"x": xc, "xs": [xc], "items": [(txt, yc, bottom_left)]})

        pairs: list[str] = []
        other: list[str] = []
        for col in cols:
            ordered = sorted(col["items"], key=lambda it: -it[1] if it[2] else it[1])
            texts = [t for t, _, _ in ordered]
            numbers = [t for t in texts if _NUMBER.match(t)]
            labels = [t for t in texts if not _NUMBER.match(t)]
            if len(numbers) == 1 and labels:
                pairs.append(f"{labels[0]} = {numbers[0]}")
                other.extend(labels[1:])
            else:
                other.append(" ".join(texts))

        parts = []
        if pairs:
            parts.append("Values by label, left to right: " + "; ".join(pairs))
        if other:
            parts.append("Other text: " + " | ".join(other))
        return ". ".join(parts)

    @staticmethod
    def _first_page_number(item) -> Optional[int]:
        prov = getattr(item, "prov", None)
        if prov:
            try:
                return prov[0].page_no
            except (IndexError, AttributeError):
                return None
        return None
