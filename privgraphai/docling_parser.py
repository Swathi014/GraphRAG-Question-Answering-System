"""Docling-based structural extraction for PrivGraphAI."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

logger = logging.getLogger("privgraphai.docling_parser")

ElementType = Literal["text", "table", "figure", "formula"]


@dataclass
class MultimodalElement:
    element_id: str
    element_type: ElementType
    page_number: Optional[int]
    content: str
    image_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class DoclingParser:
    def __init__(self, image_export_dir: str | Path = "./_extracted_images"):
        self.image_export_dir = Path(image_export_dir)
        self.image_export_dir.mkdir(parents=True, exist_ok=True)
        self._converter: Any = None

    def _get_converter(self):
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter
            except ImportError as exc:
                raise ImportError(
                    "docling is not installed. Install it with `pip install docling`."
                ) from exc
            self._converter = DocumentConverter()
        return self._converter

    def parse(self, source: str | Path) -> list[MultimodalElement]:
        converter = self._get_converter()
        logger.info("Parsing document with Docling: %s", source)
        result = converter.convert(str(source))
        doc = result.document

        elements: list[MultimodalElement] = []
        elements.extend(self._extract_text_and_formulas(doc))
        elements.extend(self._extract_tables(doc))
        elements.extend(self._extract_figures(doc))

        logger.info(
            "Extracted %d elements (%d text, %d table, %d figure, %d formula)",
            len(elements),
            sum(e.element_type == "text" for e in elements),
            sum(e.element_type == "table" for e in elements),
            sum(e.element_type == "figure" for e in elements),
            sum(e.element_type == "formula" for e in elements),
        )
        return elements

    def _extract_text_and_formulas(self, doc) -> list[MultimodalElement]:
        elements: list[MultimodalElement] = []
        for i, item in enumerate(getattr(doc, "texts", [])):
            label = str(getattr(item, "label", "text")).lower()
            element_type: ElementType = "formula" if "formula" in label else "text"
            elements.append(
                MultimodalElement(
                    element_id=f"text-{i}",
                    element_type=element_type,
                    page_number=self._first_page_number(item),
                    content=getattr(item, "text", "") or "",
                    metadata={"label": label},
                )
            )
        return elements

    def _extract_tables(self, doc) -> list[MultimodalElement]:
        elements: list[MultimodalElement] = []
        for i, table in enumerate(getattr(doc, "tables", [])):
            try:
                content = table.export_to_markdown(doc=doc)
            except Exception:
                content = str(table)
            elements.append(
                MultimodalElement(
                    element_id=f"table-{i}",
                    element_type="table",
                    page_number=self._first_page_number(table),
                    content=content,
                )
            )
        return elements

    def _extract_figures(self, doc) -> list[MultimodalElement]:
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

            elements.append(
                MultimodalElement(
                    element_id=f"figure-{i}",
                    element_type="figure",
                    page_number=self._first_page_number(picture),
                    content=caption,
                    image_path=image_path,
                )
            )
        return elements

    @staticmethod
    def _first_page_number(item) -> Optional[int]:
        prov = getattr(item, "prov", None)
        if prov:
            try:
                return prov[0].page_no
            except (IndexError, AttributeError):
                return None
        return None
