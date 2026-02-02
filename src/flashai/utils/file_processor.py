"""
Universal File Processor and Optimizer

Handles uploading, processing, and optimizing various file types:
- Images: PNG, JPEG, GIF, WebP, SVG, BMP, TIFF
- Documents: PDF, DOCX, DOC, TXT, RTF, ODT
- Data: JSON, CSV, YAML, XML, Excel (XLSX, XLS)
- Code: Python, JavaScript, TypeScript, Java, C++, Go, Rust, etc.
- Archives: ZIP, TAR, GZ, 7Z
- Media: Audio (MP3, WAV), Video (MP4, WebM)
- Markup: HTML, Markdown, LaTeX
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import mimetypes
import os
import re
import shutil
import tempfile
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

logger = logging.getLogger(__name__)


class FileCategory(Enum):
    """Categories of files."""
    IMAGE = "image"
    DOCUMENT = "document"
    DATA = "data"
    CODE = "code"
    ARCHIVE = "archive"
    AUDIO = "audio"
    VIDEO = "video"
    MARKUP = "markup"
    UNKNOWN = "unknown"


@dataclass
class FileMetadata:
    """Metadata about a processed file."""
    filename: str
    original_size: int
    processed_size: int
    mime_type: str
    category: FileCategory
    extension: str
    checksum: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    optimization_applied: list[str] = field(default_factory=list)
    extracted_content: Optional[str] = None
    extracted_metadata: dict[str, Any] = field(default_factory=dict)
    learning_data: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "original_size": self.original_size,
            "processed_size": self.processed_size,
            "mime_type": self.mime_type,
            "category": self.category.value,
            "extension": self.extension,
            "checksum": self.checksum,
            "created_at": self.created_at,
            "optimization_applied": self.optimization_applied,
            "has_extracted_content": self.extracted_content is not None,
            "extracted_metadata": self.extracted_metadata,
            "has_learning_data": self.learning_data is not None,
        }


@dataclass
class ProcessingResult:
    """Result of file processing."""
    success: bool
    metadata: Optional[FileMetadata] = None
    processed_data: Optional[bytes] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "errors": self.errors,
            "warnings": self.warnings,
            "processed_size": len(self.processed_data) if self.processed_data else 0,
        }


@dataclass
class OptimizationOptions:
    """Options for file optimization."""
    compress: bool = True
    extract_text: bool = True
    generate_learning_data: bool = True
    max_image_dimension: int = 2048
    image_quality: int = 85
    strip_metadata: bool = False
    minify_code: bool = False
    preserve_original: bool = True


# File extension mappings
EXTENSION_CATEGORIES = {
    # Images
    ".png": FileCategory.IMAGE,
    ".jpg": FileCategory.IMAGE,
    ".jpeg": FileCategory.IMAGE,
    ".gif": FileCategory.IMAGE,
    ".webp": FileCategory.IMAGE,
    ".svg": FileCategory.IMAGE,
    ".bmp": FileCategory.IMAGE,
    ".tiff": FileCategory.IMAGE,
    ".ico": FileCategory.IMAGE,

    # Documents
    ".pdf": FileCategory.DOCUMENT,
    ".doc": FileCategory.DOCUMENT,
    ".docx": FileCategory.DOCUMENT,
    ".txt": FileCategory.DOCUMENT,
    ".rtf": FileCategory.DOCUMENT,
    ".odt": FileCategory.DOCUMENT,
    ".pages": FileCategory.DOCUMENT,

    # Data
    ".json": FileCategory.DATA,
    ".csv": FileCategory.DATA,
    ".yaml": FileCategory.DATA,
    ".yml": FileCategory.DATA,
    ".xml": FileCategory.DATA,
    ".xlsx": FileCategory.DATA,
    ".xls": FileCategory.DATA,
    ".tsv": FileCategory.DATA,
    ".parquet": FileCategory.DATA,
    ".sqlite": FileCategory.DATA,
    ".db": FileCategory.DATA,

    # Code
    ".py": FileCategory.CODE,
    ".js": FileCategory.CODE,
    ".ts": FileCategory.CODE,
    ".jsx": FileCategory.CODE,
    ".tsx": FileCategory.CODE,
    ".java": FileCategory.CODE,
    ".cpp": FileCategory.CODE,
    ".c": FileCategory.CODE,
    ".h": FileCategory.CODE,
    ".hpp": FileCategory.CODE,
    ".go": FileCategory.CODE,
    ".rs": FileCategory.CODE,
    ".rb": FileCategory.CODE,
    ".php": FileCategory.CODE,
    ".swift": FileCategory.CODE,
    ".kt": FileCategory.CODE,
    ".scala": FileCategory.CODE,
    ".r": FileCategory.CODE,
    ".sql": FileCategory.CODE,
    ".sh": FileCategory.CODE,
    ".bash": FileCategory.CODE,
    ".ps1": FileCategory.CODE,
    ".lua": FileCategory.CODE,
    ".pl": FileCategory.CODE,
    ".cs": FileCategory.CODE,
    ".fs": FileCategory.CODE,
    ".hs": FileCategory.CODE,
    ".ex": FileCategory.CODE,
    ".exs": FileCategory.CODE,
    ".erl": FileCategory.CODE,
    ".clj": FileCategory.CODE,
    ".lisp": FileCategory.CODE,
    ".vue": FileCategory.CODE,
    ".svelte": FileCategory.CODE,

    # Archives
    ".zip": FileCategory.ARCHIVE,
    ".tar": FileCategory.ARCHIVE,
    ".gz": FileCategory.ARCHIVE,
    ".tgz": FileCategory.ARCHIVE,
    ".bz2": FileCategory.ARCHIVE,
    ".7z": FileCategory.ARCHIVE,
    ".rar": FileCategory.ARCHIVE,

    # Audio
    ".mp3": FileCategory.AUDIO,
    ".wav": FileCategory.AUDIO,
    ".ogg": FileCategory.AUDIO,
    ".flac": FileCategory.AUDIO,
    ".m4a": FileCategory.AUDIO,
    ".aac": FileCategory.AUDIO,

    # Video
    ".mp4": FileCategory.VIDEO,
    ".webm": FileCategory.VIDEO,
    ".avi": FileCategory.VIDEO,
    ".mov": FileCategory.VIDEO,
    ".mkv": FileCategory.VIDEO,
    ".flv": FileCategory.VIDEO,

    # Markup
    ".html": FileCategory.MARKUP,
    ".htm": FileCategory.MARKUP,
    ".md": FileCategory.MARKUP,
    ".markdown": FileCategory.MARKUP,
    ".rst": FileCategory.MARKUP,
    ".tex": FileCategory.MARKUP,
    ".latex": FileCategory.MARKUP,
    ".adoc": FileCategory.MARKUP,
    ".org": FileCategory.MARKUP,
}


class FileProcessor(ABC):
    """Base class for file processors."""

    @abstractmethod
    def supports(self, category: FileCategory) -> bool:
        """Check if this processor supports the category."""
        pass

    @abstractmethod
    async def process(
        self,
        data: bytes,
        filename: str,
        options: OptimizationOptions,
    ) -> ProcessingResult:
        """Process the file data."""
        pass


class ImageProcessor(FileProcessor):
    """Processor for image files."""

    def supports(self, category: FileCategory) -> bool:
        return category == FileCategory.IMAGE

    async def process(
        self,
        data: bytes,
        filename: str,
        options: OptimizationOptions,
    ) -> ProcessingResult:
        errors = []
        warnings = []
        optimizations = []
        processed_data = data
        extracted_text = None

        try:
            from PIL import Image

            img = Image.open(io.BytesIO(data))
            original_format = img.format
            original_size = img.size

            # Resize if too large
            if options.compress and max(img.size) > options.max_image_dimension:
                ratio = options.max_image_dimension / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                optimizations.append(f"resized from {original_size} to {new_size}")

            # Convert RGBA to RGB for JPEG
            output_format = original_format or "PNG"
            if output_format == "JPEG" and img.mode == "RGBA":
                img = img.convert("RGB")
                optimizations.append("converted RGBA to RGB")

            # Strip metadata if requested
            if options.strip_metadata:
                # Create new image without metadata
                img_data = list(img.getdata())
                img_no_meta = Image.new(img.mode, img.size)
                img_no_meta.putdata(img_data)
                img = img_no_meta
                optimizations.append("stripped metadata")

            # Save optimized
            output = io.BytesIO()
            save_kwargs = {}
            if output_format == "JPEG":
                save_kwargs["quality"] = options.image_quality
                save_kwargs["optimize"] = True
            elif output_format == "PNG":
                save_kwargs["optimize"] = True

            img.save(output, format=output_format, **save_kwargs)
            processed_data = output.getvalue()

            if len(processed_data) < len(data):
                optimizations.append(f"compressed {len(data)} -> {len(processed_data)} bytes")

            # Try OCR for text extraction
            if options.extract_text:
                extracted_text = await self._extract_text_ocr(img)

        except ImportError:
            warnings.append("PIL not available, skipping image optimization")
        except Exception as e:
            errors.append(f"Image processing error: {str(e)}")

        ext = Path(filename).suffix.lower()
        mime_type, _ = mimetypes.guess_type(filename)

        metadata = FileMetadata(
            filename=filename,
            original_size=len(data),
            processed_size=len(processed_data),
            mime_type=mime_type or "image/unknown",
            category=FileCategory.IMAGE,
            extension=ext,
            checksum=hashlib.md5(processed_data).hexdigest(),
            optimization_applied=optimizations,
            extracted_content=extracted_text,
        )

        return ProcessingResult(
            success=len(errors) == 0,
            metadata=metadata,
            processed_data=processed_data,
            errors=errors,
            warnings=warnings,
        )

    async def _extract_text_ocr(self, img) -> Optional[str]:
        """Extract text from image using OCR."""
        try:
            import pytesseract
            text = pytesseract.image_to_string(img)
            return text.strip() if text.strip() else None
        except ImportError:
            return None
        except Exception:
            return None


class DocumentProcessor(FileProcessor):
    """Processor for document files (PDF, DOCX, etc.)."""

    def supports(self, category: FileCategory) -> bool:
        return category == FileCategory.DOCUMENT

    async def process(
        self,
        data: bytes,
        filename: str,
        options: OptimizationOptions,
    ) -> ProcessingResult:
        errors = []
        warnings = []
        optimizations = []
        processed_data = data
        extracted_text = None
        extracted_metadata = {}

        ext = Path(filename).suffix.lower()

        try:
            if ext == ".pdf":
                extracted_text, extracted_metadata = await self._process_pdf(data, options)
            elif ext in [".docx", ".doc"]:
                extracted_text, extracted_metadata = await self._process_docx(data, options)
            elif ext == ".txt":
                extracted_text = data.decode("utf-8", errors="ignore")
            elif ext == ".rtf":
                extracted_text = await self._process_rtf(data)
            else:
                warnings.append(f"No specific processor for {ext}, treating as text")
                extracted_text = data.decode("utf-8", errors="ignore")

        except Exception as e:
            errors.append(f"Document processing error: {str(e)}")

        mime_type, _ = mimetypes.guess_type(filename)

        metadata = FileMetadata(
            filename=filename,
            original_size=len(data),
            processed_size=len(processed_data),
            mime_type=mime_type or "application/octet-stream",
            category=FileCategory.DOCUMENT,
            extension=ext,
            checksum=hashlib.md5(processed_data).hexdigest(),
            optimization_applied=optimizations,
            extracted_content=extracted_text,
            extracted_metadata=extracted_metadata,
        )

        # Generate learning data from document
        if options.generate_learning_data and extracted_text:
            metadata.learning_data = self._generate_learning_data(extracted_text, filename)

        return ProcessingResult(
            success=len(errors) == 0,
            metadata=metadata,
            processed_data=processed_data,
            errors=errors,
            warnings=warnings,
        )

    async def _process_pdf(self, data: bytes, options: OptimizationOptions) -> tuple[Optional[str], dict]:
        """Extract text and metadata from PDF."""
        try:
            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(data))

            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")

            text = "\n\n".join(text_parts)

            metadata = {}
            if reader.metadata:
                metadata = {
                    "title": reader.metadata.get("/Title", ""),
                    "author": reader.metadata.get("/Author", ""),
                    "subject": reader.metadata.get("/Subject", ""),
                    "pages": len(reader.pages),
                }

            return text.strip(), metadata

        except ImportError:
            return None, {"error": "pypdf not installed"}
        except Exception as e:
            return None, {"error": str(e)}

    async def _process_docx(self, data: bytes, options: OptimizationOptions) -> tuple[Optional[str], dict]:
        """Extract text and metadata from DOCX."""
        try:
            import docx

            doc = docx.Document(io.BytesIO(data))

            text_parts = []
            for para in doc.paragraphs:
                text_parts.append(para.text)

            # Also get text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text_parts.append(cell.text)

            text = "\n".join(text_parts)

            metadata = {
                "paragraphs": len(doc.paragraphs),
                "tables": len(doc.tables),
            }

            if doc.core_properties:
                metadata["title"] = doc.core_properties.title or ""
                metadata["author"] = doc.core_properties.author or ""

            return text.strip(), metadata

        except ImportError:
            return None, {"error": "python-docx not installed"}
        except Exception as e:
            return None, {"error": str(e)}

    async def _process_rtf(self, data: bytes) -> Optional[str]:
        """Extract text from RTF."""
        try:
            from striprtf.striprtf import rtf_to_text
            text = data.decode("utf-8", errors="ignore")
            return rtf_to_text(text)
        except ImportError:
            # Simple RTF text extraction fallback
            text = data.decode("utf-8", errors="ignore")
            # Remove RTF control words
            text = re.sub(r'\\[a-z]+\d*\s?', '', text)
            text = re.sub(r'[{}]', '', text)
            return text.strip()
        except Exception:
            return None

    def _generate_learning_data(self, text: str, filename: str) -> dict[str, Any]:
        """Generate learning data from document text."""
        # Split into sections/paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        examples = []
        for i, para in enumerate(paragraphs[:50]):  # Limit to 50 examples
            if len(para) > 50:  # Only meaningful paragraphs
                examples.append({
                    "query": f"What does the document '{filename}' say about section {i+1}?",
                    "answer": para[:1000],  # Limit answer length
                    "context": {"source": filename, "section": i+1},
                })

        return {
            "examples": examples,
            "source": filename,
            "total_paragraphs": len(paragraphs),
        }


class CodeProcessor(FileProcessor):
    """Processor for source code files."""

    LANGUAGE_COMMENTS = {
        ".py": ("#", '"""', "'''"),
        ".js": ("//", "/*"),
        ".ts": ("//", "/*"),
        ".java": ("//", "/*"),
        ".cpp": ("//", "/*"),
        ".c": ("//", "/*"),
        ".go": ("//", "/*"),
        ".rs": ("//", "/*"),
        ".rb": ("#",),
        ".php": ("//", "/*", "#"),
        ".swift": ("//", "/*"),
        ".kt": ("//", "/*"),
        ".scala": ("//", "/*"),
        ".r": ("#",),
        ".sql": ("--", "/*"),
        ".sh": ("#",),
        ".bash": ("#",),
        ".lua": ("--", "--[["),
        ".hs": ("--", "{-"),
        ".ex": ("#",),
        ".clj": (";",),
    }

    def supports(self, category: FileCategory) -> bool:
        return category == FileCategory.CODE

    async def process(
        self,
        data: bytes,
        filename: str,
        options: OptimizationOptions,
    ) -> ProcessingResult:
        errors = []
        warnings = []
        optimizations = []

        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = data.decode("latin-1")
            except Exception as e:
                return ProcessingResult(
                    success=False,
                    errors=[f"Could not decode file: {str(e)}"],
                )

        processed_text = text
        ext = Path(filename).suffix.lower()

        # Extract metadata
        extracted_metadata = self._extract_code_metadata(text, ext)

        # Minify if requested
        if options.minify_code:
            processed_text = self._minify_code(text, ext)
            if len(processed_text) < len(text):
                optimizations.append("minified code")

        processed_data = processed_text.encode("utf-8")

        mime_type, _ = mimetypes.guess_type(filename)

        metadata = FileMetadata(
            filename=filename,
            original_size=len(data),
            processed_size=len(processed_data),
            mime_type=mime_type or "text/plain",
            category=FileCategory.CODE,
            extension=ext,
            checksum=hashlib.md5(processed_data).hexdigest(),
            optimization_applied=optimizations,
            extracted_content=text,
            extracted_metadata=extracted_metadata,
        )

        # Generate learning data from code
        if options.generate_learning_data:
            metadata.learning_data = self._generate_code_learning_data(text, filename, ext)

        return ProcessingResult(
            success=True,
            metadata=metadata,
            processed_data=processed_data,
            errors=errors,
            warnings=warnings,
        )

    def _extract_code_metadata(self, text: str, ext: str) -> dict[str, Any]:
        """Extract metadata from code."""
        lines = text.split("\n")

        metadata = {
            "lines": len(lines),
            "characters": len(text),
            "language": self._get_language_name(ext),
        }

        # Count functions/classes (simple heuristics)
        if ext == ".py":
            metadata["functions"] = len(re.findall(r"^\s*def\s+\w+", text, re.MULTILINE))
            metadata["classes"] = len(re.findall(r"^\s*class\s+\w+", text, re.MULTILINE))
            metadata["imports"] = len(re.findall(r"^(?:import|from)\s+", text, re.MULTILINE))
        elif ext in [".js", ".ts", ".jsx", ".tsx"]:
            metadata["functions"] = len(re.findall(r"(?:function\s+\w+|const\s+\w+\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=])\s*=>)", text))
            metadata["classes"] = len(re.findall(r"class\s+\w+", text))
            metadata["imports"] = len(re.findall(r"^import\s+", text, re.MULTILINE))
        elif ext in [".java", ".kt", ".scala"]:
            metadata["classes"] = len(re.findall(r"class\s+\w+", text))
            metadata["methods"] = len(re.findall(r"(?:public|private|protected)?\s*\w+\s+\w+\s*\([^)]*\)\s*{", text))

        return metadata

    def _get_language_name(self, ext: str) -> str:
        """Get language name from extension."""
        language_map = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".jsx": "React JSX",
            ".tsx": "React TSX",
            ".java": "Java",
            ".cpp": "C++",
            ".c": "C",
            ".h": "C Header",
            ".go": "Go",
            ".rs": "Rust",
            ".rb": "Ruby",
            ".php": "PHP",
            ".swift": "Swift",
            ".kt": "Kotlin",
            ".scala": "Scala",
            ".r": "R",
            ".sql": "SQL",
            ".sh": "Shell",
            ".bash": "Bash",
            ".lua": "Lua",
            ".hs": "Haskell",
            ".ex": "Elixir",
            ".clj": "Clojure",
            ".cs": "C#",
            ".fs": "F#",
            ".vue": "Vue",
            ".svelte": "Svelte",
        }
        return language_map.get(ext, "Unknown")

    def _minify_code(self, text: str, ext: str) -> str:
        """Simple code minification."""
        lines = text.split("\n")
        minified = []

        for line in lines:
            stripped = line.strip()
            # Skip empty lines and comments
            if not stripped:
                continue

            comment_chars = self.LANGUAGE_COMMENTS.get(ext, ())
            is_comment = any(stripped.startswith(c) for c in comment_chars)

            if not is_comment:
                minified.append(stripped)

        return "\n".join(minified)

    def _generate_code_learning_data(self, text: str, filename: str, ext: str) -> dict[str, Any]:
        """Generate learning data from code."""
        examples = []
        language = self._get_language_name(ext)

        # Extract functions and their docstrings
        if ext == ".py":
            # Find functions with docstrings
            pattern = r'def\s+(\w+)\s*\([^)]*\):\s*(?:"""([^"]+)"""|\'\'\'([^\']+)\'\'\')?'
            matches = re.findall(pattern, text)
            for match in matches[:20]:
                func_name, doc1, doc2 = match
                doc = doc1 or doc2
                if doc:
                    examples.append({
                        "query": f"What does the function {func_name} do in {filename}?",
                        "answer": doc.strip(),
                    })

        # General code understanding example
        examples.append({
            "query": f"Summarize the {language} code in {filename}",
            "answer": f"This is a {language} file with {len(text.split(chr(10)))} lines of code.",
            "context": {"language": language, "file": filename},
        })

        return {
            "examples": examples,
            "source": filename,
            "language": language,
        }


class DataProcessor(FileProcessor):
    """Processor for data files (JSON, CSV, XML, etc.)."""

    def supports(self, category: FileCategory) -> bool:
        return category == FileCategory.DATA

    async def process(
        self,
        data: bytes,
        filename: str,
        options: OptimizationOptions,
    ) -> ProcessingResult:
        errors = []
        warnings = []
        optimizations = []
        processed_data = data
        extracted_metadata = {}

        ext = Path(filename).suffix.lower()

        try:
            if ext == ".json":
                extracted_metadata = await self._process_json(data)
            elif ext == ".csv":
                extracted_metadata = await self._process_csv(data)
            elif ext in [".yaml", ".yml"]:
                extracted_metadata = await self._process_yaml(data)
            elif ext == ".xml":
                extracted_metadata = await self._process_xml(data)
            elif ext in [".xlsx", ".xls"]:
                extracted_metadata = await self._process_excel(data)

        except Exception as e:
            errors.append(f"Data processing error: {str(e)}")

        mime_type, _ = mimetypes.guess_type(filename)

        metadata = FileMetadata(
            filename=filename,
            original_size=len(data),
            processed_size=len(processed_data),
            mime_type=mime_type or "application/octet-stream",
            category=FileCategory.DATA,
            extension=ext,
            checksum=hashlib.md5(processed_data).hexdigest(),
            optimization_applied=optimizations,
            extracted_metadata=extracted_metadata,
        )

        return ProcessingResult(
            success=len(errors) == 0,
            metadata=metadata,
            processed_data=processed_data,
            errors=errors,
            warnings=warnings,
        )

    async def _process_json(self, data: bytes) -> dict:
        """Process JSON file."""
        text = data.decode("utf-8")
        parsed = json.loads(text)

        metadata = {"type": type(parsed).__name__}
        if isinstance(parsed, list):
            metadata["items"] = len(parsed)
        elif isinstance(parsed, dict):
            metadata["keys"] = list(parsed.keys())[:20]

        return metadata

    async def _process_csv(self, data: bytes) -> dict:
        """Process CSV file."""
        import csv

        text = data.decode("utf-8")
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)

        return {
            "rows": len(rows),
            "columns": len(rows[0]) if rows else 0,
            "headers": rows[0] if rows else [],
        }

    async def _process_yaml(self, data: bytes) -> dict:
        """Process YAML file."""
        try:
            import yaml
            text = data.decode("utf-8")
            parsed = yaml.safe_load(text)

            metadata = {"type": type(parsed).__name__}
            if isinstance(parsed, dict):
                metadata["keys"] = list(parsed.keys())[:20]

            return metadata
        except ImportError:
            return {"error": "PyYAML not installed"}

    async def _process_xml(self, data: bytes) -> dict:
        """Process XML file."""
        import xml.etree.ElementTree as ET

        root = ET.fromstring(data)

        return {
            "root_tag": root.tag,
            "children": len(list(root)),
            "attributes": dict(root.attrib),
        }

    async def _process_excel(self, data: bytes) -> dict:
        """Process Excel file."""
        try:
            import openpyxl

            wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True)

            return {
                "sheets": wb.sheetnames,
                "sheet_count": len(wb.sheetnames),
            }
        except ImportError:
            return {"error": "openpyxl not installed"}


class ArchiveProcessor(FileProcessor):
    """Processor for archive files."""

    def supports(self, category: FileCategory) -> bool:
        return category == FileCategory.ARCHIVE

    async def process(
        self,
        data: bytes,
        filename: str,
        options: OptimizationOptions,
    ) -> ProcessingResult:
        errors = []
        warnings = []
        extracted_metadata = {}

        ext = Path(filename).suffix.lower()

        try:
            if ext == ".zip":
                extracted_metadata = await self._process_zip(data)
            elif ext in [".tar", ".gz", ".tgz"]:
                extracted_metadata = await self._process_tar(data, ext)

        except Exception as e:
            errors.append(f"Archive processing error: {str(e)}")

        mime_type, _ = mimetypes.guess_type(filename)

        metadata = FileMetadata(
            filename=filename,
            original_size=len(data),
            processed_size=len(data),
            mime_type=mime_type or "application/octet-stream",
            category=FileCategory.ARCHIVE,
            extension=ext,
            checksum=hashlib.md5(data).hexdigest(),
            extracted_metadata=extracted_metadata,
        )

        return ProcessingResult(
            success=len(errors) == 0,
            metadata=metadata,
            processed_data=data,
            errors=errors,
            warnings=warnings,
        )

    async def _process_zip(self, data: bytes) -> dict:
        """Process ZIP file."""
        with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
            files = zf.namelist()
            total_size = sum(info.file_size for info in zf.infolist())

            return {
                "files": files[:100],  # Limit list
                "file_count": len(files),
                "total_uncompressed_size": total_size,
            }

    async def _process_tar(self, data: bytes, ext: str) -> dict:
        """Process TAR file."""
        import tarfile

        mode = "r"
        if ext in [".gz", ".tgz"]:
            mode = "r:gz"
        elif ext == ".bz2":
            mode = "r:bz2"

        with tarfile.open(fileobj=io.BytesIO(data), mode=mode) as tf:
            members = tf.getnames()

            return {
                "files": members[:100],
                "file_count": len(members),
            }


class MarkupProcessor(FileProcessor):
    """Processor for markup files (HTML, Markdown, etc.)."""

    def supports(self, category: FileCategory) -> bool:
        return category == FileCategory.MARKUP

    async def process(
        self,
        data: bytes,
        filename: str,
        options: OptimizationOptions,
    ) -> ProcessingResult:
        errors = []
        warnings = []
        optimizations = []

        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1")

        ext = Path(filename).suffix.lower()
        extracted_text = text
        extracted_metadata = {}

        try:
            if ext in [".html", ".htm"]:
                extracted_text, extracted_metadata = await self._process_html(text)
            elif ext in [".md", ".markdown"]:
                extracted_metadata = self._process_markdown(text)

        except Exception as e:
            warnings.append(f"Markup processing warning: {str(e)}")

        processed_data = data
        mime_type, _ = mimetypes.guess_type(filename)

        metadata = FileMetadata(
            filename=filename,
            original_size=len(data),
            processed_size=len(processed_data),
            mime_type=mime_type or "text/plain",
            category=FileCategory.MARKUP,
            extension=ext,
            checksum=hashlib.md5(processed_data).hexdigest(),
            optimization_applied=optimizations,
            extracted_content=extracted_text,
            extracted_metadata=extracted_metadata,
        )

        # Generate learning data
        if options.generate_learning_data:
            metadata.learning_data = self._generate_markup_learning_data(extracted_text, filename)

        return ProcessingResult(
            success=True,
            metadata=metadata,
            processed_data=processed_data,
            errors=errors,
            warnings=warnings,
        )

    async def _process_html(self, text: str) -> tuple[str, dict]:
        """Extract text and metadata from HTML."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(text, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style"]):
                element.decompose()

            extracted_text = soup.get_text(separator="\n", strip=True)

            metadata = {
                "title": soup.title.string if soup.title else None,
                "links": len(soup.find_all("a")),
                "images": len(soup.find_all("img")),
                "headings": len(soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])),
            }

            return extracted_text, metadata

        except ImportError:
            # Simple fallback
            text_only = re.sub(r"<[^>]+>", " ", text)
            return text_only, {}

    def _process_markdown(self, text: str) -> dict:
        """Extract metadata from Markdown."""
        headings = re.findall(r"^#+\s+(.+)$", text, re.MULTILINE)
        links = re.findall(r"\[([^\]]+)\]\([^)]+\)", text)
        code_blocks = re.findall(r"```(\w+)?", text)

        return {
            "headings": headings[:20],
            "heading_count": len(headings),
            "links": len(links),
            "code_blocks": len(code_blocks),
            "languages": list(set(cb for cb in code_blocks if cb)),
        }

    def _generate_markup_learning_data(self, text: str, filename: str) -> dict[str, Any]:
        """Generate learning data from markup."""
        # Split by headings for markdown/html
        sections = re.split(r"\n(?=#+\s)", text)

        examples = []
        for section in sections[:30]:
            lines = section.strip().split("\n")
            if len(lines) > 1:
                title = lines[0].lstrip("#").strip()
                content = "\n".join(lines[1:]).strip()
                if title and content:
                    examples.append({
                        "query": title if "?" in title else f"What is {title}?",
                        "answer": content[:1000],
                    })

        return {
            "examples": examples,
            "source": filename,
        }


class UniversalFileProcessor:
    """
    Universal file processor that handles all supported file types.

    Usage:
        processor = UniversalFileProcessor()

        # Process a file
        result = await processor.process_file(file_path)

        # Process uploaded bytes
        result = await processor.process_bytes(data, filename)

        # Batch process
        results = await processor.process_batch(file_paths)
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = Path(storage_path) if storage_path else None

        # Initialize processors
        self.processors: list[FileProcessor] = [
            ImageProcessor(),
            DocumentProcessor(),
            CodeProcessor(),
            DataProcessor(),
            ArchiveProcessor(),
            MarkupProcessor(),
        ]

        logger.info("UniversalFileProcessor initialized")

    def get_category(self, filename: str) -> FileCategory:
        """Determine file category from filename."""
        ext = Path(filename).suffix.lower()
        return EXTENSION_CATEGORIES.get(ext, FileCategory.UNKNOWN)

    def get_processor(self, category: FileCategory) -> Optional[FileProcessor]:
        """Get appropriate processor for category."""
        for processor in self.processors:
            if processor.supports(category):
                return processor
        return None

    async def process_bytes(
        self,
        data: bytes,
        filename: str,
        options: Optional[OptimizationOptions] = None,
    ) -> ProcessingResult:
        """
        Process file bytes.

        Args:
            data: File content as bytes
            filename: Original filename
            options: Processing options

        Returns:
            Processing result
        """
        options = options or OptimizationOptions()
        category = self.get_category(filename)

        if category == FileCategory.UNKNOWN:
            # Try to detect from content
            category = self._detect_from_content(data, filename)

        processor = self.get_processor(category)

        if processor is None:
            # Return basic result for unsupported files
            return ProcessingResult(
                success=True,
                metadata=FileMetadata(
                    filename=filename,
                    original_size=len(data),
                    processed_size=len(data),
                    mime_type=mimetypes.guess_type(filename)[0] or "application/octet-stream",
                    category=category,
                    extension=Path(filename).suffix.lower(),
                    checksum=hashlib.md5(data).hexdigest(),
                ),
                processed_data=data,
                warnings=["No specific processor available for this file type"],
            )

        return await processor.process(data, filename, options)

    async def process_file(
        self,
        file_path: Union[str, Path],
        options: Optional[OptimizationOptions] = None,
    ) -> ProcessingResult:
        """Process a file from disk."""
        file_path = Path(file_path)

        if not file_path.exists():
            return ProcessingResult(
                success=False,
                errors=[f"File not found: {file_path}"],
            )

        data = file_path.read_bytes()
        return await self.process_bytes(data, file_path.name, options)

    async def process_batch(
        self,
        items: list[Union[str, Path, tuple[bytes, str]]],
        options: Optional[OptimizationOptions] = None,
    ) -> list[ProcessingResult]:
        """
        Process multiple files.

        Args:
            items: List of file paths or (data, filename) tuples
            options: Processing options

        Returns:
            List of processing results
        """
        tasks = []

        for item in items:
            if isinstance(item, tuple):
                data, filename = item
                tasks.append(self.process_bytes(data, filename, options))
            else:
                tasks.append(self.process_file(item, options))

        return await asyncio.gather(*tasks)

    def _detect_from_content(self, data: bytes, filename: str) -> FileCategory:
        """Try to detect file type from content."""
        # Check magic bytes
        if data[:4] == b"\x89PNG":
            return FileCategory.IMAGE
        if data[:2] == b"\xff\xd8":
            return FileCategory.IMAGE
        if data[:4] == b"GIF8":
            return FileCategory.IMAGE
        if data[:4] == b"%PDF":
            return FileCategory.DOCUMENT
        if data[:2] == b"PK":
            return FileCategory.ARCHIVE

        # Check if it's text
        try:
            text = data[:1000].decode("utf-8")
            if text.strip().startswith(("{", "[")):
                return FileCategory.DATA
            if text.strip().startswith("<?xml") or text.strip().startswith("<"):
                return FileCategory.MARKUP if "<html" in text.lower() else FileCategory.DATA
            return FileCategory.DOCUMENT
        except UnicodeDecodeError:
            pass

        return FileCategory.UNKNOWN

    def get_supported_extensions(self) -> dict[str, str]:
        """Get all supported file extensions."""
        return {ext: cat.value for ext, cat in EXTENSION_CATEGORIES.items()}

    async def save_processed(
        self,
        result: ProcessingResult,
        output_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """Save processed file to disk."""
        if not result.success or not result.processed_data:
            return None

        output_dir = output_dir or self.storage_path
        if not output_dir:
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / result.metadata.filename

        output_path.write_bytes(result.processed_data)
        return output_path
