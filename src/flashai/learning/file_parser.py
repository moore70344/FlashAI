"""
File Parser Utilities for Learning Data

Supports parsing learning data from various file formats:
- JSON: Single object or array of examples
- JSONL: JSON Lines format (one JSON object per line)
- CSV: Tabular data with headers
- YAML: YAML formatted data
- TXT: Plain text (line-based or document)
- Markdown: Structured markdown with Q&A sections
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


logger = logging.getLogger(__name__)


class FileFormat(Enum):
    """Supported file formats for learning data."""
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    YAML = "yaml"
    TXT = "txt"
    MARKDOWN = "md"
    UNKNOWN = "unknown"


@dataclass
class ParsedExample:
    """A single parsed learning example."""
    query: str
    answer: str
    context: Optional[dict[str, Any]] = None
    negative_samples: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "context": self.context,
            "negative_samples": self.negative_samples,
            "metadata": self.metadata,
        }


@dataclass
class ParseResult:
    """Result of parsing a learning data file."""
    success: bool
    examples: list[ParsedExample]
    format_detected: FileFormat
    total_lines: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "examples_count": len(self.examples),
            "format": self.format_detected.value,
            "total_lines": self.total_lines,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }

    def to_learning_data(self) -> dict[str, Any]:
        """Convert to format expected by the learning engine."""
        return {
            "examples": [ex.to_dict() for ex in self.examples],
            "source_format": self.format_detected.value,
            "metadata": self.metadata,
        }


class FileParser(ABC):
    """Abstract base class for file parsers."""

    @abstractmethod
    def parse(self, content: str | bytes, filename: Optional[str] = None) -> ParseResult:
        """Parse content and return learning examples."""
        pass

    @abstractmethod
    def supports_format(self, format: FileFormat) -> bool:
        """Check if this parser supports the given format."""
        pass


class JSONParser(FileParser):
    """Parser for JSON files."""

    def supports_format(self, format: FileFormat) -> bool:
        return format == FileFormat.JSON

    def parse(self, content: str | bytes, filename: Optional[str] = None) -> ParseResult:
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        examples = []
        errors = []

        try:
            data = json.loads(content)

            # Handle different JSON structures
            if isinstance(data, list):
                # Array of examples
                for i, item in enumerate(data):
                    example = self._parse_item(item, i)
                    if example:
                        examples.append(example)
                    else:
                        errors.append(f"Could not parse item at index {i}")

            elif isinstance(data, dict):
                # Single example or nested structure
                if "examples" in data:
                    # Nested examples array
                    for i, item in enumerate(data["examples"]):
                        example = self._parse_item(item, i)
                        if example:
                            examples.append(example)
                elif "query" in data or "question" in data:
                    # Single example
                    example = self._parse_item(data, 0)
                    if example:
                        examples.append(example)
                else:
                    # Try to interpret as key-value pairs
                    for key, value in data.items():
                        if isinstance(value, str):
                            examples.append(ParsedExample(
                                query=key,
                                answer=value,
                            ))

            return ParseResult(
                success=len(examples) > 0,
                examples=examples,
                format_detected=FileFormat.JSON,
                total_lines=content.count("\n") + 1,
                errors=errors,
            )

        except json.JSONDecodeError as e:
            return ParseResult(
                success=False,
                examples=[],
                format_detected=FileFormat.JSON,
                errors=[f"JSON parse error: {str(e)}"],
            )

    def _parse_item(self, item: dict, index: int) -> Optional[ParsedExample]:
        """Parse a single JSON item into an example."""
        if not isinstance(item, dict):
            return None

        # Try various field names for query
        query = (
            item.get("query") or
            item.get("question") or
            item.get("input") or
            item.get("prompt") or
            item.get("q")
        )

        # Try various field names for answer
        answer = (
            item.get("answer") or
            item.get("response") or
            item.get("output") or
            item.get("completion") or
            item.get("a")
        )

        if not query or not answer:
            return None

        return ParsedExample(
            query=str(query),
            answer=str(answer),
            context=item.get("context"),
            negative_samples=item.get("negative_samples", []),
            metadata=item.get("metadata", {}),
        )


class JSONLParser(FileParser):
    """Parser for JSON Lines files."""

    def supports_format(self, format: FileFormat) -> bool:
        return format == FileFormat.JSONL

    def parse(self, content: str | bytes, filename: Optional[str] = None) -> ParseResult:
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        examples = []
        errors = []
        json_parser = JSONParser()

        lines = content.strip().split("\n")
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                example = json_parser._parse_item(data, i)
                if example:
                    examples.append(example)
                else:
                    errors.append(f"Could not parse line {i + 1}")
            except json.JSONDecodeError as e:
                errors.append(f"JSON error on line {i + 1}: {str(e)}")

        return ParseResult(
            success=len(examples) > 0,
            examples=examples,
            format_detected=FileFormat.JSONL,
            total_lines=len(lines),
            errors=errors,
        )


class CSVParser(FileParser):
    """Parser for CSV files."""

    def supports_format(self, format: FileFormat) -> bool:
        return format == FileFormat.CSV

    def parse(self, content: str | bytes, filename: Optional[str] = None) -> ParseResult:
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        examples = []
        errors = []
        warnings = []

        try:
            # Detect dialect
            sample = content[:4096]
            try:
                dialect = csv.Sniffer().sniff(sample)
            except csv.Error:
                dialect = csv.excel

            reader = csv.DictReader(io.StringIO(content), dialect=dialect)

            # Normalize headers
            if reader.fieldnames:
                header_map = self._create_header_map(reader.fieldnames)
            else:
                return ParseResult(
                    success=False,
                    examples=[],
                    format_detected=FileFormat.CSV,
                    errors=["No headers found in CSV"],
                )

            for i, row in enumerate(reader):
                query_field = header_map.get("query")
                answer_field = header_map.get("answer")

                if query_field and answer_field:
                    query = row.get(query_field, "").strip()
                    answer = row.get(answer_field, "").strip()

                    if query and answer:
                        # Get optional fields
                        context_field = header_map.get("context")
                        context = None
                        if context_field and row.get(context_field):
                            try:
                                context = json.loads(row[context_field])
                            except json.JSONDecodeError:
                                context = {"raw": row[context_field]}

                        examples.append(ParsedExample(
                            query=query,
                            answer=answer,
                            context=context,
                            metadata={"row": i + 1},
                        ))
                    else:
                        warnings.append(f"Empty query or answer on row {i + 2}")
                else:
                    if i == 0:
                        errors.append("Could not find query/answer columns")
                    break

            return ParseResult(
                success=len(examples) > 0,
                examples=examples,
                format_detected=FileFormat.CSV,
                total_lines=i + 2 if 'i' in dir() else 1,
                errors=errors,
                warnings=warnings,
            )

        except csv.Error as e:
            return ParseResult(
                success=False,
                examples=[],
                format_detected=FileFormat.CSV,
                errors=[f"CSV parse error: {str(e)}"],
            )

    def _create_header_map(self, fieldnames: list[str]) -> dict[str, str]:
        """Map standard field names to actual column headers."""
        header_map = {}
        normalized = {name.lower().strip(): name for name in fieldnames}

        # Query field mappings
        for key in ["query", "question", "input", "prompt", "q"]:
            if key in normalized:
                header_map["query"] = normalized[key]
                break

        # Answer field mappings
        for key in ["answer", "response", "output", "completion", "a"]:
            if key in normalized:
                header_map["answer"] = normalized[key]
                break

        # Context field mappings
        for key in ["context", "ctx", "metadata"]:
            if key in normalized:
                header_map["context"] = normalized[key]
                break

        return header_map


class YAMLParser(FileParser):
    """Parser for YAML files."""

    def supports_format(self, format: FileFormat) -> bool:
        return format == FileFormat.YAML

    def parse(self, content: str | bytes, filename: Optional[str] = None) -> ParseResult:
        if not YAML_AVAILABLE:
            return ParseResult(
                success=False,
                examples=[],
                format_detected=FileFormat.YAML,
                errors=["PyYAML not installed. Install with: pip install pyyaml"],
            )

        if isinstance(content, bytes):
            content = content.decode("utf-8")

        examples = []
        errors = []
        json_parser = JSONParser()

        try:
            data = yaml.safe_load(content)

            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        example = json_parser._parse_item(item, i)
                        if example:
                            examples.append(example)
                        else:
                            errors.append(f"Could not parse item at index {i}")

            elif isinstance(data, dict):
                if "examples" in data:
                    for i, item in enumerate(data["examples"]):
                        example = json_parser._parse_item(item, i)
                        if example:
                            examples.append(example)
                elif "query" in data or "question" in data:
                    example = json_parser._parse_item(data, 0)
                    if example:
                        examples.append(example)

            return ParseResult(
                success=len(examples) > 0,
                examples=examples,
                format_detected=FileFormat.YAML,
                total_lines=content.count("\n") + 1,
                errors=errors,
            )

        except yaml.YAMLError as e:
            return ParseResult(
                success=False,
                examples=[],
                format_detected=FileFormat.YAML,
                errors=[f"YAML parse error: {str(e)}"],
            )


class TextParser(FileParser):
    """Parser for plain text files."""

    def supports_format(self, format: FileFormat) -> bool:
        return format == FileFormat.TXT

    def parse(self, content: str | bytes, filename: Optional[str] = None) -> ParseResult:
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        examples = []
        warnings = []

        # Try to detect Q&A patterns
        lines = content.strip().split("\n")

        # Pattern 1: Q: ... A: ... format
        qa_pattern = re.compile(r"^[QqQuestion:]+\s*[:.]?\s*(.+)$")
        answer_pattern = re.compile(r"^[AaAnswer:]+\s*[:.]?\s*(.+)$")

        current_query = None
        current_answer_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            q_match = qa_pattern.match(line)
            a_match = answer_pattern.match(line)

            if q_match:
                # Save previous Q&A pair
                if current_query and current_answer_lines:
                    examples.append(ParsedExample(
                        query=current_query,
                        answer="\n".join(current_answer_lines),
                    ))
                current_query = q_match.group(1).strip()
                current_answer_lines = []

            elif a_match:
                current_answer_lines.append(a_match.group(1).strip())

            elif current_query:
                # Continuation of answer
                current_answer_lines.append(line)

        # Save last pair
        if current_query and current_answer_lines:
            examples.append(ParsedExample(
                query=current_query,
                answer="\n".join(current_answer_lines),
            ))

        # Pattern 2: If no Q&A found, try line pairs
        if not examples:
            warnings.append("No Q&A pattern detected, treating as line pairs")
            for i in range(0, len(lines) - 1, 2):
                query = lines[i].strip()
                answer = lines[i + 1].strip() if i + 1 < len(lines) else ""
                if query and answer:
                    examples.append(ParsedExample(query=query, answer=answer))

        # Pattern 3: Treat entire content as single document
        if not examples:
            warnings.append("Treating as single document for learning")
            examples.append(ParsedExample(
                query="document_content",
                answer=content[:10000],  # Limit size
                metadata={"type": "document", "full_length": len(content)},
            ))

        return ParseResult(
            success=len(examples) > 0,
            examples=examples,
            format_detected=FileFormat.TXT,
            total_lines=len(lines),
            warnings=warnings,
        )


class MarkdownParser(FileParser):
    """Parser for Markdown files with structured content."""

    def supports_format(self, format: FileFormat) -> bool:
        return format == FileFormat.MARKDOWN

    def parse(self, content: str | bytes, filename: Optional[str] = None) -> ParseResult:
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        examples = []
        warnings = []

        # Extract sections with headers
        sections = self._extract_sections(content)

        for section in sections:
            title = section.get("title", "")
            body = section.get("body", "")

            if title and body:
                # Check for Q&A subsections
                qa_pairs = self._extract_qa_from_section(body)
                if qa_pairs:
                    examples.extend(qa_pairs)
                else:
                    # Treat header as query, content as answer
                    examples.append(ParsedExample(
                        query=title,
                        answer=body.strip()[:5000],
                        metadata={"level": section.get("level", 1)},
                    ))

        if not examples:
            # Fallback to text parser
            text_parser = TextParser()
            return text_parser.parse(content, filename)

        return ParseResult(
            success=len(examples) > 0,
            examples=examples,
            format_detected=FileFormat.MARKDOWN,
            total_lines=content.count("\n") + 1,
            warnings=warnings,
        )

    def _extract_sections(self, content: str) -> list[dict[str, Any]]:
        """Extract markdown sections by headers."""
        sections = []
        current_section = None
        current_body_lines = []

        header_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

        for line in content.split("\n"):
            match = header_pattern.match(line)
            if match:
                # Save previous section
                if current_section:
                    current_section["body"] = "\n".join(current_body_lines)
                    sections.append(current_section)

                level = len(match.group(1))
                title = match.group(2).strip()
                current_section = {"title": title, "level": level}
                current_body_lines = []
            elif current_section:
                current_body_lines.append(line)

        # Save last section
        if current_section:
            current_section["body"] = "\n".join(current_body_lines)
            sections.append(current_section)

        return sections

    def _extract_qa_from_section(self, body: str) -> list[ParsedExample]:
        """Extract Q&A pairs from section body."""
        examples = []

        # Look for Q/A bullet points
        qa_pattern = re.compile(
            r"\*\*(?:Q|Question)\s*[:.]?\*\*\s*(.+?)\s*\*\*(?:A|Answer)\s*[:.]?\*\*\s*(.+?)(?=\*\*Q|\*\*Question|$)",
            re.IGNORECASE | re.DOTALL
        )

        for match in qa_pattern.finditer(body):
            query = match.group(1).strip()
            answer = match.group(2).strip()
            if query and answer:
                examples.append(ParsedExample(query=query, answer=answer))

        return examples


class LearningDataParser:
    """
    Main parser that auto-detects format and parses learning data.

    Usage:
        parser = LearningDataParser()

        # Parse from file
        result = parser.parse_file("/path/to/data.json")

        # Parse from content
        result = parser.parse_content(content, filename="data.csv")

        # Parse from file upload
        result = parser.parse_upload(file_bytes, filename="training.jsonl")
    """

    def __init__(self):
        self.parsers: list[FileParser] = [
            JSONParser(),
            JSONLParser(),
            CSVParser(),
            YAMLParser(),
            MarkdownParser(),
            TextParser(),
        ]

    def detect_format(self, filename: Optional[str] = None, content: Optional[str] = None) -> FileFormat:
        """Detect file format from filename or content."""
        if filename:
            ext = Path(filename).suffix.lower().lstrip(".")

            format_map = {
                "json": FileFormat.JSON,
                "jsonl": FileFormat.JSONL,
                "ndjson": FileFormat.JSONL,
                "csv": FileFormat.CSV,
                "tsv": FileFormat.CSV,
                "yaml": FileFormat.YAML,
                "yml": FileFormat.YAML,
                "txt": FileFormat.TXT,
                "text": FileFormat.TXT,
                "md": FileFormat.MARKDOWN,
                "markdown": FileFormat.MARKDOWN,
            }

            if ext in format_map:
                return format_map[ext]

        # Try to detect from content
        if content:
            content_start = content.strip()[:100]

            # JSON detection
            if content_start.startswith(("{", "[")):
                return FileFormat.JSON

            # JSONL detection
            if content_start.startswith("{") and "\n{" in content[:1000]:
                return FileFormat.JSONL

            # YAML detection
            if content_start.startswith("---") or ": " in content_start:
                return FileFormat.YAML

            # Markdown detection
            if content_start.startswith("#"):
                return FileFormat.MARKDOWN

            # CSV detection (has comma-separated header)
            first_line = content_start.split("\n")[0]
            if "," in first_line and first_line.count(",") >= 1:
                return FileFormat.CSV

        return FileFormat.TXT

    def get_parser(self, format: FileFormat) -> Optional[FileParser]:
        """Get parser for a specific format."""
        for parser in self.parsers:
            if parser.supports_format(format):
                return parser
        return None

    def parse_file(self, file_path: Union[str, Path]) -> ParseResult:
        """Parse learning data from a file."""
        file_path = Path(file_path)

        if not file_path.exists():
            return ParseResult(
                success=False,
                examples=[],
                format_detected=FileFormat.UNKNOWN,
                errors=[f"File not found: {file_path}"],
            )

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = file_path.read_bytes().decode("latin-1")
            except Exception as e:
                return ParseResult(
                    success=False,
                    examples=[],
                    format_detected=FileFormat.UNKNOWN,
                    errors=[f"Could not read file: {str(e)}"],
                )

        return self.parse_content(content, filename=file_path.name)

    def parse_content(
        self,
        content: Union[str, bytes],
        filename: Optional[str] = None,
        format_hint: Optional[FileFormat] = None,
    ) -> ParseResult:
        """Parse learning data from content string or bytes."""
        if isinstance(content, bytes):
            try:
                content = content.decode("utf-8")
            except UnicodeDecodeError:
                content = content.decode("latin-1")

        # Detect format
        if format_hint:
            format = format_hint
        else:
            format = self.detect_format(filename=filename, content=content)

        # Get parser
        parser = self.get_parser(format)
        if not parser:
            return ParseResult(
                success=False,
                examples=[],
                format_detected=format,
                errors=[f"No parser available for format: {format.value}"],
            )

        # Parse content
        result = parser.parse(content, filename)

        logger.info(
            f"Parsed {len(result.examples)} examples from {filename or 'content'} "
            f"(format: {result.format_detected.value})"
        )

        return result

    def parse_upload(
        self,
        file_content: bytes,
        filename: str,
        content_type: Optional[str] = None,
    ) -> ParseResult:
        """Parse learning data from an uploaded file."""
        # Use content type hint if available
        format_hint = None
        if content_type:
            content_type_map = {
                "application/json": FileFormat.JSON,
                "text/csv": FileFormat.CSV,
                "text/plain": FileFormat.TXT,
                "text/markdown": FileFormat.MARKDOWN,
                "application/x-yaml": FileFormat.YAML,
            }
            format_hint = content_type_map.get(content_type)

        return self.parse_content(
            content=file_content,
            filename=filename,
            format_hint=format_hint,
        )

    def get_supported_formats(self) -> list[dict[str, Any]]:
        """Get list of supported formats with descriptions."""
        return [
            {
                "format": "json",
                "extensions": [".json"],
                "description": "JSON array of examples or object with 'examples' key",
                "example": '[{"query": "What is X?", "answer": "X is..."}]',
            },
            {
                "format": "jsonl",
                "extensions": [".jsonl", ".ndjson"],
                "description": "JSON Lines - one JSON object per line",
                "example": '{"query": "Q1", "answer": "A1"}\n{"query": "Q2", "answer": "A2"}',
            },
            {
                "format": "csv",
                "extensions": [".csv", ".tsv"],
                "description": "CSV with query/question and answer/response columns",
                "example": "query,answer\nWhat is X?,X is...",
            },
            {
                "format": "yaml",
                "extensions": [".yaml", ".yml"],
                "description": "YAML format with examples list",
                "example": "examples:\n  - query: What is X?\n    answer: X is...",
            },
            {
                "format": "markdown",
                "extensions": [".md", ".markdown"],
                "description": "Markdown with headers as queries and content as answers",
                "example": "# What is X?\n\nX is a concept that...",
            },
            {
                "format": "txt",
                "extensions": [".txt", ".text"],
                "description": "Plain text with Q:/A: patterns or line pairs",
                "example": "Q: What is X?\nA: X is...",
            },
        ]
