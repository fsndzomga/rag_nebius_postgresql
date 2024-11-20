from abc import ABC, abstractmethod
import logging
import PyPDF2
import pytesseract
from PIL import Image
import fitz
import io
import os
from typing import Type, Dict
import pikepdf


# Base parser interface
class BaseParser(ABC):
    @abstractmethod
    def parse(self, filepath: str) -> dict:
        # abstract method to be implemented by subclasses
        pass


# Concrete parser class for text files
class TxtParser(BaseParser):
    def parse(self, filepath: str) -> str:
        try:
            with open(filepath, "r") as file:
                return file.read()
        except Exception as e:
            logging.error(f"Error while parsing file: {e}")
            return None


# Concrete parser class for PDF files
class PdfParser(BaseParser):
    def repair_pdf(self, filepath: str):
        try:
            repaired_filepath = f"{filepath}.repaired"
            with pikepdf.open(filepath) as pdf:
                pdf.save(repaired_filepath)  # Save to a new file
            os.replace(repaired_filepath, filepath)  # Replace the original file
        except pikepdf.PdfError as e:
            logging.error(f"PDF repair failed: {e}")
            raise ValueError("PDF could not be repaired.")

    def validate_pdf(self, filepath: str):
        try:
            with pikepdf.open(filepath) as pdf:  # noqa: F841
                pass  # If this succeeds, the PDF is structurally valid
        except pikepdf.PdfError as e:
            logging.error(f"PDF validation failed: {e}")
            raise ValueError("The PDF is malformed or corrupted.")

    def parse(self, filepath: str) -> str:
        try:
            # Validate and repair the PDF
            self.repair_pdf(filepath)
            self.validate_pdf(filepath)

            content: str = ""
            with open(filepath, "rb") as file:
                reader = PyPDF2.PdfReader(file, strict=False)  # Disable strict mode
                if reader.is_encrypted:
                    try:
                        reader.decrypt("")
                    except Exception as e:
                        logging.error(f"Error decrypting PDF: {e}")
                        return None

                for page_num in range(len(reader.pages)):
                    try:
                        page = reader.pages[page_num]
                        page_content = page.extract_text()
                        if not page_content:
                            page_content = self._ocr_page(filepath, page_num)  # Fallback to OCR
                    except Exception as e:
                        logging.error(f"Error extracting text from page {page_num}: {e}")
                        page_content = self._ocr_page(filepath, page_num)  # Fallback to OCR
                    content += page_content

            return content
        except Exception as e:
            logging.error(f"Error while parsing file: {e}")
            return None

    def _ocr_page(self, filepath: str, page_num: int) -> str:
        try:
            document = fitz.open(filepath)
            page = document.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(img)
            document.close()
            return ocr_text
        except Exception as e:
            logging.error(f"Error during OCR: {e}")
            return ""


# Parser factory class
class ParserFactory:
    _parsers: Dict[str, Type[BaseParser]] = {}

    @classmethod
    def register_parser(cls, extension: str, parser: Type[BaseParser]):
        cls._parsers[extension] = parser

    @classmethod
    def get_parser(cls, extension: str) -> Type[BaseParser]:
        parser = cls._parsers.get(extension)
        if not parser:
            raise ValueError(f"No parser registered for extension:{extension}")
        return parser()


# Register parsers for different file extensions
ParserFactory.register_parser("txt", TxtParser)
ParserFactory.register_parser("pdf", PdfParser)


# FileParser class to parse files based on their extension
class FileParser:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.parser = self._get_parser()

    def _get_parser(self) -> BaseParser:
        extension = self.filepath.split(".")[-1]
        if extension not in ParserFactory._parsers:
            raise ValueError(f"No parser registered for extension:{extension}")
        return ParserFactory.get_parser(extension)

    def parse(self) -> str:
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")
        return self.parser.parse(self.filepath)
