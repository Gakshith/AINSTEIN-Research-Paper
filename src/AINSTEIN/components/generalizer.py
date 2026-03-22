import re
from io import BytesIO
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pypdf import PdfReader
import requests

from src.AINSTEIN import logger
from src.AINSTEIN.entity.config_entity import GeneralizerConfig
from src.AINSTEIN.utils.common import create_directories


class GeneralizerModel(BaseModel):
    generalized_research_abstract: str = Field(
        ...,
        description=(
            "A concise 2-3 sentence statement of the core research question "
            "or scientific problem, preserving the challenge exactly while "
            "avoiding method-specific details."
        ),
    )


class ReferenceSolutionModel(BaseModel):
    reference_solution: str = Field(
        ...,
        description=(
            "A concise 3-5 sentence summary of the original paper's proposed "
            "solution, method, or approach."
        ),
    )


class Generalizer:
    def __init__(self, config: GeneralizerConfig):
        self.config = config

    def _load_selected_row(self) -> pd.Series:
        dataset = pd.read_csv(self.config.data_path)

        if self.config.abstract_column not in dataset.columns:
            raise ValueError(
                f"Column '{self.config.abstract_column}' not found in {self.config.data_path}"
            )
        if self.config.pdf_url_column not in dataset.columns:
            raise ValueError(
                f"Column '{self.config.pdf_url_column}' not found in {self.config.data_path}"
            )

        if self.config.paper_id:
            if "paper_id" not in dataset.columns:
                raise ValueError("paper_id column is missing from the ingested dataset")

            paper_rows = dataset[dataset["paper_id"] == self.config.paper_id]
            if paper_rows.empty:
                raise ValueError(f"Paper id '{self.config.paper_id}' not found in dataset")
            selected_row = paper_rows.iloc[0]
        else:
            if not 0 <= self.config.row_index < len(dataset):
                raise IndexError(
                    f"Configured row_index {self.config.row_index} is outside dataset bounds"
                )
            selected_row = dataset.iloc[self.config.row_index]

        logger.info(
            "Using paper from paper_id=%s, title=%s",
            selected_row.get("paper_id", "N/A"),
            selected_row.get("title", "N/A"),
        )
        return selected_row

    def _download_pdf_bytes(self, pdf_url: str) -> bytes:
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()
        return response.content

    def _extract_pdf_text(self, pdf_bytes: bytes, max_pages: int = 3) -> str:
        reader = PdfReader(BytesIO(pdf_bytes))
        extracted_pages = []

        for page in reader.pages[:max_pages]:
            page_text = page.extract_text() or ""
            if page_text.strip():
                extracted_pages.append(page_text)

        pdf_text = "\n".join(extracted_pages).strip()
        if not pdf_text:
            raise ValueError("No readable text found in the PDF")
        return pdf_text

    def _extract_abstract_from_text(self, pdf_text: str) -> str:
        normalized_text = re.sub(r"\r", "\n", pdf_text)
        normalized_text = re.sub(r"\n{2,}", "\n\n", normalized_text)

        patterns = [
            r"(?is)\babstract\b\s*[:.\n]?\s*(.+?)(?=\n\s*(?:1\.?\s+introduction|introduction|keywords|index terms)\b)",
            r"(?is)\babstract\b\s*[:.\n]?\s*(.+?)(?=\n\s*[A-Z][A-Z\s]{3,}\n)",
            r"(?is)\babstract\b\s*[:.\n]?\s*(.+?)(?=\n\n)",
        ]

        for pattern in patterns:
            match = re.search(pattern, normalized_text)
            if match:
                abstract = re.sub(r"\s+", " ", match.group(1)).strip()
                if abstract:
                    return abstract

        raise ValueError("Unable to isolate the abstract section from PDF text")

    def load_abstract(self) -> str:
        selected_row = self._load_selected_row()
        pdf_url = str(selected_row[self.config.pdf_url_column]).strip()
        csv_abstract = str(selected_row[self.config.abstract_column]).strip()

        if not pdf_url:
            if self.config.abstract_fallback_to_csv and csv_abstract:
                logger.warning("PDF URL is empty. Falling back to CSV abstract.")
                return csv_abstract
            raise ValueError("Selected paper does not contain a PDF URL")

        try:
            pdf_bytes = self._download_pdf_bytes(pdf_url)
            pdf_text = self._extract_pdf_text(pdf_bytes)
            self.save_reference_solution(pdf_text)
            abstract = self._extract_abstract_from_text(pdf_text)
            logger.info("Abstract extracted from paper PDF successfully.")
            return abstract
        except Exception as exc:
            if self.config.abstract_fallback_to_csv and csv_abstract:
                logger.warning(
                    "Falling back to CSV abstract because PDF extraction failed: %s",
                    exc,
                )
                return csv_abstract
            raise

    def save_reference_solution(self, pdf_text: str) -> str:
        parser = PydanticOutputParser(pydantic_object=ReferenceSolutionModel)

        base_llm = HuggingFaceEndpoint(
            repo_id=self.config.generalizer_model,
            task="text-generation",
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_tokens,
        )

        llm = ChatHuggingFace(llm=base_llm)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert research analyst.
                    Read paper text and identify the original human-authored solution proposed in the paper.
                    Return ONLY valid JSON.
                    Do not add markdown fences or extra commentary.
                    Focus on the actual method, mechanism, or architecture introduced by the paper.
                    {format_instructions}
                    """,
                ),
                (
                    "human",
                    """Paper Text:
                    {paper_text}

                    Extract the original paper's solution.
                    Requirements:
                    - Write 3-5 precise sentences.
                    - Capture the main technical idea, not the research problem.
                    - Preserve important method details when present.
                    - Do not invent details not supported by the paper text.
                    """,
                ),
            ]
        ).partial(format_instructions=parser.get_format_instructions())

        chain = prompt | llm
        raw_response = chain.invoke({"paper_text": pdf_text[:12000]})
        text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
        parsed = parser.parse(text)

        output_path = Path(self.config.reference_solution_file)
        create_directories([output_path.parent])
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(parsed.reference_solution.strip() + "\n")

        logger.info("Reference solution extracted from paper PDF successfully.")
        return parsed.reference_solution

    def generalization_agent(self, abstract: str) -> GeneralizerModel:
        parser = PydanticOutputParser(pydantic_object=GeneralizerModel)

        base_llm = HuggingFaceEndpoint(
            repo_id=self.config.generalizer_model,
            task="text-generation",
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_tokens,
        )

        llm = ChatHuggingFace(llm=base_llm)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an AI researcher with 20 years of experience.
                    Your task is to read a research abstract and identify the core research question tackled by the paper.
                    Be extremely careful to:
                    - Preserve the fundamental scientific challenge
                    - Avoid hinting at specific solution methods
                    - Maintain precision and clarity
                    - Preserve critical constraints and problem structure
                    Return ONLY valid JSON. Do not add markdown fences or extra commentary.
                    {format_instructions}
                    """,
                ),
                (
                    "human",
                    """Original Research Abstract:
                    {abstract}
                    Write the generalized research abstract that captures the core scientific problem described in the abstract.
                    Requirements:
                    - Semantic Fidelity: Preserve the fundamental scientific challenge exactly.
                    - Information Preservation: Retain all critical details, constraints, and insights.
                    - Specificity: Be precise and unambiguous.
                    - Solution Blindness: Do not hint at or describe the specific solution method.
                    """,
                ),
            ]
        ).partial(format_instructions=parser.get_format_instructions())

        chain = prompt | llm
        raw_response = chain.invoke({"abstract": abstract})
        text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
        parsed = parser.parse(text)

        output_path = Path(self.config.output_file)
        create_directories([output_path.parent])
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(parsed.generalized_research_abstract.strip() + "\n")

        return parsed
