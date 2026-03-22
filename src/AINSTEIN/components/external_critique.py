from pathlib import Path

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from src.AINSTEIN.entity.config_entity import ExternalCritiqueConfig


class ExternalCritiqueModel(BaseModel):
    novelty: int = Field(description="Score from 1 to 10")
    technical_feasibility: int = Field(description="Score from 1 to 10")
    completeness: int = Field(description="Score from 1 to 10")
    final_judgement: str = Field(description="Whether the solution solves the problem")
    justification: str = Field(description="Explanation of all scores and judgement")


class ExternalCritique:
    def __init__(self, config: ExternalCritiqueConfig):
        self.config = config

    def get_external_critique_score(self) -> float:
        parser = PydanticOutputParser(pydantic_object=ExternalCritiqueModel)

        base_llm = HuggingFaceEndpoint(
            repo_id=self.config.external_critique_model,
            task="text-generation",
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_tokens,
        )

        llm = ChatHuggingFace(llm=base_llm)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert evaluator.
                    Return ONLY valid JSON.
                    Do not add markdown, headings, bullet points, or extra commentary.
                    Do not change key names. Do not add extra keys.
                    Required format:
                    {{
                     "novelty": 0,
                     "technical_feasibility": 0,
                     "completeness": 0,
                     "final_judgement": "",
                     "justification": ""
                    }}
                    {format_instructions}
                    """,
                ),
                (
                    "human",
                    """Problem Statement:
                    {problem_statement}
                    Proposed Solution:
                    {solution}
                    Evaluate based on:
                    - Novelty (1-10)
                    - Technical Feasibility (1-10)
                    - Completeness (1-10)
                    - Final Judgement
                    - Justification
                    """,
                ),
            ]
        ).partial(format_instructions=parser.get_format_instructions())

        chain = prompt | llm
        with open(self.config.abstract_path, "r", encoding="utf-8") as f:
            problem_statement = f.read()
        with open(self.config.solution_path, "r", encoding="utf-8") as f:
            solution = f.read()

        raw_response = chain.invoke(
            {
                "problem_statement": problem_statement,
                "solution": solution,
            }
        )
        text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
        parsed = parser.parse(text)

        average_external_critique_score = (
            parsed.novelty + parsed.technical_feasibility + parsed.completeness
        ) / 3
        status = average_external_critique_score >= self.config.external_critique_threshold

        output_path = Path(self.config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(
                f"Status: {status}\n"
                f"Average Score: {average_external_critique_score:.2f}\n"
                f"Novelty: {parsed.novelty}\n"
                f"Technical Feasibility: {parsed.technical_feasibility}\n"
                f"Completeness: {parsed.completeness}\n"
                f"Final Judgement: {parsed.final_judgement}\n"
                f"Justification: {parsed.justification}\n"
            )

        return average_external_critique_score
