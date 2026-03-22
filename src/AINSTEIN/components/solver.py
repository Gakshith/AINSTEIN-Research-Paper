from pathlib import Path

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from src.AINSTEIN.entity.config_entity import SolverConfig


class SolutionModel(BaseModel):
    research_solution: str = Field(
        ...,
        description="A concise solution for the core research question."
    )


class Solver:
    def __init__(self, config: SolverConfig):
        self.config = config

    def get_solution(self) -> SolutionModel:
        parser = PydanticOutputParser(pydantic_object=SolutionModel)

        base_llm = HuggingFaceEndpoint(
            repo_id=self.config.solver_model,
            task="text-generation",
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_tokens,
        )

        llm = ChatHuggingFace(llm=base_llm)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert AI research scientist.
                    Your task is to invent a plausible technical approach that could solve a given scientific problem in machine learning.
                    Return ONLY valid JSON.
                    Do not add headings, markdown, bullet points, or extra commentary.
                    The output must have exactly this key:
                    "research_solution"
                    {format_instructions}
                    """,
                ),
                (
                    "human",
                    """Problem Statement:
                    {problem_statement}
                    Your Task:
                    Propose a specific and novel technical approach, mechanism, or architecture.
                    Explain your proposed method in 3-5 sentences as if you are writing the core idea in a research paper.

                    Requirements:
                    - Novelty & Creativity: Propose a non-obvious, innovative solution.
                    - Technical Feasibility: Ensure your approach is logically sound and implementable.
                    - Completeness: Provide enough detail to understand the core methodology.
                    """,
                ),
            ]
        ).partial(format_instructions=parser.get_format_instructions())

        chain = prompt | llm

        with open(self.config.data_path, "r", encoding="utf-8") as f:
            problem_statement = f.read()

        raw_response = chain.invoke({"problem_statement": problem_statement})
        text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)

        parsed = parser.parse(text)

        output_path = Path(self.config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(parsed.research_solution.strip() + "\n")

        return parsed
