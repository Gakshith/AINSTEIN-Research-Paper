from pathlib import Path
import re
from collections import Counter
from difflib import SequenceMatcher

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from src.AINSTEIN.entity.config_entity import EvaluationConfig


class JudgeResultModel(BaseModel):
    feasible_and_complete: bool = Field(
        description="True if the generated solution is feasible and sufficiently complete for the problem."
    )
    rediscovery: bool = Field(
        description="True if the generated solution substantially matches the original human solution."
    )
    novel_and_valid: bool = Field(
        description="True if the generated solution is valid but meaningfully different from the original human solution."
    )
    justification: str = Field(
        description="Short explanation of the evaluation decision."
    )


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _read_text(self, path: Path) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            return ""

    def _read_status(self, path: Path) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("Status:"):
                        return line.split(":", 1)[1].strip().lower() == "true"
        except FileNotFoundError:
            return False
        return False

    def _run_judge(
        self,
        model_name: str,
        problem_statement: str,
        generated_solution: str,
        reference_solution: str,
    ) -> JudgeResultModel:
        parser = PydanticOutputParser(pydantic_object=JudgeResultModel)

        base_llm = HuggingFaceEndpoint(
            repo_id=model_name,
            task="text-generation",
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_tokens,
        )
        llm = ChatHuggingFace(llm=base_llm)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert evaluator of research ideas.
                    Compare an AI-generated solution to the original human-authored paper solution.
                    Return ONLY valid JSON.
                    Do not add markdown fences or extra commentary.
                    Apply these definitions exactly:
                    - feasible_and_complete = true if the generated solution is technically feasible and sufficiently complete to count as a successful solution.
                    - rediscovery = true if the generated solution substantially matches the original human solution.
                    - novel_and_valid = true if the generated solution is technically valid for the problem but meaningfully different from the original human solution.
                    - rediscovery and novel_and_valid must not both be true.
                    {format_instructions}
                    """,
                ),
                (
                    "human",
                    """Problem Statement:
                    {problem_statement}

                    Generated Solution:
                    {generated_solution}

                    Original Human Solution:
                    {reference_solution}
                    """,
                ),
            ]
        ).partial(format_instructions=parser.get_format_instructions())

        chain = prompt | llm
        raw_response = chain.invoke(
            {
                "problem_statement": problem_statement,
                "generated_solution": generated_solution,
                "reference_solution": reference_solution,
            }
        )
        text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
        return parser.parse(text)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def _jaccard_similarity(self, left: str, right: str) -> float:
        left_tokens = set(self._tokenize(left))
        right_tokens = set(self._tokenize(right))
        if not left_tokens and not right_tokens:
            return 1.0
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    def _token_f1(self, left: str, right: str) -> tuple[float, float, float]:
        left_counts = Counter(self._tokenize(left))
        right_counts = Counter(self._tokenize(right))
        overlap = sum((left_counts & right_counts).values())
        left_total = sum(left_counts.values())
        right_total = sum(right_counts.values())
        precision = overlap / left_total if left_total else 0.0
        recall = overlap / right_total if right_total else 0.0
        if precision + recall == 0:
            return precision, recall, 0.0
        return precision, recall, 2 * precision * recall / (precision + recall)

    def _keyword_overlap(self, left: str, right: str) -> float:
        stopwords = {
            "the", "and", "for", "with", "that", "this", "from", "into",
            "using", "their", "they", "been", "have", "will", "would",
            "could", "should", "about", "approach", "method", "model",
        }
        left_keywords = {token for token in self._tokenize(left) if len(token) > 4 and token not in stopwords}
        right_keywords = {token for token in self._tokenize(right) if len(token) > 4 and token not in stopwords}
        if not left_keywords and not right_keywords:
            return 1.0
        if not left_keywords or not right_keywords:
            return 0.0
        return len(left_keywords & right_keywords) / len(right_keywords)

    def _length_ratio(self, left: str, right: str) -> float:
        left_len = max(len(self._tokenize(left)), 1)
        right_len = max(len(self._tokenize(right)), 1)
        return min(left_len, right_len) / max(left_len, right_len)

    def _sequence_similarity(self, left: str, right: str) -> float:
        return SequenceMatcher(None, left, right).ratio()

    def _write_output(self, output_path: Path, result: dict):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(
                f"Success Rate Relaxed: {result['success_rate_relaxed']}\n"
                f"Success Rate Strict: {result['success_rate_strict']}\n"
                f"Rediscovery Relaxed: {result['rediscovery_relaxed']}\n"
                f"Rediscovery Strict: {result['rediscovery_strict']}\n"
                f"Novel & Valid Relaxed: {result['novel_and_valid_relaxed']}\n"
                f"Novel & Valid Strict: {result['novel_and_valid_strict']}\n"
                f"Judge Agreement: {result['judge_agreement']}\n"
                f"Judge 1 Feasible & Complete: {result['judge_1_feasible_and_complete']}\n"
                f"Judge 1 Rediscovery: {result['judge_1_rediscovery']}\n"
                f"Judge 1 Novel & Valid: {result['judge_1_novel_and_valid']}\n"
                f"Judge 1 Justification: {result['judge_1_justification']}\n"
                f"Judge 2 Feasible & Complete: {result['judge_2_feasible_and_complete']}\n"
                f"Judge 2 Rediscovery: {result['judge_2_rediscovery']}\n"
                f"Judge 2 Novel & Valid: {result['judge_2_novel_and_valid']}\n"
                f"Judge 2 Justification: {result['judge_2_justification']}\n"
                f"Token Jaccard: {result['token_jaccard']:.4f}\n"
                f"Token Precision: {result['token_precision']:.4f}\n"
                f"Token Recall: {result['token_recall']:.4f}\n"
                f"Token F1: {result['token_f1']:.4f}\n"
                f"Keyword Overlap: {result['keyword_overlap']:.4f}\n"
                f"Length Ratio: {result['length_ratio']:.4f}\n"
                f"Sequence Similarity: {result['sequence_similarity']:.4f}\n"
                f"Justification: {result['justification']}\n"
            )

    def evaluate_solution(self, critique_override: bool | None = None) -> dict:
        problem_statement = self._read_text(self.config.problem_statement_path)
        generated_solution = self._read_text(self.config.generated_solution_path)
        reference_solution = self._read_text(self.config.reference_solution_path)

        output_path = Path(self.config.output_file)

        empty_result = {
            "success_rate_relaxed": False,
            "success_rate_strict": False,
            "rediscovery_relaxed": False,
            "rediscovery_strict": False,
            "novel_and_valid_relaxed": False,
            "novel_and_valid_strict": False,
            "judge_agreement": False,
            "judge_1_feasible_and_complete": False,
            "judge_1_rediscovery": False,
            "judge_1_novel_and_valid": False,
            "judge_1_justification": "",
            "judge_2_feasible_and_complete": False,
            "judge_2_rediscovery": False,
            "judge_2_novel_and_valid": False,
            "judge_2_justification": "",
            "token_jaccard": 0.0,
            "token_precision": 0.0,
            "token_recall": 0.0,
            "token_f1": 0.0,
            "keyword_overlap": 0.0,
            "length_ratio": 0.0,
            "sequence_similarity": 0.0,
            "justification": "",
        }

        if not problem_statement or not generated_solution or not reference_solution:
            result = empty_result | {
                "justification": (
                    "Evaluation could not be completed because one or more required artifacts "
                    "were missing: problem statement, generated solution, or reference solution."
                )
            }
            self._write_output(output_path, result)
            return result

        if critique_override is None:
            internal_pass = self._read_status(self.config.internal_status_path)
            external_pass = self._read_status(self.config.external_status_path)
            critique_success = internal_pass and external_pass
        else:
            critique_success = critique_override

        judge_1 = self._run_judge(
            self.config.evaluation_model,
            problem_statement,
            generated_solution,
            reference_solution,
        )
        judge_2 = self._run_judge(
            self.config.evaluation_model_secondary,
            problem_statement,
            generated_solution,
            reference_solution,
        )

        feasible_votes = [
            judge_1.feasible_and_complete,
            judge_2.feasible_and_complete,
        ]
        rediscovery_votes = [
            judge_1.rediscovery,
            judge_2.rediscovery,
        ]
        novel_votes = [
            judge_1.novel_and_valid,
            judge_2.novel_and_valid,
        ]
        token_precision, token_recall, token_f1 = self._token_f1(generated_solution, reference_solution)

        success_rate_relaxed = critique_success and any(feasible_votes)
        success_rate_strict = critique_success and all(feasible_votes)
        rediscovery_relaxed = success_rate_relaxed and any(rediscovery_votes)
        rediscovery_strict = success_rate_strict and all(rediscovery_votes)
        novel_and_valid_relaxed = success_rate_relaxed and any(novel_votes)
        novel_and_valid_strict = success_rate_strict and all(novel_votes)

        judge_agreement = (
            judge_1.feasible_and_complete == judge_2.feasible_and_complete
            and judge_1.rediscovery == judge_2.rediscovery
            and judge_1.novel_and_valid == judge_2.novel_and_valid
        )

        result = {
            "success_rate_relaxed": success_rate_relaxed,
            "success_rate_strict": success_rate_strict,
            "rediscovery_relaxed": rediscovery_relaxed,
            "rediscovery_strict": rediscovery_strict,
            "novel_and_valid_relaxed": novel_and_valid_relaxed,
            "novel_and_valid_strict": novel_and_valid_strict,
            "judge_agreement": judge_agreement,
            "judge_1_feasible_and_complete": judge_1.feasible_and_complete,
            "judge_1_rediscovery": judge_1.rediscovery,
            "judge_1_novel_and_valid": judge_1.novel_and_valid,
            "judge_1_justification": judge_1.justification,
            "judge_2_feasible_and_complete": judge_2.feasible_and_complete,
            "judge_2_rediscovery": judge_2.rediscovery,
            "judge_2_novel_and_valid": judge_2.novel_and_valid,
            "judge_2_justification": judge_2.justification,
            "token_jaccard": self._jaccard_similarity(generated_solution, reference_solution),
            "token_precision": token_precision,
            "token_recall": token_recall,
            "token_f1": token_f1,
            "keyword_overlap": self._keyword_overlap(generated_solution, reference_solution),
            "length_ratio": self._length_ratio(generated_solution, reference_solution),
            "sequence_similarity": self._sequence_similarity(generated_solution, reference_solution),
            "justification": (
                f"Judge 1: {judge_1.justification} | Judge 2: {judge_2.justification}"
            ),
        }

        self._write_output(output_path, result)
        return result
