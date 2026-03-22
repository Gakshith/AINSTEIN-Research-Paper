import re
from collections import Counter


class BaselineGenerator:
    def _keywords(self, text: str, top_k: int = 6) -> list[str]:
        tokens = re.findall(r"[a-zA-Z]{4,}", text.lower())
        stopwords = {
            "that", "with", "from", "this", "their", "which", "using", "into",
            "problem", "research", "paper", "method", "approach", "model",
            "solution", "would", "could", "should", "have", "been", "they",
        }
        filtered = [token for token in tokens if token not in stopwords]
        return [word for word, _ in Counter(filtered).most_common(top_k)]

    def generate(self, baseline_name: str, problem_statement: str, reference_solution: str, abstract: str) -> str:
        if baseline_name == "problem_restatement":
            return (
                "We address the problem by directly optimizing for the stated objective and "
                "iteratively refining the system against the task constraints. "
                f"The approach focuses on solving: {problem_statement.strip()}"
            )

        if baseline_name == "keyword_template":
            keywords = ", ".join(self._keywords(problem_statement))
            return (
                "We propose a modular pipeline that combines representation learning, "
                "adaptive optimization, and task-specific validation. "
                f"The design prioritizes the following concepts: {keywords}. "
                "The system is evaluated by checking whether it remains feasible under the paper constraints."
            )

        if baseline_name == "abstract_copy":
            return abstract.strip()

        if baseline_name == "reference_oracle":
            return reference_solution.strip()

        raise ValueError(f"Unknown baseline: {baseline_name}")
