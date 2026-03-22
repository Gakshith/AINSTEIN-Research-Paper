from src.AINSTEIN.components.baselines import BaselineGenerator


def test_baseline_generator_returns_text_for_known_methods():
    generator = BaselineGenerator()
    problem_statement = "Design a robust model for multimodal scientific reasoning under limited data."
    reference_solution = "The paper proposes a retrieval-augmented multimodal transformer with consistency training."
    abstract = "We introduce a multimodal reasoning system with retrieval augmentation."

    for baseline_name in ["problem_restatement", "keyword_template", "abstract_copy"]:
        output = generator.generate(baseline_name, problem_statement, reference_solution, abstract)
        assert isinstance(output, str)
        assert output.strip()


def test_abstract_copy_baseline_returns_original_abstract():
    generator = BaselineGenerator()
    abstract = "This is the original abstract text."

    output = generator.generate("abstract_copy", "problem", "reference", abstract)

    assert output == abstract
