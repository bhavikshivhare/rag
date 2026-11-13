from src.evaluation.text_metrics import is_refusal, normalize_answer, split_sentences, token_f1


def test_token_f1_identical():
    assert token_f1("The refund window is 30 days.", "The refund window is 30 days.") == 1.0


def test_token_f1_partial():
    assert 0 < token_f1("refund within 30 days", "customers can request a refund within 30 days") < 1


def test_token_f1_disjoint():
    assert token_f1("mobile app", "database backups") == 0.0


def test_refusal_detection():
    assert is_refusal("I could not find this information in the provided documents.")
    assert is_refusal("This information is not available in the provided documents.")
    assert not is_refusal("The Growth plan costs $199 per month.")


def test_sentence_split_and_normalisation():
    assert len(split_sentences("One. Two! Three?")) == 3
    assert normalize_answer("The Refund, Policy!") == "refund policy"


def test_generation_metrics_rule_for_unanswerable():
    from src.core.config import EvaluationConfig

    class FakeEmbed:
        def embed_texts_raw(self, texts):
            import numpy as np

            return np.ones((len(texts), 4), dtype="float32")

    from src.evaluation.text_metrics import compute_generation_metrics

    metrics = compute_generation_metrics(
        question="What is the stock price?",
        answer="I could not find this information in the provided documents.",
        contexts=["pricing text"],
        ground_truth="Not available.",
        expected_context=[],
        embed=FakeEmbed(),
        evaluation_cfg=EvaluationConfig(),
    )
    assert metrics.is_refusal is True
    assert metrics.correctness == 1.0
