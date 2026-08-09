import pytest


@pytest.mark.eval
def test_ragas_eval_dataset_shape():
    dataset = [
        {
            "question": "How do I request vacation leave?",
            "answer": "Vacation leave should be requested before the start date.",
            "contexts": ["Employees may apply for vacation, sick, or personal leave."],
            "ground_truth": "Employees can request vacation leave before the start date.",
        }
    ]

    assert dataset[0]["question"]
    assert dataset[0]["contexts"]

