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



import pytest
from types import SimpleNamespace

from app.application.operations import _score_ragas_context
from app.domain.models import RetrievedChunk


@pytest.mark.asyncio
@pytest.mark.eval
async def test_ragas_non_llm_context_metrics_run():
    case = SimpleNamespace(ground_truth="Vacation leave should be requested before the start date.")
    chunks = [
        RetrievedChunk(
            document_id="leave",
            title="Leave",
            chunk_id="leave-1",
            text="Vacation leave should be requested before the start date.",
        )
    ]

    scores = await _score_ragas_context(case, chunks)

    assert scores["ragas_context_precision"] > 0.9
    assert scores["ragas_context_recall"] > 0.9
