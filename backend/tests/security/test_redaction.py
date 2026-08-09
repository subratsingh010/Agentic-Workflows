from app.core.logging import redact


def test_redacts_bearer_and_email():
    value = redact("Bearer abc.def.ghi sent by person@example.com")

    assert "abc.def.ghi" not in value
    assert "person@example.com" not in value

