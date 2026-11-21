from openinterpretability.core.schemas import AuditRecord


def test_model_validate_reads_a_whole_record_dict(record):
    data = record.model_dump()
    assert AuditRecord.model_validate(data) == record


def test_a_record_without_a_verdict_is_not_self_reviewed(record):
    assert not record.model_copy(update={"verdict": None}).self_reviewed
