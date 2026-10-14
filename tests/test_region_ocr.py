import numpy as np
import pytest
from PySide6.QtCore import QPoint

from translatorx.external_ocr import ExternalOcrClient, ExternalOcrError, parse_external_response
from translatorx.region_selection import normalize_selection


def test_normalize_selection_supports_all_drag_directions():
    rect = normalize_selection(QPoint(200, 150), QPoint(20, 30))
    assert rect is not None
    assert (rect.x(), rect.y(), rect.width(), rect.height()) == (20, 30, 180, 120)


def test_normalize_selection_rejects_tiny_regions():
    assert normalize_selection(QPoint(0, 0), QPoint(7, 8)) is None


def test_external_response_parses_blocks_and_balance():
    response = parse_external_response({
        "data": {"results": [
            {"text": "hello", "box": [1, 2, 20, 12]},
            {"text": "world", "left": 2, "top": 20, "width": 30, "height": 10},
        ]},
        "balance": "0.05",
    }, 100, 80)
    assert [item.text for item in response.items] == ["hello", "world"]
    assert response.balance == 0.05


def test_external_response_invalid_coordinates_falls_back_to_selection():
    response = parse_external_response({"items": [{"text": "hello", "box": [-1, 2, 20, 12]}]}, 100, 80)
    assert len(response.items) == 1
    assert response.items[0].text == "hello"
    assert response.items[0].box[0] == (0.0, 0.0)
    assert response.items[0].box[2] == (100.0, 80.0)


def test_external_client_retries_transient_failures_without_logging_payload(monkeypatch):
    class Response:
        status_code = 503

        def json(self):
            return {}

    class Session:
        def __init__(self):
            self.calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            return Response()

    session = Session()
    waits = []
    client = ExternalOcrClient("http://example.invalid/ocr", "secret-token", max_retries=3, session=session, sleep=waits.append)
    with pytest.raises(ExternalOcrError, match="网络连接失败"):
        client.recognize(np.zeros((8, 8, 3), dtype=np.uint8))
    assert session.calls == 3
    assert waits == [1.0, 2.0]
