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


@pytest.mark.parametrize("geometry", [
    {"box": [1, 2, 31, 12]},
    {"bbox": [1, 2, 31, 12]},
    {"points": [[1, 2], [31, 2], [31, 12], [1, 12]]},
    {"location": {"left": 1, "top": 2, "width": 30, "height": 10}},
    {"position": [[1, 2], [31, 2], [31, 12], [1, 12]]},
    {"rect": {"x": 1, "y": 2, "w": 30, "h": 10}},
    {"bounding_box": {"x1": 1, "y1": 2, "x2": 31, "y2": 12}},
    {"boundingBox": {"vertices": [{"x": 1, "y": 2}, {"x": 31, "y": 12}]}},
    {"polygon": [1, 2, 31, 2, 31, 12, 1, 12]},
    {"text_box": [1, 2, 31, 12]},
])
def test_external_response_accepts_nested_coordinate_formats(geometry):
    response = parse_external_response({"words_result": [{"words": "hello", **geometry}]}, 100, 80)
    assert response.items[0].box == ((1, 2), (31, 2), (31, 12), (1, 12))
    assert response.coordinate_warning == ""


def test_external_response_prefers_individual_lines_to_parent_summary():
    response = parse_external_response({
        "text": "hello world",
        "data": {"blocks": [{"text": "hello world", "lines": [
            {"text": "hello", "location": {"left": 1, "top": 2, "width": 30, "height": 10}},
            {"text": "world", "location": {"left": 1, "top": 34, "width": 30, "height": 10}},
        ]}]},
    }, 100, 80)
    assert [item.text for item in response.items] == ["hello", "world"]
    assert [item.box[0][1] for item in response.items] == [2, 34]


def test_external_response_accepts_paddle_rows():
    response = parse_external_response({"data": [[
        [[[1, 2], [31, 2], [31, 12], [1, 12]], ["hello", 0.8]],
        [[[1, 34], [31, 34], [31, 44], [1, 44]], ["world", 0.9]],
    ]]}, 100, 80)
    assert [item.text for item in response.items] == ["hello", "world"]
    assert [item.confidence for item in response.items] == [0.8, 0.9]


def test_external_response_accepts_top_level_list():
    response = parse_external_response([{"text": "hello", "box": [1, 2, 31, 12]}], 100, 80)
    assert response.items[0].text == "hello"


def test_external_response_accepts_easyocr_rows():
    response = parse_external_response([
        [[[1, 2], [31, 2], [31, 12], [1, 12]], "hello", 0.8],
    ], 100, 80)
    assert response.items[0].text == "hello"
    assert response.items[0].confidence == 0.8


@pytest.mark.parametrize("geometry", [
    {},
    {"box": [-1, 2, 20, 12]},
    {"box": [1, 2, 101, 12]},
    {"box": [1, 2, 1, 12]},
    {"box": [1, 2, float("nan"), 12]},
    {"location": {"left": 1, "top": 2, "width": float("inf"), "height": 10}},
    {"box": [[1, 2], [31, 2], [31, 12], ["bad", 12]]},
])
def test_external_response_invalid_coordinates_never_fabricates_a_selection_box(geometry):
    with pytest.raises(ExternalOcrError, match="逐条文字坐标"):
        parse_external_response({"items": [{"text": "hello", **geometry}]}, 100, 80)


def test_external_response_keeps_valid_positions_and_warns_about_unlocated_text(caplog):
    response = parse_external_response({"items": [
        {"text": "sensitive valid text", "box": [1, 2, 31, 12], "confidence": 0},
        {"text": "sensitive unlocated text"},
    ]}, 100, 80)
    assert len(response.items) == 1
    assert response.items[0].box == ((1, 2), (31, 2), (31, 12), (1, 12))
    assert response.items[0].confidence == 0
    assert "1 条文字缺少有效坐标" in response.coordinate_warning
    assert "sensitive" not in caplog.text


def test_external_response_empty_results_are_not_coordinate_failures():
    assert parse_external_response({"items": []}, 100, 80).items == []


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
