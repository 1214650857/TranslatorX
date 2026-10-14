from __future__ import annotations

import gc
import logging
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QRect, QThread
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMainWindow

from translatorx import ui, worker as worker_module
from translatorx.config_store import ConfigStore
from translatorx.external_ocr import ExternalOcrError, parse_external_response
from translatorx.models import LayoutMode, OcrItem, TranslatorConfig
from translatorx.overlay import TranslationOverlay
from translatorx.region_selection import RegionSelection


def wait_until(app, condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(5)
    assert condition(), "Timed out waiting for the region worker"


@pytest.fixture
def region_window(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    settings = ConfigStore(tmp_path / "translatorx.json")
    settings.setValue("TRANSLATORX_REGION_OCR_ENGINE", "external")
    settings.setValue("TRANSLATORX_EXTERNAL_OCR_ENDPOINT", "https://example.invalid/ocr")
    window = ui.MainWindow.__new__(ui.MainWindow)
    QMainWindow.__init__(window)
    window._logger = logging.getLogger("translatorx.test.region")
    window._region_busy = False
    window._region_context = None
    window._region_selector = None
    window._ui_settings = settings
    window._statuses = []
    window.titlebar = SimpleNamespace(set_status=window._statuses.append)
    window.credential_dialog = SimpleNamespace(credentials=lambda: {})
    window.region_overlay = TranslationOverlay(always_on_top=True)
    window._config = lambda: TranslatorConfig("baidu", "en", "zh-CN")
    window._layout_mode = lambda: LayoutMode.BELOW
    monkeypatch.setattr(worker_module.ProcessingWorker, "initialize", lambda _self: None)
    monkeypatch.setattr(ui, "grab_region", lambda *_args: np.zeros((80, 120, 3), dtype=np.uint8))
    window._start_worker()
    window.worker_thread.finished.connect(window.worker.deleteLater)
    try:
        yield app, window
    finally:
        window._region_context = None
        window.worker_thread.quit()
        assert window.worker_thread.wait(5000)
        app.processEvents()
        window.region_overlay.close()
        window.region_overlay.deleteLater()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def selection_for(app):
    return RegionSelection(app.primaryScreen().name(), QRect(10, 20, 120, 80))


def test_external_region_result_renders_on_gui_thread_and_survives_collection(region_window, monkeypatch):
    app, window = region_window
    ocr_threads = []
    render_threads = []
    translation_batches = []

    class ExternalClient:
        def recognize(self, image):
            ocr_threads.append(QThread.currentThread())
            return parse_external_response({"items": [{"text": "hello world", "box": [10, 10, 100, 25]}]}, image.shape[1], image.shape[0])

    class Translator:
        def translate_batch(self, texts):
            translation_batches.append(texts)
            return ["你好世界"]

    monkeypatch.setattr(worker_module, "ExternalOcrClient", lambda *_args: ExternalClient())
    monkeypatch.setattr(worker_module, "create_translator", lambda _config: Translator())
    render = window.region_overlay.update_region_content

    def record_render(*args):
        render_threads.append(QThread.currentThread())
        render(*args)

    monkeypatch.setattr(window.region_overlay, "update_region_content", record_render)
    for _iteration in range(8):
        window._on_region_selected(selection_for(app))
        wait_until(app, lambda: not window._region_busy)
        assert window.region_overlay.isVisible()
        gc.collect()
        window.region_overlay.update()
        QTest.qWait(50)

    assert len(ocr_threads) == len(render_threads) == 8
    assert all(thread == window.worker_thread for thread in ocr_threads)
    assert all(thread == app.thread() for thread in render_threads)
    assert translation_batches == [["hello world"]] * 8
    assert window._statuses[-1] == "区域翻译完成"
    assert window.worker._ocr is None


def test_external_region_failure_resets_busy_state_without_local_fallback(region_window, monkeypatch):
    app, window = region_window

    class ExternalClient:
        def recognize(self, _image):
            raise ExternalOcrError("外部 OCR 网络连接失败或超时")

    monkeypatch.setattr(worker_module, "ExternalOcrClient", lambda *_args: ExternalClient())
    window._on_region_selected(selection_for(app))
    wait_until(app, lambda: not window._region_busy)

    assert window._region_context is None
    assert window.region_overlay.isHidden()
    assert "外部 OCR 网络连接失败或超时" in window._statuses[-1]
    assert window.worker._ocr is None


def test_region_hotkey_does_not_start_another_selection_while_busy(region_window, monkeypatch):
    _app, window = region_window
    window._region_busy = True

    def unexpected_selection():
        pytest.fail("A second region selection was started while translation was busy")

    monkeypatch.setattr(ui, "screen_at_cursor", unexpected_selection)
    window.start_region_translation()
    assert window._statuses[-1] == "上一轮区域翻译仍在进行"


@pytest.mark.parametrize("scale", [1.0, 1.25, 1.5])
@pytest.mark.parametrize("layout", [LayoutMode.BELOW, LayoutMode.RIGHT])
@pytest.mark.parametrize("width", [180, 320])
def test_four_menu_rows_keep_their_locations_through_ocr_translation_and_display(region_window, monkeypatch, scale, layout, width):
    app, window = region_window
    source_texts = ["工作的东西", "其他", "研究生成果", "网络"]
    translated_texts = ["Work Items", "Other", "Graduate Achievements", "Network"]
    translation_batches = []
    payload = {"words_result": [
        {"words": text, "location": {"left": 63 * scale, "top": top * scale, "width": 60 * scale, "height": 12 * scale}}
        for text, top in zip(source_texts, [13, 45, 77, 109], strict=True)
    ]}

    class ExternalClient:
        def recognize(self, image):
            return parse_external_response(payload, image.shape[1], image.shape[0])

    class Translator:
        def translate_batch(self, texts):
            translation_batches.append(texts)
            return translated_texts

    monkeypatch.setattr(ui, "grab_region", lambda *_args: np.zeros((int(160 * scale), int(width * scale), 3), dtype=np.uint8))
    monkeypatch.setattr(worker_module, "ExternalOcrClient", lambda *_args: ExternalClient())
    monkeypatch.setattr(worker_module, "create_translator", lambda _config: Translator())
    window._layout_mode = lambda: layout
    window._on_region_selected(RegionSelection(app.primaryScreen().name(), QRect(10, 20, width, 160)))
    wait_until(app, lambda: not window._region_busy)

    rendered = window.region_overlay._items
    assert translation_batches == [source_texts]
    assert [item.text for item in rendered] == translated_texts
    offset = 13 if layout == LayoutMode.BELOW else 0
    assert [item.rect.top() for item in rendered] == pytest.approx([top + offset for top in [13, 45, 77, 109]])
    assert all(not previous.rect.intersects(current.rect) for previous, current in zip(rendered, rendered[1:]))


def test_region_keeps_independent_boxes_and_deduplicates_only_translation_requests(monkeypatch):
    worker = worker_module.ProcessingWorker(enable_openvino=False)
    worker._ocr = object()
    items = [
        OcrItem(((10, top), (110, top), (110, top + 12), (10, top + 12)), text, 1.0)
        for top, text in [(10, "Open a"), (25, "document"), (50, "Open a")]
    ]
    batches = []
    completed = []

    class Translator:
        def translate_batch(self, texts):
            batches.append(texts)
            return ["打开一个", "文档"]

    monkeypatch.setattr(worker, "_recognize", lambda _image: (items, []))
    monkeypatch.setattr(worker_module, "create_translator", lambda _config: Translator())
    worker.region_completed.connect(lambda *args: completed.append(args))
    worker.process_region(np.zeros((80, 120, 3), dtype=np.uint8), TranslatorConfig("baidu", "en", "zh-CN"), {"engine": "local"})

    assert batches == [["Open a", "document"]]
    assert len(completed) == 1
    assert [item.box for item in completed[0][0]] == [item.box for item in items]
    assert [item.translation for item in completed[0][0]] == ["打开一个", "文档", "打开一个"]


def test_missing_coordinates_are_reported_without_translating_or_fabricating_positions(region_window, monkeypatch):
    app, window = region_window

    class ExternalClient:
        def recognize(self, image):
            return parse_external_response({"items": [{"text": "hello world"}]}, image.shape[1], image.shape[0])

    def unexpected_translation(_config):
        pytest.fail("Unlocated text must not be submitted for translation")

    monkeypatch.setattr(worker_module, "ExternalOcrClient", lambda *_args: ExternalClient())
    monkeypatch.setattr(worker_module, "create_translator", unexpected_translation)
    window._on_region_selected(selection_for(app))
    wait_until(app, lambda: not window._region_busy)

    assert window.region_overlay.isHidden()
    assert "逐条文字坐标" in window._statuses[-1]
    assert window.worker._ocr is None


def test_partial_coordinates_keep_valid_translation_and_show_warning_on_gui_thread(region_window, monkeypatch):
    app, window = region_window

    class ExternalClient:
        def recognize(self, image):
            return parse_external_response({"items": [
                {"text": "hello", "box": [10, 10, 80, 22]},
                {"text": "world"},
            ], "balance": 0.5}, image.shape[1], image.shape[0])

    class Translator:
        def translate_batch(self, texts):
            assert texts == ["hello"]
            return ["你好"]

    monkeypatch.setattr(worker_module, "ExternalOcrClient", lambda *_args: ExternalClient())
    monkeypatch.setattr(worker_module, "create_translator", lambda _config: Translator())
    window._on_region_selected(selection_for(app))
    wait_until(app, lambda: not window._region_busy)

    assert window.region_overlay.isVisible()
    assert [item.text for item in window.region_overlay._items] == ["你好"]
    assert "1 条文字缺少有效坐标，未显示" in window._statuses[-1]
    assert window._ui_settings.value("TRANSLATORX_EXTERNAL_OCR_BALANCE", type=float) == 0.5


def test_late_region_result_is_ignored_after_context_is_cleared(region_window, monkeypatch):
    _app, window = region_window
    window._region_busy = True

    def unexpected_render(*_args):
        pytest.fail("A late result updated an abandoned region overlay")

    monkeypatch.setattr(window.region_overlay, "update_region_content", unexpected_render)
    window._on_region_result([], 120, 80, None)
    assert not window._region_busy
    assert window.region_overlay.isHidden()
