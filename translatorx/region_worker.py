from __future__ import annotations

import logging
import threading
from dataclasses import replace

from PySide6.QtCore import QObject, Signal, Slot

from .external_ocr import ExternalOcrClient
from .models import OcrItem, TranslatorConfig
from .ocr import OnnxOcrService
from .translators import create_translator
from .worker import group_translation_items, should_translate_text


class RegionTranslationWorker(QObject):
    completed = Signal(object, int, int, object)
    failed = Signal(str)

    def __init__(self, image_bgr, translator_config: TranslatorConfig, ocr_config: dict[str, object]) -> None:
        super().__init__()
        self._image = image_bgr
        self._translator_config = translator_config
        self._ocr_config = ocr_config
        self._logger = logging.getLogger("translatorx.region_worker")

    @Slot()
    def run(self) -> None:
        try:
            engine = str(self._ocr_config.get("engine", "local"))
            if engine == "external":
                response = ExternalOcrClient(
                    str(self._ocr_config.get("endpoint", "")),
                    str(self._ocr_config.get("token", "")),
                    float(self._ocr_config.get("timeout_seconds", 15.0)),
                    int(self._ocr_config.get("max_retries", 3)),
                ).recognize(self._image)
                items = response.items
                balance = response.balance
            else:
                result = []
                errors = []

                def recognize() -> None:
                    try:
                        result.append(OnnxOcrService(enable_openvino=True).recognize(self._image))
                    except BaseException as exc:
                        errors.append(exc)

                thread = threading.Thread(target=recognize, name="translatorx-region-ocr", daemon=True)
                thread.start()
                thread.join()
                if errors:
                    raise errors[0]
                items = result[0] if result else []
                balance = None

            items = [item for item in items if should_translate_text(item.text)]
            items = group_translation_items(items)
            if not items:
                self.completed.emit([], self._image.shape[1], self._image.shape[0], balance)
                return

            texts = []
            for item in items:
                if item.text not in texts:
                    texts.append(item.text)
            translations = create_translator(self._translator_config).translate_batch(texts)
            translated = dict(zip(texts, translations, strict=True))
            result = [replace(item, translation=translated.get(item.text, "")) for item in items]
            self.completed.emit(result, self._image.shape[1], self._image.shape[0], balance)
        except Exception as exc:
            self._logger.exception("区域 OCR 翻译失败")
            self.failed.emit(str(exc))
