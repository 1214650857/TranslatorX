from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np
import requests

from .models import OcrItem


class ExternalOcrError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ExternalOcrResponse:
    items: list[OcrItem]
    balance: float | None = None
    member_status: str = ""


def encode_png_base64(image_bgr: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ExternalOcrError("截图 PNG 编码失败")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _first(mapping: dict[str, Any], keys: tuple[str, ...], default=None):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _walk_candidates(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        result: list[dict[str, Any]] = []
        for value in payload:
            result.extend(_walk_candidates(value))
        return result
    if not isinstance(payload, dict):
        return []
    text = _first(payload, ("text", "words", "content", "value"))
    if isinstance(text, str) and text.strip():
        return [payload]
    result = []
    for key in ("data", "result", "results", "items", "ocr", "ocr_result", "texts", "text_list"):
        if key in payload:
            result.extend(_walk_candidates(payload[key]))
    return result


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _friendly_error(message: str) -> str:
    lowered = message.casefold()
    if any(word in lowered for word in ("token", "auth", "unauthor", "invalid key", "密钥")):
        return "外部 OCR Token 无效"
    if any(word in lowered for word in ("balance", "余额", "quota", "insufficient")):
        return "外部 OCR 余额不足"
    return "外部 OCR 服务返回错误"


def parse_external_response(payload: Any, width: int, height: int) -> ExternalOcrResponse:
    if not isinstance(payload, dict):
        raise ExternalOcrError("外部 OCR 服务返回格式异常")
    error = _first(payload, ("error", "error_msg", "message"))
    code = _first(payload, ("code", "status", "errno"))
    if not _walk_candidates(payload) and (error or str(code).casefold() in {"401", "403", "402", "unauthorized", "error", "fail", "false"}):
        message = str(error or code)
        if str(code) == "402":
            raise ExternalOcrError("外部 OCR 余额不足")
        raise ExternalOcrError(_friendly_error(message))
    raw_items = _walk_candidates(payload)
    items: list[OcrItem] = []
    invalid_reason = ""
    for raw in raw_items:
        text = str(_first(raw, ("text", "words", "content", "value"), "")).strip()
        if not text:
            continue
        box = raw.get("box") or raw.get("bbox") or raw.get("points")
        coords = None
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            if isinstance(box[0], (list, tuple)):
                xs = [_number(point[0]) for point in box if isinstance(point, (list, tuple)) and len(point) >= 2]
                ys = [_number(point[1]) for point in box if isinstance(point, (list, tuple)) and len(point) >= 2]
                if xs and ys and all(value is not None for value in (*xs, *ys)):
                    coords = (min(xs), min(ys), max(xs), max(ys))
            else:
                values = [_number(value) for value in box[:4]]
                if all(value is not None for value in values):
                    coords = tuple(values)
        if coords is None:
            left = _number(_first(raw, ("left", "x1", "x")))
            top = _number(_first(raw, ("top", "y1", "y")))
            right = _number(_first(raw, ("right", "x2")))
            bottom = _number(_first(raw, ("bottom", "y2")))
            width_value = _number(_first(raw, ("width", "w")))
            height_value = _number(_first(raw, ("height", "h")))
            if right is None and left is not None and width_value is not None:
                right = left + width_value
            if bottom is None and top is not None and height_value is not None:
                bottom = top + height_value
            if None not in (left, top, right, bottom):
                coords = (left, top, right, bottom)
        if coords is None:
            invalid_reason = "缺少坐标"
            continue
        left, top, right, bottom = coords
        if left < 0 or top < 0 or right <= left or bottom <= top or right > width or bottom > height:
            invalid_reason = f"坐标超出截图范围：({left},{top},{right},{bottom})"
            continue
        confidence = _number(_first(raw, ("confidence", "score"), 1.0)) or 1.0
        items.append(OcrItem(
            box=((left, top), (right, top), (right, bottom), (left, bottom)),
            text=text,
            confidence=confidence,
        ))

    if invalid_reason:
        logging.getLogger("translatorx.external_ocr").warning("外部 OCR 坐标异常：%s", invalid_reason)
        valid_texts = [str(_first(raw, ("text", "words", "content", "value"), "")).strip() for raw in raw_items]
        combined = " ".join(text for text in valid_texts if text)
        items = [OcrItem(
            box=((0.0, 0.0), (float(width), 0.0), (float(width), float(height)), (0.0, float(height))),
            text=combined,
            confidence=1.0,
        )] if combined else []
    balance = _number(_first(payload, ("balance", "remaining_balance", "money", "amount")))
    member_status = str(_first(payload, ("member_status", "membership", "vip"), "") or "")
    return ExternalOcrResponse(items=items, balance=balance, member_status=member_status)


class ExternalOcrClient:
    def __init__(
        self,
        endpoint: str,
        token: str,
        timeout_seconds: float = 15.0,
        max_retries: int = 3,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.endpoint = endpoint.strip()
        self.token = token
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.max_retries = max(1, min(3, int(max_retries)))
        self.session = session or requests.Session()
        self.sleep = sleep
        self._logger = logging.getLogger("translatorx.external_ocr")

    def recognize(self, image_bgr: np.ndarray) -> ExternalOcrResponse:
        if not self.endpoint:
            raise ExternalOcrError("未配置外部 OCR 服务地址")
        image = encode_png_base64(image_bgr)
        last_error: ExternalOcrError | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    self.endpoint,
                    json={"img": image, "random_code": self.token},
                    timeout=self.timeout_seconds,
                )
                if response.status_code >= 500:
                    raise requests.HTTPError(f"server status {response.status_code}")
                if response.status_code in (401, 403):
                    raise ExternalOcrError("外部 OCR Token 无效")
                if response.status_code == 402:
                    raise ExternalOcrError("外部 OCR 余额不足")
                if response.status_code >= 400:
                    raise ExternalOcrError("外部 OCR 请求被拒绝")
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise ExternalOcrError("外部 OCR 返回格式异常") from exc
                return parse_external_response(payload, image_bgr.shape[1], image_bgr.shape[0])
            except ExternalOcrError:
                raise
            except (requests.RequestException, TimeoutError):
                last_error = ExternalOcrError("外部 OCR 网络连接失败或超时")
                self._logger.warning("外部 OCR 请求失败（第 %d/%d 次）", attempt + 1, self.max_retries)
                if attempt + 1 < self.max_retries:
                    self.sleep(float(attempt + 1))
        raise last_error or ExternalOcrError("外部 OCR 请求失败")
