from __future__ import annotations

import base64
import logging
import math
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
    coordinate_warning: str = ""


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
    if isinstance(payload, (list, tuple)):
        if len(payload) == 3 and isinstance(payload[0], (list, tuple)) and isinstance(payload[1], str):
            return [{"box": payload[0], "text": payload[1], "score": payload[2]}]
        if (
            len(payload) == 2
            and isinstance(payload[0], (list, tuple))
            and isinstance(payload[1], (list, tuple))
            and payload[1]
            and isinstance(payload[1][0], str)
        ):
            return [{
                "box": payload[0],
                "text": payload[1][0],
                "score": payload[1][1] if len(payload[1]) > 1 else 1.0,
            }]
        result: list[dict[str, Any]] = []
        for value in payload:
            result.extend(_walk_candidates(value))
        return result
    if not isinstance(payload, dict):
        return []
    result = []
    for key in (
        "data", "result", "results", "items", "ocr", "ocr_result", "texts", "text_list",
        "words_result", "lines", "blocks", "regions", "content",
    ):
        if key in payload:
            result.extend(_walk_candidates(payload[key]))
    if result:
        return result
    text = _first(payload, ("text", "words", "content", "value", "transcription"))
    return [payload] if isinstance(text, str) and text.strip() else []


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _box_coordinates(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, dict):
        for key in (
            "box", "bbox", "points", "location", "position", "rect", "bounding_box",
            "boundingBox", "vertices", "polygon", "text_box",
        ):
            if key in value:
                coords = _box_coordinates(value[key])
                if coords is not None:
                    return coords
        left = _number(_first(value, ("left", "x1", "x")))
        top = _number(_first(value, ("top", "y1", "y")))
        right = _number(_first(value, ("right", "x2")))
        bottom = _number(_first(value, ("bottom", "y2")))
        width = _number(_first(value, ("width", "w")))
        height = _number(_first(value, ("height", "h")))
        if right is None and left is not None and width is not None:
            right = left + width
        if bottom is None and top is not None and height is not None:
            bottom = top + height
        if None not in (left, top, right, bottom):
            return left, top, right, bottom
    elif isinstance(value, (list, tuple)):
        if len(value) == 4 and all(_number(number) is not None for number in value):
            return tuple(_number(number) for number in value)
        if len(value) == 8 and all(_number(number) is not None for number in value):
            value = list(zip(value[::2], value[1::2]))
        if len(value) >= 2:
            points = []
            for point in value:
                if isinstance(point, dict):
                    coords = (_number(point.get("x")), _number(point.get("y")))
                elif isinstance(point, (list, tuple)) and len(point) == 2:
                    coords = (_number(point[0]), _number(point[1]))
                else:
                    return None
                if None in coords:
                    return None
                points.append(coords)
            return (
                min(point[0] for point in points), min(point[1] for point in points),
                max(point[0] for point in points), max(point[1] for point in points),
            )
    return None


def _friendly_error(message: str) -> str:
    lowered = message.casefold()
    if any(word in lowered for word in ("token", "auth", "unauthor", "invalid key", "密钥")):
        return "外部 OCR Token 无效"
    if any(word in lowered for word in ("balance", "余额", "quota", "insufficient")):
        return "外部 OCR 余额不足"
    return "外部 OCR 服务返回错误"


def parse_external_response(payload: Any, width: int, height: int) -> ExternalOcrResponse:
    if not isinstance(payload, (dict, list)) or width <= 0 or height <= 0:
        raise ExternalOcrError("外部 OCR 服务返回格式异常")
    metadata = payload if isinstance(payload, dict) else {}
    error = _first(metadata, ("error", "error_msg", "message"))
    code = _first(metadata, ("code", "status", "errno"))
    raw_items = _walk_candidates(payload)
    if not raw_items and (error or str(code).casefold() in {"401", "403", "402", "unauthorized", "error", "fail", "false"}):
        message = str(error or code)
        if str(code) == "402":
            raise ExternalOcrError("外部 OCR 余额不足")
        raise ExternalOcrError(_friendly_error(message))
    items: list[OcrItem] = []
    invalid_count = 0
    for raw in raw_items:
        text = str(_first(raw, ("text", "words", "content", "value", "transcription"), "")).strip()
        if not text:
            continue
        coords = _box_coordinates(raw)
        if coords is None:
            invalid_count += 1
            continue
        left, top, right, bottom = coords
        if (
            not all(math.isfinite(value) for value in coords)
            or left < 0 or top < 0 or right <= left or bottom <= top
            or right > width or bottom > height
        ):
            invalid_count += 1
            continue
        confidence = _number(_first(raw, ("confidence", "score"), 1.0))
        items.append(OcrItem(
            box=((left, top), (right, top), (right, bottom), (left, bottom)),
            text=text,
            confidence=confidence if confidence is not None else 1.0,
        ))

    warning = ""
    if invalid_count:
        logging.getLogger("translatorx.external_ocr").warning(
            "外部 OCR 坐标缺失或无效：%d/%d 条，保留 %d 条有效位置", invalid_count, len(raw_items), len(items),
        )
        if not items:
            raise ExternalOcrError("外部 OCR 未返回可用的逐条文字坐标，无法按原文位置显示；请启用坐标返回或提供脱敏响应样例以适配")
        warning = f"{invalid_count} 条文字缺少有效坐标，未显示；请检查外部 OCR 坐标返回"
    balance = _number(_first(metadata, ("balance", "remaining_balance", "money", "amount")))
    member_status = str(_first(metadata, ("member_status", "membership", "vip"), "") or "")
    return ExternalOcrResponse(items=items, balance=balance, member_status=member_status, coordinate_warning=warning)


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
