from __future__ import annotations

import inspect

import pytest

from src.video_processor import _call_head_shake_detect


def _make_args():
    return object(), 123.4, 42


def test_call_head_shake_detect_with_three_positional_args():
    """Callables that accept three positional arguments should receive all three."""

    def detect(landmarks, timestamp, frame_number):
        return landmarks, timestamp, frame_number

    args = _make_args()
    result = _call_head_shake_detect(detect, *args)

    assert result == args


def test_call_head_shake_detect_with_two_positional_args():
    """Legacy detectors that accept only two positional arguments should still work."""

    def detect(landmarks, timestamp):
        return landmarks, timestamp

    args = _make_args()
    result = _call_head_shake_detect(detect, *args)

    assert result == args[:2]


def test_call_head_shake_detect_with_varargs():
    """Detectors that accept *args should receive all provided positional arguments."""

    def detect(*args):
        return args

    args = _make_args()
    result = _call_head_shake_detect(detect, *args)

    assert result == args


def test_call_head_shake_detect_unbound_method_with_self_prefix():
    """Ensure helper skips leading 'self' parameter in unbound callables."""

    def detect(self, landmarks, timestamp, frame_number=0):
        return landmarks, timestamp, frame_number

    args = _make_args()
    result = _call_head_shake_detect(detect, *args)

    assert result == (args[1], args[2], 0)


def test_call_head_shake_detect_too_many_required_params_raises_type_error():
    """If callable requires more positional arguments than available, propagate the TypeError."""

    def detect(a, b, c, d):
        return a, b, c, d

    args = _make_args()

    with pytest.raises(TypeError):
        _call_head_shake_detect(detect, *args)


def test_call_head_shake_detect_when_signature_uninspectable(monkeypatch):
    """Fallback to calling the detector directly when inspect.signature fails."""

    def detect(landmarks, timestamp, frame_number):
        return landmarks, timestamp, frame_number

    args = _make_args()

    def raise_signature(_):
        raise TypeError("no signature")

    monkeypatch.setattr(inspect, "signature", raise_signature)
    result = _call_head_shake_detect(detect, *args)

    assert result == args
