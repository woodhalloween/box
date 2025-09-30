# test_video_processor_ffmpeg_import_fallback.py
import importlib
import sys
import types


def test_make_frame_iter_fallback_on_import_error(monkeypatch):
    """
    Force 'from .io.ffmpeg_io import make_frame_iter' to fail so that
    the fallback 'make_frame_iter = None' branch is executed.
    """
    # Insert a dummy module at the exact path the relative import resolves to,
    # but WITHOUT 'make_frame_iter' symbol → 'from ... import make_frame_iter' fails.
    dummy = types.ModuleType("src.io.ffmpeg_io")
    monkeypatch.setitem(sys.modules, "src.io.ffmpeg_io", dummy)

    # Now reload the module under test so its top-level try/except re-executes
    import src.video_processor as vp

    vp = importlib.reload(vp)

    # Assert the fallback was taken
    assert vp.make_frame_iter is None
