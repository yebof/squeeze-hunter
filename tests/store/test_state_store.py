"""P2 — the runtime snapshot store: atomic JSON, round-trips, tolerant load."""

from __future__ import annotations

from pathlib import Path

from squeeze_hunter.store.state import JsonStateStore


def test_roundtrip_and_atomic_write(tmp_path: Path) -> None:
    path = tmp_path / "state" / "runtime.json"
    store = JsonStateStore(path)
    assert store.load() is None
    snap = {"version": 1, "positions": {"GME": {"qty": 100}}, "killswitch": {"active": False}}
    store.save(snap)
    assert path.is_file()
    assert not list(path.parent.glob("*.tmp*")), "temp file left behind"
    loaded = store.load()
    assert loaded is not None
    assert loaded.pop("saved_at")
    assert loaded == snap


def test_corrupt_file_loads_as_none_and_is_kept_aside(tmp_path: Path) -> None:
    path = tmp_path / "runtime.json"
    path.write_text("{not json")
    store = JsonStateStore(path)
    assert store.load() is None
    assert any(p.name.startswith("runtime.json.corrupt") for p in tmp_path.iterdir())
