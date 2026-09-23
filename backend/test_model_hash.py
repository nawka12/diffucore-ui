"""Tests for the model-file hashes and their cache (pure functions)."""

from __future__ import annotations

import hashlib
import struct

import pytest

import model_hash as mh


def _write_safetensors(path, tensor_bytes: bytes,
                       header: bytes = b'{"__metadata__":{}}') -> str:
    """Write a minimal safetensors file; return its expected SwarmUI tensor hash."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))   # 8-byte little-endian header len
        f.write(header)
        f.write(tensor_bytes)
    return "0x" + hashlib.sha256(tensor_bytes).hexdigest()


def test_tensor_hash_skips_header(tmp_path):
    """The tensor hash covers only the tensor data, not the header."""
    p = tmp_path / "m.safetensors"
    expected = _write_safetensors(p, b"\x01\x02\x03\x04tensor-bytes")
    assert mh._tensor_hash(p) == expected

    q = tmp_path / "n.safetensors"
    _write_safetensors(q, b"\x01\x02\x03\x04tensor-bytes",
                       header=b'{"__metadata__":{"note":"different header"}}')
    assert mh._tensor_hash(q) == expected


def test_tensor_hash_bad_file_is_none(tmp_path):
    p = tmp_path / "tiny.safetensors"
    p.write_bytes(b"\x00\x00")   # < 8 bytes: struct.error, swallowed
    assert mh._tensor_hash(p) is None


def test_hash_file_returns_tensor_and_full(tmp_path):
    """One pass yields both the header-skipped tensor hash and the whole-file SHA256."""
    p = tmp_path / "m.safetensors"
    header = b'{"__metadata__":{}}'
    tensor_bytes = b"payload-abc"
    with open(p, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        f.write(tensor_bytes)
    tensor, full = mh._hash_file(p)
    assert tensor == "0x" + hashlib.sha256(tensor_bytes).hexdigest()   # data only
    whole = struct.pack("<Q", len(header)) + header + tensor_bytes
    assert full == hashlib.sha256(whole).hexdigest()                   # whole file, no "0x"


def test_clean_model_name_unwraps_family():
    assert mh.clean_model_name("Anima(foo.safetensors)") == "foo.safetensors"
    assert mh.clean_model_name("FLUX(bar.safetensors)") == "bar.safetensors"
    assert mh.clean_model_name("plain.safetensors") == "plain.safetensors"
    # strip_ext (SwarmUI's `model` param) drops only a known model extension.
    assert mh.clean_model_name("Anima(foo.safetensors)", strip_ext=True) == "foo"
    assert mh.clean_model_name("v1.0-model.ckpt", strip_ext=True) == "v1.0-model"
    assert mh.clean_model_name("no-ext-name", strip_ext=True) == "no-ext-name"


@pytest.fixture
def _isolated_cache(tmp_path, monkeypatch):
    """Point the hash cache at a throwaway file and reset the in-memory copy."""
    monkeypatch.setattr(mh, "_CACHE_PATH", tmp_path / ".model_hashes.json")
    monkeypatch.setattr(mh, "_CACHE", None)
    yield
    monkeypatch.setattr(mh, "_CACHE", None)


def test_ensure_then_get_hash_roundtrip(tmp_path, _isolated_cache):
    p = tmp_path / "model.safetensors"
    expected = _write_safetensors(p, b"payload-1234")
    assert mh.ensure_hash(p) == expected
    assert mh.get_hash(p) == expected        # served from cache


def test_get_hash_none_before_compute(tmp_path, _isolated_cache):
    p = tmp_path / "model.safetensors"
    _write_safetensors(p, b"payload")
    assert mh.get_hash(p) is None


def test_get_hash_busts_on_file_change(tmp_path, _isolated_cache):
    p = tmp_path / "model.safetensors"
    _write_safetensors(p, b"payload")
    mh.ensure_hash(p)
    with open(p, "ab") as f:                 # size changes
        f.write(b"more-bytes")
    assert mh.get_hash(p) is None


def test_get_hash_none_for_non_safetensors(tmp_path, _isolated_cache):
    p = tmp_path / "model.ckpt"
    p.write_bytes(b"whatever")
    assert mh.get_hash(p) is None
    assert mh.ensure_hash(p) is None
    assert mh.get_hash(None) is None


def test_get_autov2_after_ensure(tmp_path, _isolated_cache):
    p = tmp_path / "model.safetensors"
    header = b'{"__metadata__":{}}'
    with open(p, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        f.write(b"weights")
    assert mh.get_autov2(p) is None
    mh.ensure_hash(p)
    whole = struct.pack("<Q", len(header)) + header + b"weights"
    assert mh.get_autov2(p) == hashlib.sha256(whole).hexdigest()[:10]
    assert mh.get_autov2(p.with_suffix(".ckpt")) is None
    assert mh.get_autov2(None) is None


def test_ensure_upgrades_tensor_only_entry(tmp_path, _isolated_cache):
    """A tensor-only entry (no sha256) is refreshed so AutoV2 fills in."""
    p = tmp_path / "model.safetensors"
    _write_safetensors(p, b"weights")
    st = p.stat()
    mh._load_cache()[mh._rel_key(p)] = {   # the old on-disk schema
        "mtime": st.st_mtime_ns, "size": st.st_size, "hash": "0xdeadbeef"}
    assert mh.get_autov2(p) is None
    mh.ensure_hash(p)
    assert mh.get_autov2(p) is not None
