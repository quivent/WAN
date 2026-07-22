#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import shutil
import time


def main() -> int:
    native_repo = pathlib.Path(__import__("os").environ.get("WAN_NATIVE_REPO", "/opt/Wan2.2")).expanduser()
    attention = native_repo / "wan" / "modules" / "attention.py"
    if not attention.is_file():
        raise SystemExit(f"missing native attention module: {attention}")

    text = attention.read_text(encoding="utf-8")
    if "scaled_dot_product_attention fallback" in text:
        print(f"already patched: {attention}")
        return 0

    needle = """    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
"""
    replacement = """    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    if not FLASH_ATTN_2_AVAILABLE and not FLASH_ATTN_3_AVAILABLE:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                "Padding mask is disabled when using scaled_dot_product_attention fallback. It can have a significant impact on performance."
            )
        if q_scale is not None:
            q = q * q_scale
        q_dense = q.transpose(1, 2).to(dtype)
        k_dense = k.transpose(1, 2).to(dtype)
        v_dense = v.transpose(1, 2).to(dtype)
        x = torch.nn.functional.scaled_dot_product_attention(
            q_dense, k_dense, v_dense, attn_mask=None, is_causal=causal, dropout_p=dropout_p)
        return x.transpose(1, 2).contiguous().type(out_dtype)

    def half(x):
"""
    if needle not in text:
        raise SystemExit("Wan2.2 attention.py patch point not found")

    backup = attention.with_suffix(attention.suffix + f".bak.{time.strftime('%Y%m%d%H%M%S')}")
    shutil.copy2(attention, backup)
    attention.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
    print(f"patched: {attention}")
    print(f"backup:  {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
