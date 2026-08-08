"""
notes.block_chunker_inator — block-aligned chunking for notes (gap 1, stream A).

Blocks are the unit of ownership/anchoring (a stable AppFlowy block id); chunks
are the unit of retrieval (~512 tokens). The invariant:

    a chunk boundary always lands ON a block boundary — EXCEPT when a single
    oversized block is subdivided, in which case every piece carries that one
    block's id (is_partial=True).

So no chunk ever holds part-of-block-A *and* part-of-block-B. Three regimes:

  1. tiny/normal blocks  → PACK consecutive whole blocks up to the cap
                           (1 chunk : N blocks). Fixes "a chunk per 3-word block".
  2. block ≤ cap         → packs with neighbours, or stands alone.
  3. block > cap         → SPLIT; every piece stamped with the same block_id,
                           is_partial=True (N chunks : 1 block).

Atomic block types (tables, code, callouts) BYPASS the cap — kept whole even if
oversized (shredding a table destroys its meaning; mirrors the exam-question
path). Headings start a fresh chunk. No character overlap: block boundaries are
already coherent seams.

This module is pure/deterministic — the markdown/registry wiring lives elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

# Block types kept whole regardless of size (declarative, not hardcoded per case).
ATOMIC_TYPES = frozenset({"table", "code", "callout"})
HEADING_TYPES = frozenset({"heading"})

DEFAULT_MAX_TOKENS = 512


@dataclass(frozen=True)
class Block:
    block_id: str
    block_type: str
    text: str


@dataclass(frozen=True)
class BlockRef:
    block_id: str
    is_partial: bool = False


@dataclass
class ChunkPlan:
    index: int
    text: str
    blocks: list[BlockRef] = field(default_factory=list)


def _tiktoken_len(text: str) -> int:
    import tiktoken

    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def _split_oversized(text: str, max_tokens: int, token_len: Callable[[str], int]) -> list[str]:
    """Split one block's text into ≤max_tokens pieces, greedily by paragraphs
    then lines then words. Only used for a single non-atomic oversized block."""
    units = [u for u in text.split("\n\n") if u.strip()] or [text]
    pieces: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for unit in units:
        ut = token_len(unit)
        if ut > max_tokens:  # a single paragraph still too big → break by words
            if cur:
                pieces.append("\n\n".join(cur))
                cur, cur_tok = [], 0
            words = unit.split(" ")
            wbuf: list[str] = []
            wtok = 0
            for w in words:
                wt = token_len(w + " ")
                if wbuf and wtok + wt > max_tokens:
                    pieces.append(" ".join(wbuf))
                    wbuf, wtok = [], 0
                wbuf.append(w)
                wtok += wt
            if wbuf:
                pieces.append(" ".join(wbuf))
            continue
        if cur and cur_tok + ut > max_tokens:
            pieces.append("\n\n".join(cur))
            cur, cur_tok = [], 0
        cur.append(unit)
        cur_tok += ut
    if cur:
        pieces.append("\n\n".join(cur))
    return pieces or [text]


def pack_blocks(
    blocks: Iterable[Block],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    token_len: Callable[[str], int] | None = None,
    atomic_types: frozenset[str] = ATOMIC_TYPES,
) -> list[ChunkPlan]:
    """Pack blocks into block-aligned chunks. See module docstring for regimes."""
    tok = token_len or _tiktoken_len
    chunks: list[ChunkPlan] = []
    cur_parts: list[str] = []
    cur_refs: list[BlockRef] = []
    cur_tok = 0

    def flush() -> None:
        nonlocal cur_parts, cur_refs, cur_tok
        if cur_refs:
            chunks.append(ChunkPlan(len(chunks), "\n\n".join(cur_parts), cur_refs))
        cur_parts, cur_refs, cur_tok = [], [], 0

    for b in blocks:
        t = tok(b.text)
        is_heading = b.block_type in HEADING_TYPES
        is_atomic = b.block_type in atomic_types

        # Regime 3: a single non-atomic block over the cap → split, same id.
        if t > max_tokens and not is_atomic:
            flush()
            for piece in _split_oversized(b.text, max_tokens, tok):
                chunks.append(
                    ChunkPlan(len(chunks), piece, [BlockRef(b.block_id, is_partial=True)])
                )
            continue

        # Boundary before this block if it's a heading, atomic, or won't fit.
        if cur_refs and (is_heading or is_atomic or cur_tok + t > max_tokens):
            flush()

        cur_parts.append(b.text)
        cur_refs.append(BlockRef(b.block_id, is_partial=False))
        cur_tok += t

        # Atomic block stands alone (kept whole, even if oversized).
        if is_atomic:
            flush()

    flush()
    return chunks
