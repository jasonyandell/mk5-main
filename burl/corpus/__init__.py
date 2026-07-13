"""Corpus tooling for Burl STaR iterations.

Lives alongside `burl/data/` (raw corpora) and `burl/train/` (trainer).
The trainer reads flat `{messages: [...]}` JSONL; utilities here produce and
manipulate that shape without touching training code.
"""
