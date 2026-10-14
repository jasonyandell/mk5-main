"""The tournament: an append-only, Bradley–Terry-rated ladder of Texas 42 players.

Records are immutable files in a git repository (a Hugging Face dataset), one
commit per faceoff; ratings are derived from them on every read. See
wiki/topics/tournament.md.
"""
