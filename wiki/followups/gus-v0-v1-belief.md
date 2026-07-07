Reviewed against code on 2026-07-07 — no issues found.

All headline numbers (v0 34.6%/100% train, v1 37.5%/74% train, per-decision 33/42/45/75%), the 33-token/5-channel layout, the 183-dim v0 features, and the "data not architecture" verdict match commits c04bda3 and 8dbf7f3 and the code in gus/model/ (features.py, dataset_seq.py, tokenize.py, student.py).

- The metrics live only in commit messages, not result artifacts (no eval JSON/log in-repo); a cheap follow-up would be checking in the eval printouts for future audits.
- The per-decision table's "~13"/"~4" unseen counts are exact (13, 4) in the commit message; the tildes on the page are harmlessly conservative.
