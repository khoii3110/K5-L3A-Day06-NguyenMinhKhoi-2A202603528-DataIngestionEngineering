# PostgreSQL 17 benchmark draft

`postgresql17_questions.jsonl` contains 20 questions: 15 dev and 5 test.
The test questions use five held-out source sections; no two questions use
the same gold section. Keep this split and the test labels unchanged after
retrieval results are observed. This is a prepared benchmark, not an evaluated
result.

The corpus is HTML without page numbers, so `gold_page` stores the source
section ID. `gold_source_id` matches the manifest. Each evidence quote must be
checked against the cited section before running a benchmark. `filter` is
provided query context, not a value inferred from the answer. The chapter
filter is supported by the corpus metadata and is used only when present.
