# Repetition metrics for token-only rollouts

FMP rollouts intentionally keep `Sample.response` empty and store the response
as token IDs, with `metadata["response_decoded"] = False`. Checking that empty
string incorrectly reported no repetition.

## Behavior

The agentic generation worker now temporarily decodes the final response tokens
using its existing tokenizer, after the session server has merged and truncated
the response. This supports both v1 and v2 results, including v2 sample lists;
readable v2 samples keep using their existing response text. It runs the
existing detector and stores `metadata["repetition_metric"]`: a boolean result,
detector version, response length, and digest of the response tokens. The decoded
text is discarded. Rewards, tokens, masks, log probabilities and statuses are
unchanged; this is monitoring, not a reward penalty or training filter.

The heuristic is unchanged: more than 10,000 characters, with a zlib compression
ratio greater than 10 on the last 10,000 characters. It decodes the entire response
span, including intermediate tool/environment content, without filtering by the
loss mask or dropping special tokens. The initial prompt is excluded. This
preserves the previous merged-text interpretation, not an assistant-only detector.

Default train and eval logging, including correctness/category splits, use:

- `repetition_frac`: fraction of inspected samples flagged as repetitive.
- `repetition_coverage`: fraction of all samples that could be inspected.

For example, one repetitive sample, one non-repetitive sample and one unknown
sample produce fraction `0.5` and coverage `2/3`. An entirely unknown group emits
coverage `0` and omits the fraction; it does not report a misleading zero.

Readable legacy samples continue using their text. Token-only samples require a
valid stored result. Missing, malformed, wrong-version or stale results are
unknown. Retries, response replacement, merging and truncation invalidate the
result; a token digest additionally detects unanticipated edits, including
same-length replacements. Readers validate each sample once per logging pass and
reuse its result across groups. JSON/checkpoint serialization retains the result.
The session wire codec removes inherited input telemetry before reading the new
response, but preserves valid telemetry carried by the response itself.

Decoding errors warn without logging response text and leave the result unknown;
they do not abort a rollout. Decoding runs through the default thread pool, with
no sample mutation in the background thread, so cancellation cannot attach a
late result. Threads do not guarantee freedom from Python GIL contention.

## Tests and limits

`tests/fast/utils/test_repetition.py` checks the threshold, repeated/varied and
Unicode content, prompt exclusion, stale tokens, retries, truncation, merging,
serialization, malformed data, failures and cancellation. It checks that only the
new metadata field changes. `tests/fast/rollout/test_repetition_metrics.py` covers
train/eval logging and every group split. Agentic generation integration tests in
`tests/fast/rollout/generate_hub/test_agentic_tool_call.py` verify measurement after
final truncation for train/eval and both session versions, with real session HTTP
handlers and a mock inference backend. These three files add 43 cases. The
existing 16-trajectory v1/v2 parity test also compares repetition results while
retaining bitwise comparisons of all training fields.

The local CPU benchmark used the configured BBQ FMP tokenizer, not model weights:
about 3 ms per 16K-token response and 17–20 ms per 98K-token response, plus about
0.2–2.4 ms for digest validation. The result occupies roughly 110 JSON bytes.
Sixteen concurrent 98K-token responses took about 362 ms and delayed the event
loop by up to 340 ms. These are local observations, not throughput guarantees;
large-scale rollout latency still needs validation. Evidence and the benchmark
script are in `/tmp/fmp-local-validation.nVQslY` on the validation host
(`repetition-benchmark-rebased.json`, `benchmark_repetition.py`). These measurements
use the `fmp-rl-run-rebased` port, the upstreamed Miles image, and OpenAI 2.6.1.

This change does not repair generic text-based graders, partial-rollout text
checks, or display/debug output. FMP uses the stored Harbor reward rather than a
generic text grader. GPU training and the full sandbox lifecycle are separate
validation gates.
