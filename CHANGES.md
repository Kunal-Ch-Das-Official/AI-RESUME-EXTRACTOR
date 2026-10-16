# v4.2 – reliability + token optimisation

## Why `profile-summary/generate` returned 504
1. Gemini answered `ServerError` (503 overloaded). The old retry code read `exc.status_code`,
   which google-genai does not have (it is `exc.code`), so errors were never classified.
2. Gemini then OpenRouter ran **sequentially in worker threads**. httpx/SDK timeouts are per
   network read, not total, and `asyncio.wait_for` cannot cancel a thread, so the "16s + 14s"
   budget was never enforced; the 36s route limit fired first -> 504.
3. `max_output_tokens=1000` on Gemini 2.5 Flash is shared with hidden thinking tokens.

## Fix: `app/services/llm.py`
Single async router for every AI feature: hard total deadline (real cancellation), failover
Gemini -> Gemini fallback model -> OpenRouter, circuit breaker, validation inside the loop,
30s cache + single-flight, token-usage logging. Exhausted budget returns a clean
`503 + Retry-After` instead of a 504.

## Token savings (approx.)
| Call | Before | After |
|---|---|---|
| Resume extraction input | resume x2 + 25k-char schema x2 (~15k tokens for a 2-page CV) | resume x1 + ~600-token instructions |
| Resume extraction output | full ResumeInfo + evidence list (unused) | compact list shape, no evidence |
| Profile summary | title/style/summary x4 + thinking | summaries only (titles fixed server-side), thinking off |
| Responsibilities | titles + bullets | bullets only, tense decided in code |
| Embeddings | 1 request per chunk | 1 request per 100 chunks |
| Repeat clicks / re-uploads | paid every time | served from cache |

## Other fixes
* Gemini could not see the item shape of `dict[str, Item]` fields (schema keyword stripped);
  extraction now uses a list-based wire format converted to ResumeInfo in code.
* AI URLs/emails that do not appear in the source are dropped (anti-hallucination).
* Merge no longer mixes placeholder local records ("Employer not specified") with AI records.
* PDF hyperlinks (LinkedIn/GitHub) now reach the text; PDF blocks are read in reading order;
  DOCX hyperlinks captured.
* Embeddings/retriever were blocking the event loop; now async + shared client.
* Confidence no longer loses 0.12 on every import for "no RAG knowledge".
* Logging configured (INFO logs were invisible), CORS origins configurable, `/health` no
  longer pings an unconfigured DB, runtime vs dev requirements split.
* Optional `regenerate: true` on the two generate endpoints bypasses the 30s cache.

## v4.2.1 – model retirement handling
* `gemini-2.5-flash` was shut down by Google on 2026-06-17 (404 NOT_FOUND). Defaults are now
  `gemini-3.5-flash` / `gemini-3.5-flash-lite`.
* `app/services/model_resolver.py` asks the API which models the key can use and swaps in the
  best stable Flash model when a configured name is gone (logged as an ERROR with the right names).
* Gemini 3.x uses `thinking_level=MINIMAL` (2.x keeps `thinking_budget=0`); temperature is left at
  the model default for 3.x, as Google recommends.
* Bad key / unknown model on every provider now returns a clear "AI is not configured correctly"
  message instead of "try again shortly".
