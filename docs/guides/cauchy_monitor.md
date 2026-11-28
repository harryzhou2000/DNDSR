# Cauchy HTTP history and monitoring

The hub uses HTTP for numerical and boundary histories for both local and
remote runs. Local control files are discovery inputs, not CSV transport.
An SSH tunnel to the solver/companion port is sufficient; the hub does not
need a mounted remote output directory. Existing solvers without the query
endpoint require the companion below or a new executable on their next run.
The old `/api/v1/history` endpoint remains the 256-record telemetry ring for
backward compatibility; it is not a full-history service.

## Query contract

`GET /api/v1/history/query` requires the same bearer token as the control API.
Only CSV paths registered in the server's run manifest are selectable.

| Parameter | Meaning |
| --- | --- |
| `domain` | Nonnegative domain index, default 0 |
| `boundary` | Optional canonical boundary name; omitted selects nonlinear/step history |
| `limit` | Returned page size, 1–10000, default 2000 |
| `cursor` | Opaque cursor from the previous response; omit for a fresh query |
| `kind` | `all`, `nonlinear`, `stage`, or `step` |
| `axis` | Interval axis: `iteration` (default), `time`, `step`, or `wall` |
| `start`, `end` | Optional finite inclusive interval endpoints |
| `columns` | Optional comma-separated column names |

Response schema `dndsr.cauchy.history-query.v1` contains `header`, `rows`,
`indices`, `totalRows`, `returnedRows`, `nextCursor`, `hasMore`, `generation`
and `reset`. `totalRows` counts complete raw records before filtering;
`indices` are their zero-based positions. Keep the same filters while paging.
Follow `nextCursor` until `hasMore=false`, then poll that cursor for appends.
An empty page can advance the cursor across filtered-out rows.

The cursor belongs to one history generation. Replacement, truncation or
changes to an earlier segment reset the index. A stale cursor returns the
first new-generation page with `reset=true`; the client must discard its old
rows before appending that response. Restarting the server also resets cursors.
Malformed queries return 400, unknown histories 404, and unavailable files 503.

The service caches complete-row offsets at 256-row intervals, not all parsed
history rows. It scans only new bytes when files grow and defers a trailing row
until its newline arrives. Page reads seek to the nearest indexed block. CSV
schema must be consistent across numbered segments. Cauchy history is a numeric,
single-line-record format; nonfinite values are represented by JSON null.
File indexing runs in HTTP worker threads under a history-cache mutex and does
not acquire solver/MPI state locks or run on the numerical thread.

`GET /api/v1/manifest` publishes domain and boundary descriptors over HTTP.
The hub caches received rows, requests append pages, and resamples them for
plotting. Its own `/api/runs/<id>/history?maxPoints=0` returns all rows collected
so far. A very large initial history catches up over successive requests,
with an explicit warning while incomplete. The hub never substitutes the
telemetry ring and presents it as full history.

## Python companion for existing or completed runs

The companion uses Python's standard library only; it does not load DNDSR
bindings, MPI, CUDA, Flask or solver fields. It can serve growing CSVs as well
as a completed output directory. It is read-only and cannot steer a solver.

```bash
python3 scripts/cauchy_history.py \
  --manifest /absolute/path/run-run-manifest.json \
  --host 127.0.0.1 --port 53050 \
  --token-file /private/path/orion-history.token
```

The token file is created with mode 0600 when absent. Attach the companion URL
and token in the hub's Remote section. For a remote machine, forward its port:

```bash
ssh -L 53050:127.0.0.1:53050 remote-host
```

Run it under a user service or tmux when persistent availability is required.
Keep the manifest, resolved config and all CSV segments together when moving
an archive; relative paths can be resolved alongside the manifest. The companion
does not follow arbitrary file paths supplied by an HTTP caller. Stop the
companion independently of the simulation.

## Browser behavior

Refresh intervals are selectable from 1, 2, 5, 10, 30 and 60 seconds, default 2.
The choice persists in browser local storage. Hidden tabs and disabled live mode
pause polling; slow requests do not overlap timer-driven refreshes. Manual
refresh remains available. The optional status SSE endpoint has its independent
default 2-second interval; the current browser uses ordinary HTTP polling.

Plot trace visibility persists across refreshes. Bounds/plot-mode changes use
serialized Plotly updates with CSS-safe trace identities. Both dark and light
themes provide explicit tooltip colors. The attach dialog has separate titled
local-discovery and remote-URL sections.
