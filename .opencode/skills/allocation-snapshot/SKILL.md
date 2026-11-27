---
name: allocation-snapshot
description: Generate and verify interactive memory treemaps from DNDSR Runtime allocation snapshots for solver studies and reports.
---

# Runtime allocation snapshots

Use the maintained generator from the DNDSR checkout. It produces the offline, drillable treemap with placement filters, byte counts, and leaf details.

```sh
python3 scripts/diagnostics/allocation_explorer.py \
  --input 'Step 1, rank 0=/path/to/run-memory-R0-S1.txt' \
  --output /path/to/study/allocation.html
```

Repeat `--input LABEL=FILE` for snapshots that the reader should be able to switch between. A filename and its process ID can be exposed in a single-rank export. Check the embedded provenance before publishing. The [generator guide](../../../scripts/diagnostics/README.md) describes gzip input, scoped fragments and browser checks.

For an all-rank view, parse each rank at the **same saved instant** with `parse_snapshots`. Check the expected rank set, reconcile each rank's unique bytes with its header, then sum by recorded owner, placement and provider. Rank-local IDs are meaningful only within their source snapshot. Preserve unnamed capacity and count aliases once. Label aggregate leaves as groups and record how many buffers and ranks contributed. Keep distinct steps selectable. State whether a reported peak is a maximum over ranks or a sum of independently timed high-water marks.

Pass the reduced `dndsr-allocation-explorer-v1` dataset to the same renderer with `--data DATA.json --output EXPLORER.html`. Verify host and GPU totals against the study at every saved instant. Registry capacity excludes driver and provider-private storage.

Open the finished HTML and check tile drilldown, **Up**, placement filtering, and snapshot selection when applicable. For a web report, test inside the rendered page too. A GitHub blob link shows source. An offline file opens directly in a browser, and a post can display a pinned copy in a sandboxed iframe.
