# Training campaigns and scratch work separated

Subsequent status: the user stopped the campaign and authorized the
[run-owned model-storage cleanup](2026-11-06-run-owned-models.md). That follow-up
implemented the output defaults deferred in this original receipt.

User-requested organization: `runs/` for training campaigns; `scratch/` for ad hoc
testing, audits, benchmarks and debugging; `models/` stays the durable store.
Both working-output directories are Git-ignored.

## Restored from trash to `runs/`

Source: `/home/vita/data/.Trash-1000/files/runs/`.

- `20261104-151420-size9-training`
- `20261104-215404-pass-balanced`
- `20261105-092434-leaky-value`
- `20261105-214158-score-ownership-size9`
- `20261106-111101-losing-pass-fix`
- `20261106-122518-fresh-search-annotations` (campaign preparation, not a completed run)
- `20261106-130621-fresh-policy256`
- `20261106-133757-policy256-score-bound`

These were same-filesystem directory renames to their original repository paths,
not newly launched training. All destination paths were checked for conflicts
before moving. Every entry's device, inode, size and mtime was compared before
and after each move; all matched. Old model stores, store-migration backups,
benchmark artifacts and other non-training trash were left untouched. The trash
metadata still describes the remaining partially restored `runs` container.

## Follow-up: campaigns below 20 completed rounds returned to trash

At the user's request, these four campaigns were returned to their original
locations beneath `/home/vita/data/.Trash-1000/files/runs/`:

- `20261106-111101-losing-pass-fix`: no new completed rounds.
- `20261106-122518-fresh-search-annotations`: preparation only.
- `20261106-130621-fresh-policy256`: five initial epochs, no completed rounds.
- `20261106-133757-policy256-score-bound`: one completed round.

The original trash container and its `.trashinfo` remain intact. Destinations
were checked for conflicts and no live process referenced these campaign paths.
Every entry's device, inode, size and mtime matched after the moves. Five campaigns
remain in `runs/`; the active campaign was untouched.

## Moved from `runs/` to `scratch/`

- `20261106-182938-pass-audit`
- `20261106-190000-old-critic-comparison`
- `20261106-193548-midgame-passes`
- `20261106-200313-playout-audit`
- `20261106-213310-loss-gradients`

The same entry-level rename checks passed for these directories. Documentation
now points to the new artifact paths. The old/new critic comparison script's old
campaign location was corrected to its restored `runs/` path. Saved logs, JSON
references, checkpoint metadata and source snapshots were not rewritten; their
historical paths may refer to the former locations.

## Active campaign and deferred defaults

`runs/20261106-142353-scratch-score/` is an active training campaign, despite its
name, and was not moved, renamed, restarted or otherwise modified.

Its launcher pins every Python module's hash. To preserve that runtime guard,
production Python code was not changed. Standalone jobs must currently specify
`--output scratch/...` explicitly. Updating their automatic defaults and adding
`scratch/` to durable-store exclusion guards was deferred at this point; both
were subsequently implemented in the [storage cleanup](2026-11-06-run-owned-models.md)
after the user stopped the campaign.
