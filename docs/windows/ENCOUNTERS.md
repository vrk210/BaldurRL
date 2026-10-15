# Offline encounter trials

`integration.encounter_trial` prepares and analyzes bounded evidence for one
human attack, one autonomous attack, or one resource-exhaustion rejection. It
does not execute BG3, inject input, retry commands, infer game rules, or certify
Windows/BG3 readiness. It leaves the observation and command streams unchanged.

## Create and edit a trial

Create a new version-one template at a path that does not already exist:

```console
python -m integration.encounter_trial init trial.json --kind human_attack
python -m integration.encounter_trial init autonomous.json --kind autonomous_attack
python -m integration.encounter_trial init rejection.json --kind resource_rejection
```

`init` refuses to replace an existing file. Before analysis, replace the null
facts with values from the captured session:

- `trial_id`: a filename-safe identifier, 1–128 ASCII characters.
- `capture_origin`: `unknown`, `synthetic`, or `live_operator_reported`.
- `actor_entity_id`, `target_entity_id`, and `ability_id`: the original opaque
  IDs observed in the game integration stream. Do not substitute simulator IDs.
- `combat_id`: the observed ID, or null when the capture did not provide one.
- `before_sequence` and `after_sequence`: ordered snapshot sequences bounding
  the whole trial. Intermediate snapshots and events are allowed.
- `story_action_id`: an observed nonnegative integer or null. Zero is valid.
- `command_id`: the exact published command for an autonomous trial; otherwise
  null. The analyzer does not guess a command from timestamps or StoryActionID.
- `expected_resource_deltas`: measured-control expectations. Each object has
  exactly `resource_id`, `level`, and `expected_delta`. `level` may be null;
  `expected_delta` is a finite number. At least one expectation with the actual
  resource ID and level is required to verify resource consumption. An empty
  list is inconclusive and never means that an action was free.
- `completion_evidence` and `rejection_evidence`: null or exactly
  `{"source": "...", "reason": "..."}`. Human completion and resource
  rejection require the applicable explicit operator evidence.
- `game_version`, `script_extender_version`, and `notes`: operator-reported
  nonempty strings or null. They are provenance, not proof of behavior.

All objects reject unknown fields. Trial, evidence, and resource types are
strict; missing IDs and bounds stay inconclusive rather than being filled.

## Analyze

```console
python -m integration.encounter_trial analyze trial.json --records session/records
python -m integration.encounter_trial analyze autonomous.json \
  --records session/records --session-dir session --output report.json
```

For an autonomous trial, `--records` must resolve to the supplied session's
`records` subdirectory. This prevents a command result from being combined with
a different observation namespace that happens to reuse the same sequences or
IDs.

The tool reads the observation collection, validates a contiguous embedded
sequence beginning at 1, and
uses the explicit snapshot bounds. A gap, malformed record, wrong boundary type,
or identity contradiction is reported precisely. Generic and targeted ability
events with the same non-null StoryActionID count as one attempt. Separate IDs,
or multiple events that lack an ID, remain separate and therefore ambiguous.
Damage, critical-hit, and miss notifications do not create extra attempts. A
miss still proves execution when the matching attempted action is present.

Human completion needs `completion_evidence`. Autonomous completion needs the
explicitly linked command to match actor, ability, entity target, combat, and
decision snapshot, followed by a `resolved` acknowledgement and a valid fresh
post-execution snapshot attestation. `accepted` alone is inconclusive. UsingSpell
events, animation, or HP change alone do not prove completion.

Resource deltas are `after - before` for the exact actor resource ID and level.
A missing entry on either side is unknown; it is never treated as zero. A turn
or round boundary, or a snapshot with unknown round/active-turn identity, makes
cost attribution inconclusive because refresh may have changed the measured
amount. Boundary controlled-actor or known combat-ID contradictions fail. A rejection passes only with explicit rejection
evidence, available declared resource measurements, matching deltas, and no
contradictory attempted/completed action evidence. Event absence alone does not
prove rejection.

The JSON report contains `PASS`, `FAIL`, or `INCONCLUSIVE`, individual checks
with reasons and supporting record sequences, raw observed resource deltas,
capture provenance, and `live_verified: false`. Exit code 0 means PASS, 1 means
FAIL, and 2 means INCONCLUSIVE. CLI usage, malformed JSON, and validation errors
also exit 2 but emit `status: "ERROR"` and
`error_type: "usage_or_validation_error"` on stderr, distinguishing them from a
valid inconclusive report.

`--output` may replace an ordinary report file, but it refuses the source trial
path and every path inside the supplied raw records, commands, or
acknowledgements directories. Reports therefore cannot overwrite or be mixed
into the evidence collection.

## Synthetic walkthrough

The checked-in `bg3/trials/synthetic_human_attack` fixture demonstrates the
edit-and-analyze shape without BG3:

```console
python -m integration.encounter_trial analyze \
  bg3/trials/synthetic_human_attack/trial.json \
  --records bg3/trials/synthetic_human_attack/records
```

It deliberately uses synthetic IDs, snapshots, evidence, and a miss. A PASS
means only that those synthetic records support their bounded claim;
`capture_origin` remains `synthetic` and `live_verified` remains false. An
`unknown` capture origin prevents PASS until it is explicitly labeled.

## Checks still required on Windows

Run separate future live trials to discover the actual actor, target, ability,
combat, resource, and level IDs; record the installed BG3 and Script Extender
versions; and capture fresh before/after snapshots before any turn refresh. A
human trial must record explicit observed completion. An autonomous trial must
exercise the approved normal UI/input route, publish one command, and retain its
resolved acknowledgement and post-execution attestation. The exhausted-resource
trial must record the visible rejection layer and rule out Extra Attack, Action
Surge, haste, or other free attacks. These live checks remain unperformed by the
offline harness.
