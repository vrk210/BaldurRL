# BaldurRL

BaldurRL is building toward an agent that plays Baldur's Gate 3. The maintained
system is a frozen M0–M6 simulator, policy/evaluation tools, and a passive BG3
collector with a Python semantic command protocol and mock executor. Nothing
has run against the game yet; the [integration contract](docs/BG3_INTEGRATION.md#python-use)
lists what is implemented and what still needs live verification.

[Architecture and module inventory](docs/ARCHITECTURE.md) explain ownership and
supported entry points. Simulator contracts live in the owning modules' docstrings
and field/action metadata. [RESULTS.md](docs/RESULTS.md) caches the measured
results that current decisions rest on, and the full experiment history is on the
`history` branch;
[GAME_KNOWLEDGE.md](docs/GAME_KNOWLEDGE.md) separates simulator hypotheses from
verified game facts. Agent working rules are in [AGENTS.md](AGENTS.md).

## Install and verify

Use Python 3.11 for the complete development/simulator environment. Versions
in requirements.txt match the tested environment, including direct and portable
transitive dependencies. Wheels and platform runtime libraries differ by OS;
this is not a byte-identical cross-platform lock. Windows CI installs from this
manifest and runs the full suite. The smaller harness is tested on 3.11/3.13.

Windows PowerShell, from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/check_repository.py
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
```

macOS/Linux:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/check_repository.py
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -q
```

For capture-only Windows setup, use the [H0 runbook](bg3/h0/README.md#first-windows-session).
Runtime integration tools use the standard library. requirements-harness-test.txt
adds pytest/Lua for offline verification; requirements-quality.txt adds the
repository CI tools. A Lua-test skip is not Lua verification.

## Run the maintained tools

Use the Python executable from your environment for the commands below. Outputs
belong under ignored runs/. Existing smoke seeds are for regression checks;
reserve fresh experiment ranges with `python -m evaluation.seeds --check START COUNT`.
Never use frozen final-test encounters to select a configuration.

```text
python -m evaluation.evaluate --stage m6 --agent heuristic --heuristic-variant adaptive --seed-start 0 --episodes 10 --trace-dir runs/smoke/m6
python -m evaluation.paired runs/a/run_card.json runs/b/run_card.json
python -m evaluation.compare runs/a/run_card.json runs/b/run_card.json --output runs/comparison.json
python -m integration.mock_executor runs/mock-session
```

The last command runs the synthetic command handshake and needs a fresh, empty
directory; the [integration contract](docs/BG3_INTEGRATION.md#synthetic-cli-demonstration)
says what it shows and what it doesn't.

Training remains available through `python train.py --help`; value collection
and fitting through `python value_training.py --help`; exact planner evaluation
through `python -m evaluation.planning_eval --help`. These tools are maintained
for reproducibility and targeted research; new sweeps need an explicit question.
Policy vectors/action indices are simulator-specific and do not transfer to BG3.
