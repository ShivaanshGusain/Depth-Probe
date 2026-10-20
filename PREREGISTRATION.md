# Pre-registration (draft, fill in before confirmatory runs)

Fill this in **after** the GPT-2 small pilot and **before** any run on confirmatory models.
Commit it with a timestamp. Once committed, changes go in an amendment log below, never silently.

- Exploratory (pilot) models, excluded from confirmatory claims: GPT-2 small
- Confirmatory models: _list_
- Primary lens: _tuned / raw_
- Data, context length, stride, token count per model: _..._
- Token filter (`--min-context`), reversal tolerance (`--eps`), first state in E3 (`--from-layer`): _..._

| Sub-hypothesis | Metric (exact definition) | Counts as support | Counts as refutation |
|---|---|---|---|
| H1a convergence | tuned-lens mean KL(p_L ‖ p_l) per state, document-bootstrap 95% CI | _..._ | _..._ |
| H1b per-token monotonicity | fraction of monotone tokens vs random-init and shuffled controls | _..._ | _..._ |
| H1c noise recovery | _define once E6 exists_ | _..._ | _..._ |
| H1d noise ↔ depth | _define once E7 exists_ | _..._ | _..._ |
| H1e residual structure | _..._ | _..._ | _..._ |

Required number of confirmatory models that must meet each criterion: _..._

## Amendment log

| Date | Change | Reason |
|---|---|---|
