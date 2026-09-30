# O\*NET Work Activity → task-part mapping

`onet_activity_map.csv` assigns each O\*NET Generalized Work Activity (GWA) to
one of the four parts of the task composition used in step 02:

| Part | Meaning |
|---|---|
| `c1_nonroutine_analytic` | non-routine analytic |
| `c2_nonroutine_interpersonal` | non-routine interpersonal |
| `c3_routine_cognitive` | routine cognitive |
| `c4_residual` | everything else (manual, physical, equipment) |

This file is a **methodological choice you must be able to defend**. It is
kept under version control, so every change is dated in the git history.
`src/02_build_task_composition.py` only reads it; it never regenerates or
overwrites it. Change it here, commit with a message saying why, and re-run.

Step 02 **stops** if any name here is missing from the O\*NET release, and
**warns** about O\*NET activities that are not mapped (these are dropped).

## Change log

- Removed `Coding/Encoding Information` (not an O\*NET GWA) and renamed
  `Communicating with Persons Outside Your Organization` →
  `Communicating with People Outside the Organization` (the official GWA
  name). These two entries stopped step 02 against release 30.3.
- **Three remaining GWAs assigned** (O\*NET 30.3 has 41 GWAs; all 41 are now
  mapped, so the composition is exhaustive). Each follows the rule the rest
  of the file already uses:
  - `Judging the Qualities of Objects, Services, or People` →
    `c1_nonroutine_analytic`. O\*NET places it among the *Mental Processes*
    (with Analyzing Data, Making Decisions, Evaluating Compliance), which this
    file assigns to c1; it is evaluative judgement, not a codified procedure.
  - `Staffing Organizational Units` → `c2_nonroutine_interpersonal`.
    Recruiting, interviewing and hiring are face-to-face judgements about
    people, like Coaching and Developing Others and Developing and Building
    Teams (c2).
  - `Monitoring and Controlling Resources` → `c3_routine_cognitive`. O\*NET
    groups it with *Administering* (monitoring budgets and spending) alongside
    Performing Administrative Activities, which this file assigns to c3.
  These are defensible defaults, not the only reasonable choices: state them
  in Chapter 3, and report the sensitivity check in which the three are
  dropped (the previous mapping, commit `c8958f4`).

## Alternative mapping (robustness)

`onet_activity_map_alt.csv` differs from the main file in three GWAs, following
the audit of 30 September 2026:

- `Interpreting the Meaning of Information for Others` → `c1_nonroutine_analytic`
  (Acemoglu & Autor 2011 count it as non-routine analytic).
- `Communicating with Supervisors, Peers, or Subordinates` and
  `Communicating with People Outside the Organization` →
  `c2_nonroutine_interpersonal` (communication is interpersonal work, not a
  codified routine).

The pipeline runs it as the `results_mapping_alt` specification. Neither file
is anchored GWA by GWA in a published scheme; citing one for each assignment
(or justifying the departures) is still to be done in Chapter 3.
