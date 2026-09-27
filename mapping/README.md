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
- **Still open:** O\*NET's 41 GWAs include three that are not assigned here,
  so step 02 will warn and drop them:
  `Judging the Qualities of Objects, Services, or People`,
  `Staffing Organizational Units`,
  `Monitoring and Controlling Resources`.
  Assign each to a part (and justify it in Chapter 3) so the composition is
  exhaustive before final estimation.
