# Teaching notes

## Preferences and established background

The user wants familiarity with all implementation details as if they had written the code. They spent substantial time deciding the latest design at a high level. General purpose is familiar; pathology context and the expanded implementation are less familiar. Python, async, Ray, and PostgreSQL depth have not yet been established.

Teach through actual functions, values, call order, branch conditions, tests, and failure cases. Explain the domain where it changes code behavior. A package overview alone does not meet the mission.

The user also wants source navigation supported by docstrings on modules, classes,
and every function, including private helpers. Use Google-style `Args`, `Returns`
or `Yields`, and `Raises` where applicable, with field meanings on data classes.
Document actual error conditions and optional-result meanings, not generic filler.

## Session state

- 2026-10-06: Mission established from the user's clarification.
- First lesson prepared: `lessons/0001-schedule-admission.html`.
- User has not yet answered its exercises. Coverage is not evidence of understanding.
- 2026-10-07: The user requested thorough diagrams of the real immediate and scheduled tests, including transports, deployment containers, and limitations. Added `reference/test-request-flow.html`, reusable SVG diagrams, and a full-flow appendix in lesson 0001. The deployment account follows the final October 6 record; source/configuration was checked at HEAD `35ad233`. No live validation was rerun for the diagrams.
- 2026-10-07: The user accepted keeping the scheduler outside Ray Serve and requested documenting the rationale. Recorded `docs/adr/0001-run-scheduler-outside-ray-serve.md` and linked it from the flow reference. The decision distinguishes independent API/maintenance availability from PostgreSQL-backed durability, and separates scheduler placement from immediate HTTP routing. No exercise response or demonstrated mastery was recorded.
- Next conversation: ask for the final race explanation, then follow the admitted record into `claim_due`. If prerequisite gaps appear, address them using the same code path.
- Begin future sessions with a brief recall question from earlier lessons. Revisit race and deadline reasoning after intervening topics.

## Course maintenance

Use `COVERAGE.md` to track remaining implementation areas. Split each area into several short lessons as needed. Add learning records only for demonstrated understanding, disclosed background, corrected misconceptions, or mission changes.

The first lesson was checked against working-tree source at HEAD `ed9b7ac`. There is unrelated work in this repository. Re-read relevant source when preparing later lessons. The specification and progress live outside this clone; `docs/README.md` routes to them. Historical live results and later local corrections have different validation boundaries.
