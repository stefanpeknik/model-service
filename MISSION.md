# Mission: Know model-service as if I had written it

## Why
I helped make the high-level decisions for the expanded design. I want an author's understanding of the implementation, including the pathology context behind its inputs and outputs, so I can reason about the code without relying on architectural summaries.

## Success looks like
- Trace a public request through every relevant function, adapter, transaction, and model call.
- Explain data structures, state changes, concurrency, failure handling, and cleanup from the source.
- Connect pathology concepts to tile selection, physical resolution, predictions, and artifacts.
- Predict edge cases, locate the tests that establish behavior, and explain a change's consequences.
- Navigate every current code and configuration area, and recognize historical prototypes.

## Constraints
- General repository purpose and high-level design are already familiar.
- Pathology context and the expanded implementation need deliberate study.
- Use short lessons with real source reading, prediction exercises, and follow-up discussion.
- Anchor implementation claims to the current working tree and requirements to the primary specification.

## Out of scope
No codebase area has been excluded. Biological and clinical detail should follow the needs of the implementation.
