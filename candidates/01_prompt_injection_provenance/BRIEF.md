# Prompt Injection: Provenance vs Role Confusion

## Research question

When prompt injection succeeds, does the model lose track of the text's true formal source, or does it retain provenance information while behaving according to an inferred functional role?

## Core hypotheses

1. Provenance is genuinely degraded during successful injection.
2. Provenance remains decodable, but a competing functional-role / authority signal wins.
3. Both remain represented, and the failure occurs further downstream.

## Tournament goal

Find the cheapest experiment that distinguishes these explanations enough to decide whether this project deserves the remaining research budget.

## Constraint

Keep experiments simple enough to understand end-to-end. Do not build a large prompt-injection evaluation or probing framework during the tournament stage.
