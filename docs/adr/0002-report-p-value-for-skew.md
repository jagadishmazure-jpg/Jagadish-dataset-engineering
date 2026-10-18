# ADR 0002: Report a p-value with the labeler-skew statistic

Status: accepted (2026-10-18)

## Context
Skew is flagged when a chi-square statistic is above a fixed threshold (10.0).
The same statistic means different things with 2 label options and with 6,
because the degrees of freedom differ.

## Decision
Keep the threshold flag for backward compatibility. Add `df` and a `p_value`,
computed in pure Python from the regularized incomplete gamma function and
tested against standard table values, so the rule can be set per label set.

## Consequences
- The demo shows the gap: one labeler has p = 0.009 and is not flagged at the
  fixed threshold.
- A later change can switch the flag to a p-value rule with a minimum label
  count. That would change behaviour, so it is a separate decision.
