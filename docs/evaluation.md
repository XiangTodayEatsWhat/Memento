# Evaluation protocol

Run inference over every frame of every selected evaluation example. Different examples may reference the same video with different queries. Do not collapse them by UID or truncate long videos.

## Temporal metrics

For each reference assistant turn, a temporal hit occurs when at least one predicted assistant turn satisfies abs(predicted_time-reference_time) < 5 seconds. The exact boundary of 5 seconds is excluded. TimeRecall is hits divided by reference turns. Spatial and temporal subsets are identified by the task name. The long subset uses absolute video timestamps greater than 1,500 seconds.

Redundancy is the proportion of predicted assistant turns that lie outside every reference window in the same example. If there are no predicted responses, Redundancy is null and TimeRecall is zero. Empty predictions remain in evaluation. 

The response threshold starts at 0.5, increases by 0.1 after each 10 consecutive response frames, and resets after 30 consecutive silent frames. Inference consumes questions at their timestamps, retains dialogue context and updates memory from incoming frames.

## Quality judge

For each reference with temporally matched predictions, the judge receives its task focus, active question, reference answer and all candidate responses in that window. It assigns 1–10 to candidate answers and returns the maximum as Overall Score. The task-specific prompt is benchmark/judge_prompt.json. The requested and returned model must both be gpt-3.5-turbo-0125.

The implementation reports two explicit aggregates:

- score_matched: mean judge score over references that have candidate responses.
- score_all_zero_for_misses: mean over all reference turns, assigning zero to references with no temporally matched prediction.

The two aggregates use different denominators. Unmatched reference turns contribute zero only to `score_all_zero_for_misses`; failed judge requests must be completed before aggregation.

Configure the judge API as described in the [evaluation guide](getting_started.md#7-answer-quality-evaluation). Comparisons should use the same test revision, judge model and aggregation convention.
