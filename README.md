# Memory Reward Inflation in Self-Improving LLM Agents

> **Code release notice.** This repository currently contains only this README.
> The source code for the paper will be released **after the paper is published**.
> Nothing else is available here yet.

## About the paper

Self-improving LLM agents learn from experience without updating any weights: each
episode is stored in an external memory, scored, and retrieved for similar future
tasks. When ground-truth labels are unavailable, which is the normal situation in
deployment, the stored score is an LLM assessment rather than a verified reward.

The paper studies what goes wrong when that score is self-graded:

- **The Echo Gap.** Incorrect episodes receive inflated rewards, and the agent
  preferentially reuses the mistakes it is most confident about. The error compounds
  through memory retrieval instead of averaging out.
- **The Error-Independence Criterion (EIC).** A signal can only correct the inflation
  if it both tracks truth and has errors that are decorrelated from the memory's
  self-grade bias. A more capable parametric judge, on its own, does not provide this.
- **LUCID.** An answer-free de-inflation algorithm that demotes the reward of
  memories flagged by an independent, label-free signal. On the BIRD text-to-SQL
  benchmark it reaches 57.6% execution accuracy, compared with 53.7% for a
  Memento-style self-graded agent and 52.4% for a memory-less agent.

## Authors

Mohammad Asadolahi, Amir Amini, Samira Talebi, Amirfarhad Farhadi, and Azadeh Zamanifar.

## Citation

Citation details will be added here once the paper is published.

## Status

| Item | Status |
| --- | --- |
| Paper | Forthcoming |
| Source code | To be released after publication |

Please watch or star this repository to be notified when the code is released.
