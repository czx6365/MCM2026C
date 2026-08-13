# MCM 2026 Problem C — Modeling Voting and Elimination in *Dancing with the Stars*

**2026 Mathematical Contest in Modeling (MCM) — Meritorious Winner**

This project studies how judging scores, hidden fan preferences, voting rules, professional dancers, celebrity characteristics, and elimination design interact in *Dancing with the Stars* (DWTS). The analysis is organized around the four questions of MCM 2026 Problem C.

Rather than treating the competition as a simple prediction task, we modeled it as a sequence of **inverse inference, counterfactual simulation, statistical effect estimation, and mechanism design** problems.

## Research Overview

```text
Historical judge scores + observed eliminations
                    ↓
Q1  Infer latent fan-vote shares and uncertainty
                    ↓
Q2  Replay alternative voting rules and Judges Save
                    ↓
Q3  Estimate effects of professional dancers and celebrity traits
                    ↓
Q4  Design and compare alternative elimination mechanisms
```

The dataset covers 34 seasons of DWTS. After preprocessing, the main modeling table contains season-week-contestant observations, observed judge scores, active-roster information, and elimination events.

---

# Q1 — Inferring Hidden Fan Votes

## Problem

Fan votes are not publicly released. We therefore need to infer a weekly fan-vote distribution that is consistent with the observed judge scores, the historical voting rule, and the contestants who were actually eliminated.

For contestant $i$ in week $t$, let

```math
p_{i,t} \ge 0, \qquad \sum_i p_{i,t}=1
```

be the unknown fan-vote share.

The key difficulty is that the data provide an **elimination outcome**, not a direct numerical fan-vote label. This makes Q1 an inverse problem with a large set of possible latent vote distributions.

## Model 1: Historical Voting-Rule Simulator

For each season-week, the inferred fan shares are passed through the corresponding historical elimination mechanism.

### Percent rule

Judge scores and fan votes are converted to shares:

```math
C_i = \frac{J_i}{\sum_j J_j} + p_i
```

Contestants with the smallest combined scores are eliminated.

### Rank rule

Judge scores and fan votes are separately ranked. Let $r_J(i)$ denote judge rank and $r_F(i)$ denote fan-vote rank:

```math
R_i = r_J(i) + r_F(i)
```

Contestants with the largest rank sums are eliminated.

This simulator converts any candidate fan-vote vector into a predicted elimination set.

## Model 2: Dynamic Dirichlet Prior

We model the weekly fan-vote vector with a Dirichlet distribution. For the first modeled week of a season, a symmetric prior is used. For later weeks, the previous week's inferred vote shares provide a dynamic prior. Using $D(\alpha_t)$ to denote a Dirichlet distribution with concentration vector $\alpha_t$:

```math
p_t \sim D(\alpha_t)
```

The concentration parameter controls how strongly fan popularity is assumed to persist from one week to the next. This gives the inference process temporal continuity without forcing fan preferences to remain fixed.

## Model 3: Hard Approximate Bayesian Computation (ABC)

Because the elimination rule is discrete and non-differentiable, we use **ABC rejection sampling** rather than a closed-form likelihood.

```text
sample fan-vote vector from Dirichlet prior
        ↓
apply historical Rank / Percent rule
        ↓
compare predicted elimination with observed elimination
        ↓
accept only if the elimination set matches exactly
```

Accepted samples form an approximate posterior over weekly fan-vote shares. From these samples we compute posterior mean `p_mean`, a 90% uncertainty interval, posterior entropy, and an accepted-sample MAP estimate `p_map`.

`p_map` is selected from the accepted region, so it remains consistent with the observed elimination constraint.

## Model 4: Soft ABC and Posterior-Predictive Consistency

Exact rejection can be inefficient when the feasible region is narrow. We therefore also define a **soft distance** between predicted and observed elimination structures.

For the Percent rule, the distance measures how strongly an eliminated contestant outranks a survivor in combined percentage score. For the Rank rule, an analogous rank-gap violation is used.

A Gaussian-style kernel converts the distance into a weight:

```math
w \propto \exp\left[-\left(\frac{d}{\epsilon}\right)^2\right]
```

This is used to evaluate posterior-predictive consistency and quantify how strongly the inferred fan distribution supports the observed outcome.

## Robustness and Sensitivity

The Q1 pipeline also varies the Dirichlet initialization and temporal-smoothing strength. The main conclusion is stable across tested settings: feasible fan-vote distributions can be recovered, while stronger temporal concentration narrows uncertainty intervals.

### Main insight

Q1 produces a **distribution over plausible fan preferences**, rather than pretending that a single hidden vote vector can be uniquely recovered from elimination data.

---

# Q2 — Comparing Voting Rules and Judges Save

Q2 uses the inferred fan-vote distributions from Q1 to study whether different voting mechanisms systematically favor judges or fans.

## Q2.1 — Rank vs. Percent

### Rank aggregation

```math
R_i = r_i^{(J)} + r_i^{(F)}
```

Only ordering information is preserved. A large difference in fan support may collapse to a one-rank difference.

### Percent aggregation

```math
C_i = s_i^{(J)} + s_i^{(F)}
```

Here the magnitude of support is retained: a contestant with substantially more fan support receives a proportionally larger advantage.

## Comparison Strategy

For every elimination week we compute both counterfactual elimination sets and identify **disagreement weeks** where Rank and Percent would eliminate different contestants.

Let $\bar p_R$ be the mean inferred fan share of contestants eliminated by Rank, and $\bar p_P$ the corresponding quantity for Percent. We compare:

```math
\Delta_F = \bar p_R - \bar p_P
```

A positive value means Percent eliminates contestants with lower fan support and therefore preserves the more popular contestants.

### Key finding

Across all comparable weeks the difference is modest, because both rules usually eliminate the same contestant. However, among the **17 weeks where the methods disagree**, Percent eliminates the lower-fan-share contestant in **100% of cases**.

This suggests that Percent aggregation matters most near close decision boundaries because it preserves the magnitude of fan preference rather than only its rank.

## Q2.2 — Monte Carlo Counterfactual Replay

We then study controversial contestants including Jerry Rice, Billy Ray Cyrus, Bristol Palin, and Bobby Bones.

Fan-vote uncertainty from Q1 is propagated through the simulation instead of using only one point estimate. Let $T(p_L,p_M,p_U)$ denote a triangular distribution with lower bound, mode, and upper bound from the Q1 interval:

```math
p_i^{(m)} \sim T(p_L,p_M,p_U)
```

Each simulated season is replayed under four scenarios:

```text
Percent
Rank
Percent + Judges Save
Rank + Judges Save
```

The historical number of eliminations in each week is preserved. Repeating the replay produces a distribution over final placements rather than a single deterministic counterfactual.

## Judges Save Model

When Judges Save is active, the combined rule determines the bottom two and the judges determine which contestant leaves. The deterministic version eliminates the bottom-two contestant with the lower judge score; the implementation also supports a probabilistic logistic alternative.

### Main insight

The counterfactual distributions show that voting-system design can materially change controversial outcomes. Judges Save tends to reduce the advantage of contestants whose survival is driven primarily by strong fan support despite relatively weak judge scores.

---

# Q3 — Professional Dancers, Celebrity Traits, and Competition Performance

## Problem

A contestant can be popular with fans but weak with judges, or vice versa. We therefore estimate the two mechanisms separately instead of fitting one pooled outcome model.

The main explanatory variables include professional dancer, celebrity age, industry, home country/region, season effects, and week effects.

## Model 1: Judge-Score Regression

For active contestants, standardized judge scores are modeled using OLS. Using compact notation for season, week, age, industry, country, and professional-dancer effects:

```math
Y_J = \beta_0 + \beta_S + \beta_W + \beta_A + \beta_I + \beta_C + \beta_P + \varepsilon
```

Professional dancers enter as fixed effects. Standard errors are clustered by contestant-season pair to account for repeated weekly observations from the same partnership.

## Model 2: Uncertainty-Weighted Fan Regression

Fan-vote shares are first mapped to logit space and standardized. Because Q1 estimates have different uncertainty levels, the fan model uses **Weighted Least Squares (WLS)** with approximately inverse-variance weights.

Let $c_{i,t}$ denote the confidence-interval width of the inferred fan vote for contestant $i$ in week $t$. Then:

```math
w_{i,t} \propto c_{i,t}^{-2}
```

Thus, highly uncertain fan-vote estimates contribute less to coefficient estimation. The same explanatory variables are used as in the judge model, again with pair-level cluster-robust standard errors.

## Model 3: Incremental R²

To measure the contribution of professional dancers, we compare nested models with and without professional-dancer effects.

- Judges: $\Delta R^2 = 0.0179$
- Fans: $\Delta R^2 = 0.0445$

Professional dancers therefore explain substantially more additional variation in fan response than in judge response.

The estimated professional-dancer effects on judges and fans are also almost uncorrelated. Let $\rho$ denote the correlation coefficient:

```math
\rho\left(\widehat{\beta}_P^{(J)},\widehat{\beta}_P^{(F)}\right) \approx 0.0328
```

## Model 4: How Far Does a Contestant Survive?

At the contestant-season level, the response variable is `weeks_survived`. The model uses mean judge performance, mean inferred fan support, celebrity characteristics, and professional-dancer effects.

Nested models are compared through incremental $R^2$ to distinguish the contribution of performance, celebrity traits, and professional partners to competition longevity.

### Main insight

The factors that influence judges are not necessarily the factors that influence fans. Modeling these two audiences separately reveals a professional-dancer effect that would be obscured by a single combined outcome model.

---

# Q4 — Designing a Better Elimination Mechanism

Q4 treats the competition rules themselves as a mechanism-design problem. Four elimination methods are replayed on historical weeks and evaluated against observed eliminations and high-disagreement cases.

## Method 1: Rank

```math
R_i = r_i^{(J)} + r_i^{(F)}
```

It is simple and robust to scale differences, but discards information about how large score or vote gaps actually are.

## Method 2: Percent

```math
C_i = s_i^{(J)} + s_i^{(F)}
```

This preserves the magnitude of support and provides the strongest overall historical replay match among the four tested methods.

## Method 3: Dynamic Weighting + Judges Save

Judge-score dispersion is summarized with a coefficient-of-variation-style quantity, while fan concentration is summarized with the Herfindahl-Hirschman Index (HHI). These statistics determine a dynamic fan weight:

```math
S_i = w_J s_i^{(J)} + w_F s_i^{(F)}, \qquad w_J + w_F = 1
```

When there is a single elimination, the combined score selects the bottom two and Judges Save eliminates the contestant with the lower judge score.

The idea is to adapt the voting balance to the competitive structure of each week rather than imposing a constant 50/50 rule.

## Method 4: Uncertainty-Aware Geometric Fusion

Fan confidence is derived from Q1 relative interval width:

```math
c_F = \frac{1}{1+u_F}
```

Judge confidence is derived from judge-score dispersion:

```math
c_J = \frac{d_J}{d_J+\tau}
```

The judge weight becomes

```math
\alpha_t = \frac{c_J}{c_J+c_F}
```

and the final score uses geometric fusion:

```math
G_i = \left(s_i^{(J)}\right)^{\alpha_t}\left(s_i^{(F)}\right)^{1-\alpha_t}
```

This multiplicative form penalizes contestants who are weak on one dimension rather than allowing an extreme value on one side to completely compensate for the other.

For single-elimination weeks, the method also applies a protection step within the lowest-ranked candidates before selecting the final loser.

## Evaluation

| Method | Historical Match Rate | Extreme-Disagreement Hit Rate |
| --- | ---: | ---: |
| Rank | 56.38% | 2.99% |
| **Percent** | **76.60%** | 19.40% |
| Dynamic + Judges Save | 54.26% | 14.93% |
| **Uncertainty + Geometric** | 68.09% | **20.90%** |

The evaluation contains 188 elimination weeks, including 67 weeks identified as extreme judge-fan disagreement cases.

### Main insight

No single rule dominates every objective:

- **Percent** gives the best overall replay consistency;
- **Uncertainty + Geometric** performs best on extreme judge-fan disagreement weeks.

This motivates a broader conclusion: an elimination system should be evaluated not only by overall accuracy, but also by how it behaves when judges and fans strongly disagree.

---

# Main Repository Modules

| Module | Purpose |
| --- | --- |
| [`data/`](data) | DWTS data preprocessing and season-week modeling tables |
| [`t1/`](t1) | ABC-based fan-vote inference and uncertainty analysis |
| [`t2_1/`](t2_1) | Rank vs. Percent comparison and fan-preference analysis |
| [`t2_2/`](t2_2) | Monte Carlo counterfactual replay of controversial contestants |
| [`t3/`](t3) | Judge/fan regression, professional-dancer effects, and survival analysis |
| [`t4/`](t4) | Alternative elimination mechanism design and replay evaluation |
| [`人气/`](人气) | Exploratory external-popularity proxy based on Wikipedia pageviews |

---

# Team Contributions

### Zixi Chen

- Led modeling across **all four questions (Q1–Q4)**.
- Implemented the project's computational models, simulations, statistical analyses, and supporting code.
- Reviewed the full manuscript, including the mathematical formulation, experimental consistency, and final presentation of the modeling results.

### Jiayi Chen

- Co-developed the modeling approach across **all four questions (Q1–Q4)**.
- Produced **all figures and visualizations** used in the paper.
- Wrote the initial manuscript draft and participated in the full-paper revision process.

### Muyang Li

- Co-developed the modeling approach across **all four questions (Q1–Q4)**.
- Contributed to the initial manuscript draft.
- Participated in manuscript revision and refinement.

---

# Award

**2026 Mathematical Contest in Modeling (MCM) — Meritorious Winner**

The project demonstrates a complete modeling workflow spanning Bayesian inverse inference, Monte Carlo simulation, uncertainty quantification, regression with robust inference, counterfactual analysis, and voting-mechanism design.
