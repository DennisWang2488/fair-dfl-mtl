# Results revision plan — multi-instance redesign & editorial strategy

*Working handoff. Captures the discussion on (1) what the new multi-instance
healthcare (HC) results mean, (2) how an IJOC editor will read them, (3) the
precise HC completion checklist, and (4) the execution order. MD knapsack
redesign is deferred — see §6.*

*Date: 2026-06-06. Branch: `claude/experiment-results-narrative-FmU4B`.*

---

## 0. TL;DR — the single most important things

1. **The headline story improved.** Under the new multi-instance HC design,
   decision-focused learning (the FDFL family) genuinely sits at the regret
   floor (~16% below predict-then-optimize), with a clean "fairness almost for
   free" trade-off. This is a real, significant win — not a within-noise tie.
2. **The remaining work is defensive, not offensive.** The job is to make the
   HC result bulletproof and to stop over-claiming at a granularity the
   statistics don't support.
3. **Do HC first, MD last.** HC uses a closed-form gradient (fast); MD uses
   `cvxpylayers` (slow). Finish HC into an unassailable anchor, write the
   results section around it, and scope MD down afterward.
4. **The #1 editor comment will be about training budget / convergence** — which
   is exactly the "did we train it well enough?" worry. Answer it head-on with a
   convergence panel; it is cheap on HC.

---

## 1. Context — what changed

The HC experiment was redesigned from a **single-instance** setup (effectively
`N_train = 1`, the whole 48,784-patient cohort as one optimization instance) to
a **multi-instance** setup:

- One sample = one optimization instance (a cohort of `m` patients + its own
  budget). Train **one** predictor across `N_train` instances; evaluate on
  `N_test = 30` held-out instances.
- `m = 5000` (~600 minority), stratified bootstrap resample, patient-disjoint
  80/20 pool split stratified on race.
- `N_train ∈ {10, 20, 50}`, `α ∈ {0.5, 2.0}`, `λ ∈ {0,0.5,1,2}`,
  `μ ∈ {0.1,0.5,1}`, 3 seeds, MLP-64, budget `Q = 30%`, 70 SGD steps, no
  warm-start, no early stopping.

**This protocol change flipped the conclusion.** On the old single-instance /
MD-D3 tables, PCGrad Pareto-dominated. On the new multi-instance HC, the FDFL
family wins on regret and PCGrad/MGDA fall behind. This sensitivity to protocol
is both an opportunity (the multi-instance design is more realistic and
statistically sound) and a risk (see §3, item 1).

---

## 2. The corrected central story (what the new numbers support)

> On a realistic multi-instance fair-allocation task, decision-focused learning
> (the FDFL family) sits at the regret floor — ~16% below predict-then-optimize —
> and a light fairness regularizer buys a large prediction-fairness improvement
> for a negligible regret cost. The advantage appears exactly where theory
> predicts: high welfare concavity (α=2.0), finite predictor capacity, and
> adequate instance size. Dynamic MTL balancers split: Nash bargaining matches
> the floor; PCGrad/MGDA trade regret for raw fairness.

Key supporting numbers (α=2.0, N_train=50, MAD, best-λ per method):

| Claim | Evidence |
|---|---|
| FDFL family at the regret floor | FPLG 0.1284, DFL 0.1291, FDFL-Scal 0.1293, NashMTL 0.1293, FDFL 0.1298 — all within ~0.003 |
| ~16% below prediction baselines | FPTO 0.1530, PTO 0.1547, WDRO 0.1555 vs ~0.128 floor → ~16.1% |
| Fairness almost for free | DFL MAD 90.5 → FDFL(λ=1) MAD 26.7 = **71% cut** for **+0.5% regret** (0.1291→0.1298) |
| MOO splits | NashMTL 0.1293 (in band); PCGrad 0.1351, MGDA 0.1370 (~4–6% above) |
| PCGrad best raw fairness | PCGrad MAD 20.7 (lowest, tied with FDFL-Scal-μ0.1 at 20.6) |
| Stable across N_train | regret flat over N ∈ {10,20,50}; reproduces single-cohort 15.8% |
| No divergence | zero instance/seed blow-ups at m=5000 |

---

## 3. How an IJOC editor will read it — likely decision-letter comments (ranked)

1. **(Training budget — highest kill risk)** "The central comparison rests on a
   particular training protocol (70 SGD steps, no early stopping). The authors
   must demonstrate that all methods — particularly the predict-then-optimize
   baselines — are trained to convergence under an equal budget; otherwise the
   16% regret gap may reflect optimization, not the method." Note PTO's seed std
   (~0.015) is 3–4× FDFL's (~0.004), which invites exactly this question.
2. **(Protocol consistency)** "The two testbeds use different evaluation
   protocols and reach different conclusions (FDFL on HC, PCGrad on MD). The
   authors must reconcile this." → align MD to multi-instance, or justify the
   divergence explicitly.
3. **(Significance)** "Several headline differences are within one standard
   deviation at 3 seeds. Statistical significance and more seeds are required."
4. **(Regime of validity)** "The advantage narrows as predictor capacity grows
   (the capacity ablation shows PTO regret falling 0.165→0.154 while FDFL stays
   flat). The authors should clarify the regime in which the method is
   preferable." → this is the capacity-ablation double edge; preempt it.

Comments 1 and 3 both point at HC and are cheap to resolve (closed-form). Comment
2 (MD) is the most expensive but lowest priority. That ordering drives §5.

### Claim-granularity guardrails (avoid self-inflicted rejects)
- **Claim the family/regime, not a single method.** Within-FDFL-family
  differences are within noise at 3 seeds; do not crown FPLG (a hybrid built on
  someone else's PLG) as "the method."
- **Do not say "FDFL beats MOO."** NashMTL (a MOO method) is in the FDFL band;
  only PCGrad/MGDA lag. Correct line: "decision-focused and Nash-bargaining
  share the floor; PCGrad/MGDA trade regret for raw fairness."
- **Present α=0.5 honestly.** At α=0.5 PCGrad wins and vanilla FDFL is mid-pack;
  frame the advantage as scaling with welfare concavity, not as universal.
- **Frame the capacity ablation as characterization, not a selling point.**
  "Even at MLP-64 the gap is ~16%, and it widens under the finite capacity
  typical of real deployments" — not "a weaker predictor widens the gap!"
- **The old 'PTO has low MSE' pillar no longer holds** in the multi-instance
  table (PTO MSE 140 is beaten by scalarized/MOO at 124–128; pure DFL is worst
  at 347). Re-verify which MSE claim survives before building narrative on it.

---

## 4. HC completion checklist — original canonical grid vs new multi-instance run

### 4.1 Dimension-by-dimension gap table

| Dimension | Original single-instance table (current paper) | New multi-instance run | Gap |
|---|---|---|---|
| Fairness measure | 5: mad, dp, atkinson, bias_parity, wasserstein2_dp | **MAD only** | **missing 4** ← largest |
| α | 0.5, 2.0 | 0.5, 2.0 | ✓ |
| Seeds | 5 (11,22,33,44,55) | **3** | +2 needed |
| λ | {0, 0.5, 1, 2} | {0, 0.5, 1, 2} | ✓ |
| μ (pred-anchor) | {0.01, 0.1, 0.5, 1, 2} | {0.1, 0.5, 1} | missing μ=0.01, μ=2 |
| Method pool | 11 main + per-fairness extras | 11 main (incl. PTO, DFL) | see 4.2 |
| N_train | none (single instance) | {10,20,50} (new axis) | new — an improvement |
| Training budget | 70 steps, no early-stop, MLP-64 | same | both lack convergence evidence |
| Packaged in repo | `run_hc_group.py` present | multi-instance driver **not in repo** | reproducibility gap |

### 4.2 Method gaps (vs original pool)
Original main pool (11): `FPTO, SAA, WDRO, FDFL, FDFL-0.1, FDFL-0.5, FDFL-Scal,
FPLG, PCGrad, MGDA, NashMTL`; plus per-fairness extras — mad: `CAGrad,
FDFL-Scal-mu2, FDFL-Scal-mu0.01, PLG-kappa1`; dp: `CAGrad, FDFL-Scal-mu2`;
wasserstein2_dp: `CAGrad, FDFL-Scal-mu2, PTO`; DFL = `fdfl@λ=0` alias.

Missing from the new run:
1. **CAGrad** — pending (mad/dp/w2-dp in original).
2. **FDFL-Scal-μ2** — pending (mad/dp/w2-dp in original).
3. **FDFL-Scal-μ0.01** — missing (mad only in original).
4. **PLG-kappa1** — missing (mad only in original).

(PTO and DFL are already present in the new run — cleaner than the original,
where PTO appeared only as a w2-dp extra.)

### 4.3 What HC must add to be submission-ready
- **(A) Methods:** add CAGrad, FDFL-Scal-μ2 (already planned). See §5 for the
  recommendation to *drop* μ0.01 and PLG-kappa1 rather than restore them.
- **(B) Fairness measures:** decide scope — see §5, item 1 (the dominant
  compute lever).
- **(C) Seeds:** 3 → 5 minimum.
- **(D) Convergence/training-budget evidence:** none today — add it (§5, item 4).
- **(E) Statistical testing:** aggregator outputs mean±std only — add paired
  significance (§5, item 5).
- **(F) Packaging:** add the multi-instance driver + aggregator to `experiments/`
  and `aggregate/`.
- **(G) Consistency fixes:** see §5, item 7.

---

## 5. Revision recommendations (opinionated — not "restore everything")

1. **Fairness measures: do NOT redo all 5. Cut to 2 — MAD (primary) + one
   standard measure (DP or W2-DP).** The fairness *measure* is not the
   contribution; the framework is. Five measures × multi-instance × 5 seeds × N
   grid is heavy compute for marginal reviewer value, but a single measure
   invites "is this MAD-specific?" Two is the sweet spot: MAD headline, the
   second as a robustness appendix. **Recommended second measure: W2-DP or DP**
   (most method separation in the original; the most expected fairness notions).
   Demote atkinson/bias_parity to an appendix mention or drop.
   **→ This is the one decision that's yours; it sets the rerun size.**
2. **Methods: add CAGrad + FDFL-Scal-μ2; drop μ0.01 and PLG-kappa1.**
   μ ∈ {0.1,0.5,1,2} already shows the prediction-anchor trend; μ0.01 is
   redundant with μ0.1 and PLG-kappa1 is low-value clutter. State that they were
   omitted as redundant rather than chasing exact parity.
3. **Seeds: go to 5.** Multi-instance already averages over N_test=30, so
   per-seed variance is low — 5 seeds give tight CIs. Cheap (closed-form).
4. **Add a convergence / training-budget panel — highest value-per-hour.**
   Reuse the existing **Variant B** config (`hc_v2_train_cfg_b`: 150 steps + LR
   decay + early stopping). For representative methods (PTO, FDFL, PCGrad,
   NashMTL) show that the ranking and the ~16% gap are unchanged under (a) 70
   fixed steps and (b) 150 steps + early stopping. This one comparison kills the
   "under-trained" attack for all methods at once and settles the internal
   worry.
5. **Add paired significance (Wilcoxon / CI) to the aggregator.** Then state in
   prose: within-FDFL-family differences are NOT significant (claim the family);
   FDFL-family vs PTO IS significant.
6. **Package the multi-instance driver + aggregator into the repo** (parameterize
   Run-A and the ablations) for IJOC reproducibility.
7. **Fix two consistency hazards a referee will catch:**
   - `configs.make_task_cfg` defaults `budget_rho=0.35`, but README / hc_v2 /
     Run-A all use **0.30** — stale default; fix it.
   - Ablations use budget 0.25 / m=1000 / MLP-32; headline uses 0.30 / m=5000 /
     MLP-64. Numbers are not comparable across tables (headline FDFL 0.1298 vs
     ablation 0.1302). Label "headline anchor" vs "diagnostic ablation" operating
     points explicitly in the text.

---

## 6. MD knapsack — deferred, scoped down

MD has no closed form (uses `cvxpylayers`) and is slow, so it is the long pole.
**Do not block the paper on it.**

- Multi-instance for MD does **not** need m=5000 — it is synthetic and you
  control m; use a few hundred per instance.
- Cut the MD method grid to the essential contrast (PTO / FDFL / one MOO).
- Vectorize / GPU / cache the decision layer where possible.
- Fallback: keep MD single-instance but **reframe it explicitly** as a
  controlled-curvature stress test in a different regime, stating upfront it uses
  a different protocol — turning the inconsistency into an intentional design.
  (Weaker than aligning protocols; use only if time-constrained.)
- MD's role is the synthetic complement to the HC headline; IJOC does not require
  two real datasets if one is thorough and the synthetic is well-motivated.

---

## 7. Execution order (by editor-impact per hour; all HC steps are closed-form)

1. **Convergence panel (Variant B comparison)** — cheapest, kills the #1 editor
   comment, answers the "did we train it well enough?" worry. Do first.
2. **Complete the MAD column** — add CAGrad + FDFL-Scal-μ2; seeds → 5.
3. **Add significance + rewrite the results section anchored on MAD.**
4. **Second fairness measure (W2-DP/DP)** as a robustness appendix.
5. **Package the multi-instance driver + aggregator into the repo.**
6. **MD multi-instance** — scoped down per §6; last.

---

## 8. Open decision for you

**Fairness-measure scope for the multi-instance rerun:** MAD + W2-DP (recommended
default), MAD + DP, MAD only, or keep all 5? This sets the rerun size before any
config is written.

---

## Appendix — reference pointers
- Canonical method registry: `fdfl_harness/configs.py` (`ALL_METHOD_CONFIGS`).
- Original HC grid: `experiments/run_hc_group.py` (`FULL_POOL`,
  `METHODS_BY_FAIRNESS`) and `fdfl_harness/hc_v2.py` (seeds, steps, budget,
  Variant A/B).
- Original single-cohort numbers: recovered `results_reference/healthcare/
  SUMMARY.md` (git commit `4a84bcb`).
- New multi-instance numbers: `STATUS_REPORT.md` (Run-A pilot + §2.4 ablations);
  source `results/healthcare/main_v5_multiinstance/`.
- Method definitions / pseudocode: `METHODS.md`.
