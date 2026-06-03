# Methods Reference — IJOC Reproduction Package

This document catalogs every method evaluated in the paper, mapping
**paper label ↔ code symbol ↔ pseudocode ↔ original publication**.

It is the canonical reference for what each method does at the code level.
Each entry includes:

1. **Bucket & role** in the framework.
2. **MethodSpec flags** (`use_dec`, `use_pred`, `use_fair`, `pred_weight_mode`,
   `gradient_merge`, `mo_method`) as set by
   [`fdfl_harness/configs.py`](fdfl_harness/configs.py).
3. **Code entry point** — the function or class that implements the update.
4. **LaTeX pseudocode** (one-step per-batch view; train_single_stage in
   [`fair_dfl/training/loop.py`](../wheels/) wraps it with the lambda schedule).
5. **Original reference**.

Throughout: $f_\theta(x)$ is the predictor, $\hat r = f_\theta(x)$ the predicted
reward, $r$ the true reward, $d^\star(\hat r)$ the decision-task solution at
prediction $\hat r$, $W_\alpha$ the $\alpha$-fair welfare, $\ell_{\mathrm{pred}}$
the MSE prediction loss, and $\ell_{\mathrm{fair}}$ the prediction-fairness
penalty (e.g., MAD). All gradients are with respect to predictor parameters
$\theta$ unless stated otherwise.

---

## Overview table

| Paper label        | Bucket             | Code key (`method`) | Inline flags / `mo_method`             | Reference                |
|--------------------|--------------------|---------------------|----------------------------------------|--------------------------|
| PTO                | No-fairness        | `fpto` (λ=0)        | `use_pred=T`                           | Elmachtoub & Grigas 2022 |
| FPTO               | Prediction-focused | `fpto`              | `use_pred=T, use_fair=T`               | Berk et al. 2017         |
| SAA                | Robust baseline    | `saa`               | constant predictor (mean of y)         | Shapiro et al. 2014      |
| WDRO               | Robust baseline    | `wdro`              | `use_pred=T` + Wasserstein gradient pen. | Gao & Kleywegt 2023      |
| DFL                | Decision-only      | `fdfl` (λ=0)        | `use_dec=T`                            | Elmachtoub & Grigas 2022 |
| FDFL               | End-to-end (focal) | `fdfl`              | `use_dec=T, use_fair=T`                | this paper               |
| FDFL-Scal(λ,μ)     | Integrated · static scalarized | `fdfl_scal` | `use_dec=T, use_pred=T, use_fair=T, gradient_merge="raw"` | this paper |
| FDFL-PLG (was FPLG-κ) | Integrated · hybrid | `fplg`           | `use_dec=T, use_pred=T, use_fair=T, gradient_merge="guided"` | Jeon et al. 2025 (PLG) + this paper (fair ext.) |
| FDFL-PCGrad        | Integrated · dynamic | `pcgrad`          | `mo_method="pcgrad"`                   | Yu et al. NeurIPS 2020   |
| FDFL-MGDA          | Integrated · dynamic | `mgda`            | `mo_method="mgda"`                     | Sener & Koltun NeurIPS 2018 |
| FDFL-NashMTL       | Integrated · dynamic | `nashmtl`         | `mo_method="nashmtl"`                  | Navon et al. NeurIPS 2022 |
| FDFL-CAGrad        | Integrated · dynamic | `cagrad`          | `mo_method="cagrad"`                   | Liu et al. ICLR 2021     |

> **Naming taxonomy (advisor writing-plan, 2026-05-28; provisional pending June 4):**
> *Separate* = SAA, PTO, FPTO, WDRO. *Integrated · static scalarized* = FDFL,
> FDFL-Scal(λ,μ). *Integrated · hybrid* = FDFL-PLG. *Integrated · dynamic* =
> FDFL-PCGrad/MGDA/NashMTL/CAGrad. The MOO handlers are re-prefixed `FDFL-` to
> mark that they instantiate the same DF-MTFL framework. **"Prediction anchor"
> is retired** as a term: FDFL-Scal's μ is the **prediction-loss weight**; the
> prediction-loss *guidance* mechanism is what FDFL-PLG does.

All MOO handlers reside in
[`fair_dfl/algorithms/mo_handler.py`](../wheels/) (bundled wheel).
All scalarized methods route through
`fair_dfl/training/loop.py::_combine_prediction_gradients`.

---

## 1. PTO — Predict-Then-Optimize (no fairness)

**Code:** `method="fpto"` with `lambda_value=0`. Sets `use_pred=True`,
`pred_weight_mode="fixed1"`, `use_fair=False`.

**Description.** Train $f_\theta$ by minimizing prediction MSE only; at
deployment, plug $\hat r$ into the welfare-optimal decision. Classical
two-stage baseline.

**Pseudocode (LaTeX).**

```latex
\begin{algorithm}[H]
\caption{PTO}
\KwIn{data $\{(x_i, r_i)\}$, predictor $f_\theta$}
\For{$t=1,\dots,T$}{
  $g_\mathrm{pred} \gets \nabla_\theta \tfrac{1}{B}\sum_{i\in B_t} (f_\theta(x_i) - r_i)^2$\;
  $\theta \gets \theta - \eta_t\, g_\mathrm{pred}$\;
}
\KwOut{$\theta$, decisions $d^\star(f_\theta(x_\mathrm{test}))$}
\end{algorithm}
```

**Reference.** Elmachtoub & Grigas, *Smart "Predict, then Optimize"*,
Management Science 68(1), 9–26, 2022. [`arXiv:1710.08005`](https://arxiv.org/abs/1710.08005).

---

## 2. FPTO — Fairness-regularized Predict-Then-Optimize

**Code:** `method="fpto"`, `lambda_value > 0`.
`use_pred=True, use_fair=True, pred_weight_mode="fixed1"`. The dec branch is
inactive (`use_dec=False`).

**Description.** PTO with an added prediction-side fairness penalty
$\ell_{\mathrm{fair}}(\hat r)$ at weight $\lambda$. The MSE loss is augmented
with the MAD (or Atkinson / DP / BP / W$_2$-DP) violation between protected
groups, but the decision task is still solved post-hoc.

**Pseudocode.**

```latex
\begin{algorithm}[H]
\caption{FPTO ($\lambda$ scalarization)}
\KwIn{data, $\lambda \geq 0$}
\For{$t=1,\dots,T$}{
  $\hat r \gets f_\theta(x_{B_t})$\;
  $g_\mathrm{pred} \gets \nabla_\theta \ell_\mathrm{MSE}(\hat r, r_{B_t})$\;
  $g_\mathrm{fair} \gets \nabla_\theta \ell_\mathrm{fair}(\hat r, s_{B_t})$\;
  $\theta \gets \theta - \eta_t \,(g_\mathrm{pred} + \lambda\, g_\mathrm{fair})$\;
}
\end{algorithm}
```

**Reference.** Berk, Heidari, Jabbari, Joseph, Kearns, Morgenstern, Neel,
Roth. *A Convex Framework for Fair Regression*. FATML 2017.
[`arXiv:1706.02409`](https://arxiv.org/abs/1706.02409). MAD penalty form
follows the group-mean-discrepancy literature; see also Calders & Verwer 2010.

---

## 3. SAA — Sample-Average Approximation (data-poor baseline)

**Code:** `method="saa"`. Implemented in
[`training/loop.py:376`](../wheels/) as a featureless constant predictor.
No predictor is trained.

**Why we include it.** Decision-focused learning is fundamentally a
*data-driven optimization* problem, and SAA is the canonical **data-poor
baseline** in that literature: the decision maker who ignores the covariates
and prescribes from the unconditional empirical distribution of the uncertain
parameter. \citet{bertsimas2020predictive} formalize exactly this role —
the sample average is "the data-poor prediction baseline," and their
coefficient of prescriptiveness $P$ measures how far features take a model
*beyond* SAA toward perfect foresight. Reporting SAA therefore calibrates how
much the feature-aware methods actually buy.

**Description.** SAA replaces the per-individual prediction with the
**per-resource sample mean** of the training benefits — the best featureless
predictor — then solves the allocation once:

$$\hat r_{i,j} = \frac{1}{N}\sum_{k=1}^{N} r_{k,j} \quad \text{(same for every individual } i\text{)}.$$

**Pseudocode.**

```latex
\begin{algorithm}[H]
\caption{SAA (data-poor / featureless)}
$\bar r_j \gets \tfrac{1}{N}\sum_{k=1}^{N} r_{k,j}$ for each resource $j$\;
$\hat r_{i,j} \gets \bar r_j$ for all individuals $i$ \tcp*{per-resource mean, no features}
$d^\star \gets \arg\max_d\, W_\alpha(\hat r \odot d)$ subject to the budget\;
\end{algorithm}
```

**⚠️ Implementation note (audit 2026-06-02).** The current MD code at
`loop.py:382` uses `np.mean(y)` — a single **scalar** over the whole
`(N, n_resources)` benefit array — which collapses all resources to one
constant and is degenerate. The correct SAA uses the **per-resource** mean
`y.mean(axis=0)` (shape `(n_resources,)`), as in the pseudocode above. Fix is
a few lines + a rerun of the 25 MD-SAA cells (no training). The healthcare
SAA is already correct (its $y$ is one-dimensional).

**Reader caveat.** SAA's per-resource mean is group-independent, so its MAD
reflects only finite-sample noise around a constant and is naturally low —
this is a sample-average artifact, not a fairness mechanism. See §5.2 footnote.

**Reference.** Classical SAA: Kleywegt, Shapiro & Homem-de-Mello, *The Sample
Average Approximation Method for Stochastic Discrete Optimization*,
*SIAM J. Optim.* 12(2), 479–502, 2001; Shapiro, Dentcheva & Ruszczyński,
*Lectures on Stochastic Programming*, 2nd ed., SIAM, 2014, Ch. 5. SAA as the
data-poor baseline in contextual/data-driven optimization: Bertsimas & Kallus,
*From Predictive to Prescriptive Analytics*, *Management Science* 66(3),
1025–1044, 2020 [`arXiv:1402.5481`](https://arxiv.org/abs/1402.5481).

---

## 4. WDRO — Wasserstein DRO (variation-regularization form)

**Code:** `method="wdro"`. The training loop
([`loop.py:521-549`](../wheels/)) adds an input-gradient-norm penalty to the
MSE objective. Coefficient $\varepsilon_W$ is `wdro_epsilon` (default 0.1).

**What this actually is (be precise, audit 2026-06-02).** We do **not** solve
the exact min–max robust program. We implement the **variation /
gradient-norm regularization form** of Wasserstein-1 DRO. This is a recognized,
tractable instantiation of WDRO — not an ad-hoc regularizer:

- **Exact WDRO** is
  $\min_\theta \sup_{Q:\,W_1(Q,\hat P)\le\varepsilon} \mathbb{E}_Q[\ell(f_\theta(x), r)]$.
- For a broad class of losses — explicitly **including neural networks** —
  \citet{gao2024wasserstein} prove this is (asymptotically, and exactly in the
  regularization form) equivalent to ERM plus a gradient-norm penalty:
  $\min_\theta \tfrac1N\sum_i \ell_i + \varepsilon\cdot\tfrac1N\sum_i \lVert\nabla_x \ell_i\rVert_*$.
- \citet{sinha2018certifying} give the Lagrangian/penalty training procedure
  for deep models, which is what our implementation follows.

The **exact convex reformulation** of \citet{esfahani2018data} applies only to
convex losses (e.g. linear/logistic regression) and is **intractable for our
MLP predictor**, so the regularization form is the standard choice here. This
framing is what keeps a reviewer from objecting "this is not WDRO."

**Pseudocode.**

```latex
\begin{algorithm}[H]
\caption{WDRO (Wasserstein-1 variation-regularization form)}
\KwIn{$\varepsilon_W > 0$ (Wasserstein radius)}
\For{$t=1,\dots,T$}{
  $\ell_i \gets (f_\theta(x_i) - r_i)^2$ for $i \in B_t$\;
  $\pi \gets \varepsilon_W \cdot \tfrac{1}{B}\sum_i \|\nabla_{x_i}\ell_i\|_2$ \tcp*{$\approx$ worst-case loss over $W_1$ ball}
  $g \gets \nabla_\theta\!\left(\tfrac{1}{B}\sum_i \ell_i + \pi\right)$\;
  $\theta \gets \theta - \eta_t \, g$\;
}
\end{algorithm}
```

**Reference.** WDRO foundations: Mohajerin Esfahani & Kuhn, *Data-driven DRO
using the Wasserstein metric*, *Math. Programming* 171, 115–166, 2018; Gao &
Kleywegt, *DRO with Wasserstein Distance*, *Math. of OR* 48(2), 603–655, 2023
[`arXiv:1604.02199`](https://arxiv.org/abs/1604.02199). **Equivalence to
gradient/variation regularization** (the form we use): Gao, Chen & Kleywegt,
*Wasserstein DRO and Variation Regularization*, *Operations Research* 72(3),
1177–1191, 2024 [`arXiv:1712.06050`](https://arxiv.org/abs/1712.06050).
**Deep-learning penalty procedure:** Sinha, Namkoong & Duchi, *Certifying Some
Distributional Robustness with Principled Adversarial Training*, ICLR 2018
[`arXiv:1710.10571`](https://arxiv.org/abs/1710.10571); see also Volpi et al.,
NeurIPS 2018 [`arXiv:1805.12018`](https://arxiv.org/abs/1805.12018).

**Decision (for June 4):** keep this implementation and reframe + cite as
above (no compute cost, no genuine min–max RO baseline needed — exact WDRO is
intractable for the MLP anyway). Adding a separate exact-RO baseline is
optional scope, not required for correctness.

---

## 5. DFL — Decision-Focused Learning (no fairness)

**Code:** `method="fdfl"`, `lambda_value=0`. Sets `use_dec=True,
use_pred=False, use_fair=False, pred_weight_mode="zero"`.

**Description.** Train the predictor to *minimize downstream decision regret*
directly. The decision-side gradient $g_\mathrm{dec}$ is computed via the
closed-form $\alpha$-fair gradient (Proposition 1, healthcare) or
cvxpylayers (MD knapsack). Because `use_pred=False`, the prediction-MSE
branch is skipped, and `_combine_prediction_gradients` returns
$g_\mathrm{dec}$ directly — no `gradient_merge` is consulted.

**Pseudocode.**

```latex
\begin{algorithm}[H]
\caption{DFL}
\For{$t=1,\dots,T$}{
  $\hat r \gets f_\theta(x_{B_t})$\;
  $d^\star \gets \arg\max_d W_\alpha(\hat r \cdot d)$\;
  $g_\mathrm{dec} \gets \nabla_\theta\, \mathrm{regret}(\hat r, d^\star, r_{B_t})$\;
  $\theta \gets \theta - \eta_t\, g_\mathrm{dec}$\;
}
\end{algorithm}
```

**Reference.** Elmachtoub & Grigas 2022 (op. cit., §2.3); Wilder, Dilkina &
Tambe, *Melding the Data-Decisions Pipeline*, AAAI 2019,
[`arXiv:1809.05504`](https://arxiv.org/abs/1809.05504).

---

## 6. FDFL — Fair Decision-Focused Learning (this paper)

**Code:** `method="fdfl"`, `lambda_value > 0`. `use_dec=True, use_pred=False,
use_fair=True`.

**Description.** DFL augmented with a prediction-side fairness penalty.
Because `use_pred=False`, `_combine_prediction_gradients` takes the
*dec-only* branch and returns $g_\mathrm{dec} + \lambda\, g_\mathrm{fair}$.
This was not affected by the FDFL-Scal bug (see [`FDFL_SCAL_FIX_NOTE`](../docs/HANDOFF_RAW_FDFL_SCAL_REPLACEMENT.md)).

**Pseudocode.**

```latex
\begin{algorithm}[H]
\caption{FDFL ($\lambda$ scalarization)}
\For{$t=1,\dots,T$}{
  $\hat r \gets f_\theta(x_{B_t})$;\;
  $g_\mathrm{dec} \gets \nabla_\theta\, \mathrm{regret}(\hat r, d^\star(\hat r), r_{B_t})$\;
  $g_\mathrm{fair} \gets \nabla_\theta\, \ell_\mathrm{fair}(\hat r, s_{B_t})$\;
  $\theta \gets \theta - \eta_t \,(g_\mathrm{dec} + \lambda\, g_\mathrm{fair})$\;
}
\end{algorithm}
```

**Reference.** This paper, §3 (DF-MTFL framework) and §4 (closed-form
$\alpha$-fair gradient, Prop. 1).

---

## 7. FDFL-Scal(λ,μ) — Statically scalarized FDFL

**Code:** `method="fdfl_scal"`. `use_dec=True, use_pred=True, use_fair=True,
pred_weight_mode ∈ {"0.1","fixed1"}, gradient_merge="raw"`. Implemented in
`_combine_prediction_gradients`,
[`loop.py:221-224`](../wheels/).

**Description.** Three-objective raw scalarization:

$$g = g_\mathrm{dec} + \mu \cdot g_\mathrm{pred} + \lambda \cdot g_\mathrm{fair}.$$

$\mu$ is the **prediction-loss weight** — the static coefficient on the MSE
gradient $g_\mathrm{pred}$ (it regularizes the regret loss toward predictive
accuracy); $\lambda$ is the fairness weight. Paper sweeps $\mu \in \{0.1, 1\}$,
$\lambda \in \{0, 0.5, 1, 2\}$. Note **FDFL (§6) is the $\mu=0$ edge case** of
this family (prediction-loss term off); we keep "FDFL" as the headline label
for that point.

> **Terminology (2026-06-02):** $\mu$ was previously called the "prediction
> anchor weight." That term is **retired** to avoid collision with FDFL-PLG's
> prediction-loss *guidance*. $\mu$ is now simply the *prediction-loss weight*.

**Bug-fix history.** Pre-fix, `_combine_prediction_gradients` always used
the geometric-mean **guided** merge regardless of `pred_weight_mode`,
making FDFL-Scal numerically equivalent to FPLG at $\alpha=1$. The
`gradient_merge="raw"` switch restores the intended raw weighted-sum
semantics. See [`HANDOFF_RAW_FDFL_SCAL_REPLACEMENT.md`](../docs/HANDOFF_RAW_FDFL_SCAL_REPLACEMENT.md).

**Pseudocode (raw, post-fix).**

```latex
\begin{algorithm}[H]
\caption{FDFL-Scal($\lambda,\mu$) (raw scalarization)}
\KwIn{$\mu \geq 0$ (prediction-loss weight), $\lambda \geq 0$ (fairness weight)}
\For{$t=1,\dots,T$}{
  $\hat r \gets f_\theta(x_{B_t})$\;
  $g_\mathrm{dec} \gets \nabla_\theta\, \mathrm{regret}(\hat r, d^\star, r_{B_t})$\;
  $g_\mathrm{pred} \gets \nabla_\theta\, \ell_\mathrm{MSE}(\hat r, r_{B_t})$\;
  $g_\mathrm{fair} \gets \nabla_\theta\, \ell_\mathrm{fair}(\hat r, s_{B_t})$\;
  $g \gets g_\mathrm{dec} + \mu\, g_\mathrm{pred} + \lambda\, g_\mathrm{fair}$ \tcp*{raw merge}
  $\theta \gets \theta - \eta_t\, g$\;
}
\end{algorithm}
```

**Reference.** This paper, §3.2 (scalarized instantiation of DF-MTFL).

---

## 8. FDFL-PLG — Fair Prediction-Loss-Guided DFL (was FPLG-κ)

**Origin.** PLG = **Prediction-Loss-Guided** decision-focused learning,
introduced by \citet{jeon2025plg}. Their idea: rather than scalarize, *guide*
the decision-loss gradient with the prediction-loss gradient via a decaying
schedule, so prediction information stabilizes early training and fades as
decisions take over — "requires no extra training, composes with any DFL
solver." **FDFL-PLG is our fairness extension** of PLG: we add the
prediction-fairness gradient to the guided direction.

**Code:** `method="fplg"`. `use_dec=True, use_pred=True, use_fair=True,
pred_weight_mode="schedule", continuation=True, gradient_merge="guided"`, with
schedule parameter $\kappa$ (`mo_plg_kappa_decay`). The guided merge is
`merge_guided_dec_pred_gradient`
([`algorithms/torch_utils.py:82`](../wheels/)), reached via
`_combine_prediction_gradients`.

**Description.** Instead of a raw weighted sum, the dec/pred gradients are
combined as **unit directions** and the result is rescaled to the
**geometric mean** of their norms; the fairness term is added afterward:

$$\hat u_\mathrm{dec}=\tfrac{g_\mathrm{dec}}{\lVert g_\mathrm{dec}\rVert},\quad
\hat u_\mathrm{pred}=\tfrac{g_\mathrm{pred}}{\lVert g_\mathrm{pred}\rVert},\qquad
g = \sqrt{\lVert g_\mathrm{dec}\rVert\,\lVert g_\mathrm{pred}\rVert}\cdot
\frac{\hat u_\mathrm{dec}+\alpha_t\,\hat u_\mathrm{pred}}{\lVert\hat u_\mathrm{dec}+\alpha_t\,\hat u_\mathrm{pred}\rVert}
+ \lambda\, g_\mathrm{fair},$$

where the guidance weight $\alpha_t$ follows the PLG decay schedule
$\kappa_t = \kappa_0/(1+\kappa_\mathrm{decay}\,t)$.

**⚠️ Execution-path audit (2026-06-02 — needs a code trace before §5.1).**
There are **two** PLG-flavored mechanisms in the package: (a) the guided
geometric-mean merge above (`torch_utils.merge_guided_dec_pred_gradient`,
reached when `method="fplg"` with `gradient_merge="guided"`), and (b) a more
literal decaying-orthogonal-perturbation handler
`PLGHandler3Obj` (`mo_method="plg3"`, `mo_handler.py:1049`, uses
`direction = d_primary + κ_t·g_orth`). The experiment label **`FPLG-κ1`** is
configured as base `FPLG` + `mo_plg_kappa_decay=0.01`; whether that routes
through (a) or (b) at run time is **ambiguous from config alone** and must be
traced so the paper describes the exact mechanism it ran. The formula above
documents path (a); confirm before finalizing §5.1.

**Pseudocode (path a).**

```latex
\begin{algorithm}[H]
\caption{FDFL-PLG (guided geometric-mean merge)}
\KwIn{$\kappa$ decay rate, $\lambda$; guidance weight $\alpha_t=\kappa_0/(1+\kappa_\mathrm{decay}\,t)$}
\For{$t=1,\dots,T$}{
  $g_\mathrm{dec}, g_\mathrm{pred}, g_\mathrm{fair}$ as in FDFL-Scal\;
  $\hat u_\mathrm{dec} \gets g_\mathrm{dec}/\|g_\mathrm{dec}\|$;\quad $\hat u_\mathrm{pred} \gets g_\mathrm{pred}/\|g_\mathrm{pred}\|$\;
  $v \gets \hat u_\mathrm{dec} + \alpha_t\,\hat u_\mathrm{pred}$\;
  $g_\mathrm{merge} \gets \sqrt{\|g_\mathrm{dec}\|\,\|g_\mathrm{pred}\|}\cdot v/\|v\|$ \tcp*{geom-mean rescale}
  $\theta \gets \theta - \eta_t (g_\mathrm{merge} + \lambda g_\mathrm{fair})$\;
}
\end{algorithm}
```

**Reference.** Jeon, Bae, Kim, Lee & Kim, *Prediction Loss Guided
Decision-Focused Learning*, 2025
[`arXiv:2509.08359`](https://arxiv.org/abs/2509.08359) (the PLG method);
this paper, §3.3 (fairness extension). Related norm-balancing idea: Chen et
al., *GradNorm*, ICML 2018 [`arXiv:1711.02257`](https://arxiv.org/abs/1711.02257).

---

## 9. PCGrad — Projecting Conflicting Gradients

**Code:** `mo_method="pcgrad"` — handled by `PCGradHandler` in
[`algorithms/mo_handler.py:165`](../wheels/).

**Description.** For each pair of objective gradients $(g_i, g_j)$ with
negative cosine, project $g_i$ onto the normal plane of $g_j$:
$g_i \leftarrow g_i - \tfrac{\langle g_i, g_j\rangle}{\|g_j\|^2}\, g_j$.
Pairs are processed in random order. The descent direction is the sum of
the projected per-objective gradients. In our 3-objective setting
$(g_\mathrm{dec}, g_\mathrm{pred}, g_\mathrm{fair})$ are merged this way.

**Pseudocode.**

```latex
\begin{algorithm}[H]
\caption{PCGrad}
\KwIn{$\{g_i\}_{i=1}^M$ objective gradients}
$\tilde g_i \gets g_i$ for all $i$\;
\ForEach{$i \in \mathrm{shuffle}(1{:}M)$}{
  \ForEach{$j \neq i$}{
    \If{$\langle \tilde g_i, g_j \rangle < 0$}{
      $\tilde g_i \gets \tilde g_i - \tfrac{\langle \tilde g_i, g_j\rangle}{\|g_j\|^2} g_j$\;
    }
  }
}
$g \gets \sum_i \tilde g_i$\;
\end{algorithm}
```

**Reference.** Yu, Kumar, Gupta, Levine, Hausman & Finn, *Gradient
Surgery for Multi-Task Learning*, NeurIPS 2020.
[`arXiv:2001.06782`](https://arxiv.org/abs/2001.06782).

---

## 10. MGDA — Multiple-Gradient Descent Algorithm

**Code:** `mo_method="mgda"` — `MGDAHandler` in
[`algorithms/mo_handler.py:837`](../wheels/).

**Description.** Solve for the minimum-norm convex combination of the
objective gradients:

$$\lambda^\star = \arg\min_{\lambda \in \Delta^{M-1}} \left\| \sum_i \lambda_i\, g_i \right\|_2^2, \qquad g \gets \sum_i \lambda_i^\star g_i.$$

The resulting $g$ is a common descent direction for all objectives
(stationarity $\Leftrightarrow$ $\|g\|=0$). The QP is solved by
Frank–Wolfe in `_solve_mgda_qp`.

**Reference.** Sener & Koltun, *Multi-Task Learning as Multi-Objective
Optimization*, NeurIPS 2018.
[`arXiv:1810.04650`](https://arxiv.org/abs/1810.04650). Underlying MGDA:
Désidéri, *C. R. Math.* 350(5–6), 2012.

---

## 11. NashMTL — Nash Bargaining for Multi-Task Learning

**Code:** `mo_method="nashmtl"` — `NashMTLHandler` in
[`algorithms/mo_handler.py:1383`](../wheels/).

**Description.** Choose per-objective weights $\alpha_i \geq 0$ as the
Nash bargaining solution: $\alpha^\star = \arg\max_{\alpha}
\prod_i \alpha_i$ subject to $G\alpha \in B$ where $G_{ij} = g_i^\top g_j$
is the Gram matrix and $B$ is the unit ball. Equivalently, find the
direction inside the feasible set that maximizes the geometric mean of
per-task improvements. Solved by Frank–Wolfe (`_nash_frank_wolfe`).
Periodically renormalizes per-task weights every `update_weights_every`
steps.

**Reference.** Navon, Shamsian, Achituve, Maron, Kawaguchi, Chechik &
Fetaya, *Multi-Task Learning as a Bargaining Game*, NeurIPS 2022.
[`arXiv:2202.01017`](https://arxiv.org/abs/2202.01017).

---

## 12. CAGrad — Conflict-Averse Gradient Descent

**Code:** `mo_method="cagrad"` — `CAGradHandler` in
[`algorithms/mo_handler.py:912`](../wheels/), with hyperparameter
$c \in [0,1]$ (`mo_cagrad_c`, default 0.5).

**Description.** Solve for a descent direction $g$ that maximizes the
worst-case per-task improvement subject to staying within $c$ times the
norm of the average gradient:

$$g = \arg\max_{g} \min_i \langle g, g_i \rangle \quad \text{s.t.}\quad \|g - \bar g\|_2 \leq c\, \|\bar g\|_2,$$

where $\bar g = \tfrac{1}{M}\sum_i g_i$. Solved by the dual QP in
`_solve_cagrad_qp`. The hyperparameter $c$ trades off average-direction
fidelity ($c=0$: exactly $\bar g$, i.e., weighted sum) against worst-task
guarantee ($c=1$: closer to MGDA's minimum-norm direction).

**Cross-experiment behavior.** Strong on HC at $\alpha=2.0$ (regret 0.128)
but on the MD synthetic family records high MAD (0.94–1.53) and loses
seeds at $\mathrm{imb}\geq 0.6$. See §5.2 footnote.

**Reference.** Liu, Liu, Jin, Stone & Liu, *Conflict-Averse Gradient
Descent for Multi-Task Learning*, ICLR 2021.
[`arXiv:2110.14048`](https://arxiv.org/abs/2110.14048).

---

## Cross-cutting code facts

- **`_combine_prediction_gradients` (`loop.py:192`)** only consults
  `gradient_merge` when both `use_dec` *and* `use_pred` are true. Methods
  with `use_pred=False` (FDFL, DFL) take the dec-only branch
  $g_\mathrm{dec} + \lambda\, g_\mathrm{fair}$ unconditionally — they were
  never affected by the FDFL-Scal bug.

- **MOO handlers bypass the merge entirely.** When `mo_method` is set,
  `train_single_stage` dispatches to the handler's `compute_direction`
  with the *per-objective parameter-space gradients*, and the
  `gradient_merge` flag is ignored.

- **Decision-gradient backends:** healthcare uses the closed-form
  $\alpha$-fair gradient of Proposition 1
  ([`fair_dfl/decision/strategies/analytic.py`](../wheels/));
  MD knapsack uses `cvxpylayers` for the conic decision gradient
  ([`cvxpylayers.py`](../wheels/)).

- **Seed-dropping convention:** runs whose final regret exceeds 5
  *or* final fairness violation exceeds 10 are dropped from the per-cell
  aggregate. The high-imbalance MD blow-ups (mainly seed=33 at
  $\mathrm{imb}\geq 0.6$) affect raw FDFL-Scal *and* guided baselines
  (DFL, FDFL, MGDA) on the same splits — they reflect the regime, not the
  scalarization choice.

---

*Last updated 2026-05-30 — locked for IJOC submission.*
