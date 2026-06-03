# Methods Reference

This document catalogs every method evaluated in the paper and provides the
canonical mapping between **paper label**, **code symbol**, **update rule**, and
**original publication**. It is the authoritative reference for what each method
computes at the code level.

Each entry is laid out as a reference page:

- **Bucket** — role in the framework.
- **Code key** — the `method` / `mo_method` value that selects it.
- **Flags** — the `MethodSpec` fields (`use_dec`, `use_pred`, `use_fair`,
  `pred_weight_mode`, `gradient_merge`) set in
  [`fdfl_harness/configs.py`](fdfl_harness/configs.py).
- **Entry point** — the function or class that implements the update (paths refer
  to the bundled wheel `wheels/fair_dfl_moo-*.whl`).
- **Description** and **Update step** (rendered pseudocode).
- **Reference**.

### Notation

| Symbol | Meaning |
|---|---|
| $f_\theta(x)$ | predictor with parameters $\theta$ |
| $\hat r = f_\theta(x)$ | predicted reward; $r$ is the true reward |
| $d^\star(\hat r)$ | decision-task solution at prediction $\hat r$ |
| $W_\alpha$ | $\alpha$-fair welfare |
| $\ell_{\text{MSE}}$ | mean-squared prediction loss |
| $\ell_{\text{fair}}$ | prediction-fairness penalty (e.g. MAD) |
| $g_{\text{dec}}, g_{\text{pred}}, g_{\text{fair}}$ | parameter-space gradients of the decision, prediction, and fairness objectives |

All gradients are taken with respect to $\theta$ unless stated otherwise, and
$B_t$ denotes the minibatch at step $t$.

---

## Overview

| Paper label | Bucket | Code key (`method`) | Inline flags / `mo_method` | Reference |
|---|---|---|---|---|
| PTO | No-fairness | `fpto` (λ=0) | `use_pred=T` | Elmachtoub & Grigas 2022 |
| FPTO | Prediction-focused | `fpto` | `use_pred=T, use_fair=T` | Berk et al. 2017 |
| SAA | Robust baseline | `saa` | constant predictor (mean of $y$) | Shapiro et al. 2014 |
| WDRO | Robust baseline | `wdro` | `use_pred=T` + Wasserstein gradient pen. | Gao & Kleywegt 2023 |
| DFL | Decision-only | `fdfl` (λ=0) | `use_dec=T` | Elmachtoub & Grigas 2022 |
| FDFL | End-to-end (focal) | `fdfl` | `use_dec=T, use_fair=T` | this paper |
| FDFL-Scal(λ,μ) | Integrated · static scalarized | `fdfl_scal` | `use_dec=T, use_pred=T, use_fair=T, gradient_merge="raw"` | this paper |
| FDFL-PLG | Integrated · hybrid | `fplg` | `use_dec=T, use_pred=T, use_fair=T, gradient_merge="guided"` | Jeon et al. 2025 (PLG) + this paper |
| FDFL-PCGrad | Integrated · dynamic | `pcgrad` | `mo_method="pcgrad"` | Yu et al. NeurIPS 2020 |
| FDFL-MGDA | Integrated · dynamic | `mgda` | `mo_method="mgda"` | Sener & Koltun NeurIPS 2018 |
| FDFL-NashMTL | Integrated · dynamic | `nashmtl` | `mo_method="nashmtl"` | Navon et al. NeurIPS 2022 |
| FDFL-CAGrad | Integrated · dynamic | `cagrad` | `mo_method="cagrad"` | Liu et al. ICLR 2021 |

> [!NOTE]
> **Naming taxonomy.** *Separate* methods are SAA, PTO, FPTO, WDRO.
> *Integrated · static scalarized* are FDFL and FDFL-Scal(λ,μ).
> *Integrated · hybrid* is FDFL-PLG. *Integrated · dynamic* are
> FDFL-PCGrad / MGDA / NashMTL / CAGrad. The multi-objective handlers carry the
> `FDFL-` prefix to mark that they instantiate the same DF-MTFL framework.
> FDFL-Scal's $\mu$ is the **prediction-loss weight**; the prediction-loss
> *guidance* mechanism is what FDFL-PLG implements.

**Code organization.** All multi-objective handlers reside in
`fair_dfl/algorithms/mo_handler.py`. All scalarized methods route through
`fair_dfl/training/loop.py::_combine_prediction_gradients`.

---

## PTO — Predict-Then-Optimize (no fairness)

- **Bucket:** No-fairness (separate)
- **Code key:** `method="fpto"`, `lambda_value=0`
- **Flags:** `use_pred=True, use_fair=False, pred_weight_mode="fixed1"`

**Description.** Train $f_\theta$ by minimizing prediction MSE only; at deployment,
plug $\hat r$ into the welfare-optimal decision. The classical two-stage baseline.

**Update step.**

$$
\begin{aligned}
&\textbf{for } t = 1, \dots, T:\\
&\quad g_{\text{pred}} \leftarrow \nabla_\theta\, \tfrac{1}{B}\textstyle\sum_{i \in B_t} (f_\theta(x_i) - r_i)^2\\
&\quad \theta \leftarrow \theta - \eta_t\, g_{\text{pred}}\\
&\textbf{return } \theta,\ d^\star\!\big(f_\theta(x_{\text{test}})\big)
\end{aligned}
$$

**Reference.** Elmachtoub & Grigas, *Smart "Predict, then Optimize"*, Management
Science 68(1), 9–26, 2022. [arXiv:1710.08005](https://arxiv.org/abs/1710.08005).

---

## FPTO — Fairness-regularized Predict-Then-Optimize

- **Bucket:** Prediction-focused (separate)
- **Code key:** `method="fpto"`, `lambda_value > 0`
- **Flags:** `use_pred=True, use_fair=True, use_dec=False, pred_weight_mode="fixed1"`

**Description.** PTO with an added prediction-side fairness penalty
$\ell_{\text{fair}}(\hat r)$ at weight $\lambda$. The MSE loss is augmented with the
MAD (or Atkinson / DP / BP / W₂-DP) violation between protected groups, but the
decision task is still solved post hoc.

**Update step.**

$$
\begin{aligned}
&\textbf{Input: } \lambda \ge 0\\
&\textbf{for } t = 1, \dots, T:\\
&\quad \hat r \leftarrow f_\theta(x_{B_t})\\
&\quad g_{\text{pred}} \leftarrow \nabla_\theta\, \ell_{\text{MSE}}(\hat r, r_{B_t})\\
&\quad g_{\text{fair}} \leftarrow \nabla_\theta\, \ell_{\text{fair}}(\hat r, s_{B_t})\\
&\quad \theta \leftarrow \theta - \eta_t\,(g_{\text{pred}} + \lambda\, g_{\text{fair}})
\end{aligned}
$$

**Reference.** Berk, Heidari, Jabbari, Joseph, Kearns, Morgenstern, Neel, Roth,
*A Convex Framework for Fair Regression*, FATML 2017.
[arXiv:1706.02409](https://arxiv.org/abs/1706.02409). The MAD penalty follows the
group-mean-discrepancy literature; see also Calders & Verwer 2010.

---

## SAA — Sample-Average Approximation (data-poor baseline)

- **Bucket:** Robust baseline (separate)
- **Code key:** `method="saa"`
- **Entry point:** `fair_dfl/training/loop.py` — featureless constant predictor; no predictor is trained.

**Description.** SAA is the canonical *data-poor* baseline in data-driven
optimization: the decision maker ignores covariates and prescribes from the
unconditional empirical distribution of the uncertain parameter. It replaces the
per-individual prediction with the **per-resource sample mean** of the training
benefits — the best featureless predictor — then solves the allocation once:

$$\hat r_{i,j} = \frac{1}{N}\sum_{k=1}^{N} r_{k,j} \quad \text{(identical for every individual } i\text{)}.$$

Reporting SAA calibrates how much the feature-aware methods actually buy:
Bertsimas & Kallus formalize the sample average as "the data-poor prediction
baseline," and their coefficient of prescriptiveness $P$ measures how far features
move a model beyond SAA toward perfect foresight.

**Update step.**

$$
\begin{aligned}
&\bar r_j \leftarrow \tfrac{1}{N}\textstyle\sum_{k=1}^{N} r_{k,j} \quad \text{for each resource } j\\
&\hat r_{i,j} \leftarrow \bar r_j \quad \text{for all } i \qquad \triangleright\ \text{per-resource mean, no features}\\
&d^\star \leftarrow \arg\max_d\, W_\alpha(\hat r \odot d) \quad \text{s.t. budget}
\end{aligned}
$$

> [!WARNING]
> **Implementation caveat.** The correct featureless predictor is the
> **per-resource** mean `y.mean(axis=0)` (shape `(n_resources,)`), as above. A
> single **scalar** mean `np.mean(y)` over the whole `(N, n_resources)` benefit
> array collapses all resources to one constant and is degenerate. The healthcare
> SAA is unaffected (its $y$ is one-dimensional).

> [!NOTE]
> SAA's per-resource mean is group-independent, so its MAD reflects only
> finite-sample noise around a constant and is naturally low — a sample-average
> artifact, not a fairness mechanism.

**Reference.** Classical SAA: Kleywegt, Shapiro & Homem-de-Mello, *The Sample
Average Approximation Method for Stochastic Discrete Optimization*, SIAM J. Optim.
12(2), 479–502, 2001; Shapiro, Dentcheva & Ruszczyński, *Lectures on Stochastic
Programming*, 2nd ed., SIAM, 2014, Ch. 5. SAA as the data-poor baseline:
Bertsimas & Kallus, *From Predictive to Prescriptive Analytics*, Management
Science 66(3), 1025–1044, 2020.
[arXiv:1402.5481](https://arxiv.org/abs/1402.5481).

---

## WDRO — Wasserstein DRO (variation-regularization form)

- **Bucket:** Robust baseline (separate)
- **Code key:** `method="wdro"`
- **Entry point:** `fair_dfl/training/loop.py` — adds an input-gradient-norm penalty to the MSE objective; coefficient $\varepsilon_W$ is `wdro_epsilon` (default 0.1).

**Description.** This implements the **variation / gradient-norm regularization
form** of Wasserstein-1 DRO — a recognized, tractable instantiation, not an
ad-hoc regularizer. Exact WDRO is the min–max program

$$\min_\theta\ \sup_{Q:\,W_1(Q,\hat P)\le\varepsilon}\ \mathbb{E}_Q[\ell(f_\theta(x), r)].$$

For a broad class of losses — explicitly including neural networks — this is
equivalent (asymptotically, and exactly in the regularization form) to ERM plus a
gradient-norm penalty,

$$\min_\theta\ \tfrac1N\textstyle\sum_i \ell_i + \varepsilon\cdot\tfrac1N\textstyle\sum_i \lVert\nabla_x \ell_i\rVert_*,$$

and Sinha, Namkoong & Duchi give the Lagrangian/penalty training procedure for
deep models that this implementation follows. The exact convex reformulation of
Mohajerin Esfahani & Kuhn applies only to convex losses and is intractable for the
MLP predictor, so the regularization form is the standard choice here.

**Update step.**

$$
\begin{aligned}
&\textbf{Input: } \varepsilon_W > 0 \ \text{(Wasserstein radius)}\\
&\textbf{for } t = 1, \dots, T:\\
&\quad \ell_i \leftarrow (f_\theta(x_i) - r_i)^2, \quad i \in B_t\\
&\quad \pi \leftarrow \varepsilon_W \cdot \tfrac{1}{B}\textstyle\sum_i \lVert \nabla_{x_i} \ell_i \rVert_2 \qquad \triangleright\ \text{worst-case loss over } W_1 \text{ ball}\\
&\quad g \leftarrow \nabla_\theta\!\left( \tfrac{1}{B}\textstyle\sum_i \ell_i + \pi \right)\\
&\quad \theta \leftarrow \theta - \eta_t\, g
\end{aligned}
$$

**Reference.** WDRO foundations: Mohajerin Esfahani & Kuhn, *Data-driven DRO using
the Wasserstein metric*, Math. Programming 171, 115–166, 2018; Gao & Kleywegt,
*DRO with Wasserstein Distance*, Math. of OR 48(2), 603–655, 2023.
[arXiv:1604.02199](https://arxiv.org/abs/1604.02199). Equivalence to
gradient/variation regularization (the form used here): Gao, Chen & Kleywegt,
*Wasserstein DRO and Variation Regularization*, Operations Research 72(3),
1177–1191, 2024. [arXiv:1712.06050](https://arxiv.org/abs/1712.06050).
Deep-learning penalty procedure: Sinha, Namkoong & Duchi, *Certifying Some
Distributional Robustness with Principled Adversarial Training*, ICLR 2018.
[arXiv:1710.10571](https://arxiv.org/abs/1710.10571). See also Volpi et al.,
NeurIPS 2018, [arXiv:1805.12018](https://arxiv.org/abs/1805.12018).

---

## DFL — Decision-Focused Learning (no fairness)

- **Bucket:** Decision-only
- **Code key:** `method="fdfl"`, `lambda_value=0`
- **Flags:** `use_dec=True, use_pred=False, use_fair=False, pred_weight_mode="zero"`

**Description.** Train the predictor to minimize downstream decision regret
directly. The decision-side gradient $g_{\text{dec}}$ comes from the closed-form
$\alpha$-fair gradient (Proposition 1, healthcare) or from `cvxpylayers` (MD
knapsack). Because `use_pred=False`, the prediction-MSE branch is skipped and
`_combine_prediction_gradients` returns $g_{\text{dec}}$ directly —
`gradient_merge` is not consulted.

**Update step.**

$$
\begin{aligned}
&\textbf{for } t = 1, \dots, T:\\
&\quad \hat r \leftarrow f_\theta(x_{B_t})\\
&\quad d^\star \leftarrow \arg\max_d\, W_\alpha(\hat r \cdot d)\\
&\quad g_{\text{dec}} \leftarrow \nabla_\theta\, \text{regret}(\hat r, d^\star, r_{B_t})\\
&\quad \theta \leftarrow \theta - \eta_t\, g_{\text{dec}}
\end{aligned}
$$

**Reference.** Elmachtoub & Grigas 2022 (op. cit., §2.3); Wilder, Dilkina & Tambe,
*Melding the Data-Decisions Pipeline*, AAAI 2019.
[arXiv:1809.05504](https://arxiv.org/abs/1809.05504).

---

## FDFL — Fair Decision-Focused Learning (this paper)

- **Bucket:** End-to-end (focal)
- **Code key:** `method="fdfl"`, `lambda_value > 0`
- **Flags:** `use_dec=True, use_pred=False, use_fair=True`

**Description.** DFL augmented with a prediction-side fairness penalty. Because
`use_pred=False`, `_combine_prediction_gradients` takes the dec-only branch and
returns $g_{\text{dec}} + \lambda\, g_{\text{fair}}$. This path is independent of
the scalarized merge and is the $\mu=0$ edge case of FDFL-Scal.

**Update step.**

$$
\begin{aligned}
&\textbf{Input: } \lambda \ge 0\\
&\textbf{for } t = 1, \dots, T:\\
&\quad \hat r \leftarrow f_\theta(x_{B_t})\\
&\quad g_{\text{dec}} \leftarrow \nabla_\theta\, \text{regret}(\hat r, d^\star(\hat r), r_{B_t})\\
&\quad g_{\text{fair}} \leftarrow \nabla_\theta\, \ell_{\text{fair}}(\hat r, s_{B_t})\\
&\quad \theta \leftarrow \theta - \eta_t\,(g_{\text{dec}} + \lambda\, g_{\text{fair}})
\end{aligned}
$$

**Reference.** This paper, §3 (DF-MTFL framework) and §4 (closed-form $\alpha$-fair
gradient, Prop. 1).

---

## FDFL-Scal(λ,μ) — Statically scalarized FDFL

- **Bucket:** Integrated · static scalarized
- **Code key:** `method="fdfl_scal"`
- **Flags:** `use_dec=True, use_pred=True, use_fair=True, pred_weight_mode ∈ {"0.1","fixed1"}, gradient_merge="raw"`
- **Entry point:** `fair_dfl/training/loop.py::_combine_prediction_gradients`

**Description.** Three-objective raw scalarization,

$$g = g_{\text{dec}} + \mu \cdot g_{\text{pred}} + \lambda \cdot g_{\text{fair}}.$$

$\mu$ is the **prediction-loss weight** — the static coefficient on the MSE
gradient $g_{\text{pred}}$, regularizing the regret loss toward predictive
accuracy; $\lambda$ is the fairness weight. The paper sweeps $\mu \in \{0.1, 1\}$
and $\lambda \in \{0, 0.5, 1, 2\}$. FDFL (above) is the $\mu=0$ edge case of this
family; "FDFL" is retained as the headline label for that point.

**Update step.**

$$
\begin{aligned}
&\textbf{Input: } \mu \ge 0 \ \text{(prediction-loss weight)}, \ \lambda \ge 0 \ \text{(fairness weight)}\\
&\textbf{for } t = 1, \dots, T:\\
&\quad \hat r \leftarrow f_\theta(x_{B_t})\\
&\quad g_{\text{dec}} \leftarrow \nabla_\theta\, \text{regret}(\hat r, d^\star, r_{B_t})\\
&\quad g_{\text{pred}} \leftarrow \nabla_\theta\, \ell_{\text{MSE}}(\hat r, r_{B_t})\\
&\quad g_{\text{fair}} \leftarrow \nabla_\theta\, \ell_{\text{fair}}(\hat r, s_{B_t})\\
&\quad g \leftarrow g_{\text{dec}} + \mu\, g_{\text{pred}} + \lambda\, g_{\text{fair}} \qquad \triangleright\ \text{raw merge}\\
&\quad \theta \leftarrow \theta - \eta_t\, g
\end{aligned}
$$

> [!NOTE]
> **Merge semantics.** An earlier revision applied the geometric-mean *guided*
> merge regardless of `pred_weight_mode`, which made FDFL-Scal numerically
> equivalent to FDFL-PLG at $\alpha=1$. Setting `gradient_merge="raw"` restores the
> intended raw weighted-sum semantics shown above.

**Reference.** This paper, §3.2 (scalarized instantiation of DF-MTFL).

---

## FDFL-PLG — Fair Prediction-Loss-Guided DFL

- **Bucket:** Integrated · hybrid
- **Code key:** `method="fplg"`
- **Flags:** `use_dec=True, use_pred=True, use_fair=True, pred_weight_mode="schedule", continuation=True, gradient_merge="guided"`; schedule parameter $\kappa$ = `mo_plg_kappa_decay`
- **Entry point:** `fair_dfl/algorithms/torch_utils.py::merge_guided_dec_pred_gradient`, reached via `_combine_prediction_gradients`

**Description.** PLG (Prediction-Loss-Guided DFL) *guides* the decision-loss
gradient with the prediction-loss gradient on a decaying schedule, so prediction
information stabilizes early training and fades as decisions take over. FDFL-PLG is
the fairness extension: the prediction-fairness gradient is added to the guided
direction. Rather than a raw weighted sum, the dec/pred gradients are combined as
**unit directions** and rescaled to the **geometric mean** of their norms; the
fairness term is added afterward:

$$\hat u_{\text{dec}}=\frac{g_{\text{dec}}}{\lVert g_{\text{dec}}\rVert},\quad
\hat u_{\text{pred}}=\frac{g_{\text{pred}}}{\lVert g_{\text{pred}}\rVert},\qquad
g = \sqrt{\lVert g_{\text{dec}}\rVert\,\lVert g_{\text{pred}}\rVert}\cdot
\frac{\hat u_{\text{dec}}+\alpha_t\,\hat u_{\text{pred}}}{\lVert\hat u_{\text{dec}}+\alpha_t\,\hat u_{\text{pred}}\rVert}
+ \lambda\, g_{\text{fair}},$$

where the guidance weight follows the PLG decay schedule
$\alpha_t = \kappa_0/(1+\kappa_{\text{decay}}\,t)$.

**Update step.**

$$
\begin{aligned}
&\textbf{Input: } \kappa \ \text{decay rate}, \ \lambda; \quad \alpha_t = \kappa_0/(1+\kappa_{\text{decay}}\,t)\\
&\textbf{for } t = 1, \dots, T:\\
&\quad \text{compute } g_{\text{dec}}, g_{\text{pred}}, g_{\text{fair}} \ \text{as in FDFL-Scal}\\
&\quad \hat u_{\text{dec}} \leftarrow g_{\text{dec}}/\lVert g_{\text{dec}}\rVert, \quad \hat u_{\text{pred}} \leftarrow g_{\text{pred}}/\lVert g_{\text{pred}}\rVert\\
&\quad v \leftarrow \hat u_{\text{dec}} + \alpha_t\,\hat u_{\text{pred}}\\
&\quad g_{\text{merge}} \leftarrow \sqrt{\lVert g_{\text{dec}}\rVert\,\lVert g_{\text{pred}}\rVert}\cdot v/\lVert v\rVert \qquad \triangleright\ \text{geom-mean rescale}\\
&\quad \theta \leftarrow \theta - \eta_t\,(g_{\text{merge}} + \lambda\, g_{\text{fair}})
\end{aligned}
$$

> [!NOTE]
> **Two PLG code paths.** The package contains two PLG-flavored mechanisms: (a) the
> guided geometric-mean merge above (`torch_utils.merge_guided_dec_pred_gradient`,
> reached when `method="fplg"` with `gradient_merge="guided"`), documented here; and
> (b) a literal decaying-orthogonal-perturbation handler `PLGHandler3Obj`
> (`mo_method="plg3"`), which uses $\text{direction} = d_{\text{primary}} + \kappa_t\, g_{\text{orth}}$.
> The `FPLG-κ1` configuration is base `FPLG` + `mo_plg_kappa_decay=0.01` and routes
> through path (a).

**Reference.** Jeon, Bae, Kim, Lee & Kim, *Prediction Loss Guided Decision-Focused
Learning*, 2025. [arXiv:2509.08359](https://arxiv.org/abs/2509.08359). Fairness
extension: this paper, §3.3. Related norm-balancing idea: Chen et al., *GradNorm*,
ICML 2018. [arXiv:1711.02257](https://arxiv.org/abs/1711.02257).

---

## FDFL-PCGrad — Projecting Conflicting Gradients

- **Bucket:** Integrated · dynamic
- **Code key:** `mo_method="pcgrad"`
- **Entry point:** `PCGradHandler`, `fair_dfl/algorithms/mo_handler.py`

**Description.** For each pair of objective gradients $(g_i, g_j)$ with negative
cosine, project $g_i$ onto the normal plane of $g_j$. Pairs are processed in random
order; the descent direction is the sum of the projected per-objective gradients.
The three objectives merged this way are $(g_{\text{dec}}, g_{\text{pred}}, g_{\text{fair}})$.

**Update step.**

$$
\begin{aligned}
&\textbf{Input: } \{g_i\}_{i=1}^{M} \ \text{objective gradients}\\
&\tilde g_i \leftarrow g_i \ \text{for all } i\\
&\textbf{for } i \in \text{shuffle}(1{:}M):\\
&\quad \textbf{for } j \neq i:\\
&\quad\quad \textbf{if } \langle \tilde g_i, g_j\rangle < 0:\\
&\quad\quad\quad \tilde g_i \leftarrow \tilde g_i - \frac{\langle \tilde g_i, g_j\rangle}{\lVert g_j\rVert^2}\, g_j\\
&g \leftarrow \textstyle\sum_i \tilde g_i
\end{aligned}
$$

**Reference.** Yu, Kumar, Gupta, Levine, Hausman & Finn, *Gradient Surgery for
Multi-Task Learning*, NeurIPS 2020.
[arXiv:2001.06782](https://arxiv.org/abs/2001.06782).

---

## FDFL-MGDA — Multiple-Gradient Descent Algorithm

- **Bucket:** Integrated · dynamic
- **Code key:** `mo_method="mgda"`
- **Entry point:** `MGDAHandler`, `fair_dfl/algorithms/mo_handler.py`

**Description.** Solve for the minimum-norm convex combination of the objective
gradients,

$$\lambda^\star = \arg\min_{\lambda \in \Delta^{M-1}} \left\lVert \sum_i \lambda_i\, g_i \right\rVert_2^2, \qquad g \leftarrow \sum_i \lambda_i^\star g_i.$$

The resulting $g$ is a common descent direction for all objectives (stationarity
$\Leftrightarrow \lVert g\rVert=0$). The QP is solved by Frank–Wolfe in
`_solve_mgda_qp`.

**Reference.** Sener & Koltun, *Multi-Task Learning as Multi-Objective
Optimization*, NeurIPS 2018.
[arXiv:1810.04650](https://arxiv.org/abs/1810.04650). Underlying MGDA: Désidéri,
C. R. Math. 350(5–6), 2012.

---

## FDFL-NashMTL — Nash Bargaining for Multi-Task Learning

- **Bucket:** Integrated · dynamic
- **Code key:** `mo_method="nashmtl"`
- **Entry point:** `NashMTLHandler`, `fair_dfl/algorithms/mo_handler.py`

**Description.** Choose per-objective weights $\alpha_i \ge 0$ as the Nash
bargaining solution,

$$\alpha^\star = \arg\max_{\alpha}\ \textstyle\prod_i \alpha_i \quad \text{s.t.}\ G\alpha \in B,$$

where $G_{ij} = g_i^\top g_j$ is the Gram matrix and $B$ is the unit ball.
Equivalently, find the direction inside the feasible set that maximizes the
geometric mean of per-task improvements. Solved by Frank–Wolfe
(`_nash_frank_wolfe`); per-task weights are renormalized every
`update_weights_every` steps.

**Reference.** Navon, Shamsian, Achituve, Maron, Kawaguchi, Chechik & Fetaya,
*Multi-Task Learning as a Bargaining Game*, NeurIPS 2022.
[arXiv:2202.01017](https://arxiv.org/abs/2202.01017).

---

## FDFL-CAGrad — Conflict-Averse Gradient Descent

- **Bucket:** Integrated · dynamic
- **Code key:** `mo_method="cagrad"`
- **Entry point:** `CAGradHandler`, `fair_dfl/algorithms/mo_handler.py`; hyperparameter $c \in [0,1]$ = `mo_cagrad_c` (default 0.5)

**Description.** Solve for a descent direction $g$ that maximizes the worst-case
per-task improvement while staying within $c$ times the norm of the average
gradient,

$$g = \arg\max_{g}\ \min_i\ \langle g, g_i\rangle \quad \text{s.t.}\quad \lVert g - \bar g\rVert_2 \le c\, \lVert \bar g\rVert_2,$$

where $\bar g = \tfrac{1}{M}\sum_i g_i$. Solved by the dual QP in
`_solve_cagrad_qp`. The hyperparameter $c$ trades average-direction fidelity
($c=0$: exactly $\bar g$, i.e. weighted sum) against the worst-task guarantee
($c=1$: closer to MGDA's minimum-norm direction).

> [!NOTE]
> **Cross-experiment behavior.** Strong on healthcare at $\alpha=2.0$ (regret
> 0.128), but on the MD synthetic family it records high MAD (0.94–1.53) and loses
> seeds at $\text{imb} \ge 0.6$.

**Reference.** Liu, Liu, Jin, Stone & Liu, *Conflict-Averse Gradient Descent for
Multi-Task Learning*, ICLR 2021.
[arXiv:2110.14048](https://arxiv.org/abs/2110.14048).

---

## Cross-cutting implementation facts

- **`_combine_prediction_gradients`** consults `gradient_merge` only when both
  `use_dec` and `use_pred` are true. Methods with `use_pred=False` (FDFL, DFL) take
  the dec-only branch $g_{\text{dec}} + \lambda\, g_{\text{fair}}$ unconditionally.

- **Multi-objective handlers bypass the merge entirely.** When `mo_method` is set,
  `train_single_stage` dispatches to the handler's `compute_direction` with the
  per-objective parameter-space gradients, and `gradient_merge` is ignored.

- **Decision-gradient backends.** Healthcare uses the closed-form $\alpha$-fair
  gradient of Proposition 1 (`fair_dfl/decision/strategies/analytic.py`); the MD
  knapsack uses `cvxpylayers` for the conic decision gradient.

- **Seed-dropping convention.** Runs whose final regret exceeds 5 *or* final
  fairness violation exceeds 10 are dropped from the per-cell aggregate. The
  high-imbalance MD blow-ups (mainly `seed=33` at $\text{imb} \ge 0.6$) affect raw
  FDFL-Scal *and* the guided baselines (DFL, FDFL, MGDA) on the same splits — they
  reflect the regime, not the scalarization choice.

---

*Last updated 2026-06-02.*
