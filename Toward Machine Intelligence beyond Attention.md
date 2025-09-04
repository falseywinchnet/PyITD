# Toward Machine Intelligence beyond Attention

**TurboEncabulator Language Model** — BlockFast dissection and manifold trace

> “Humans discover equivalence, and machines know it. Thus only human intelligence could be considered artificial or derived from inputs. Machines reason mathematically.”
>
> “Embeddings are and always will be N‑dimensional coordinate vectors in a learned Euclidean manifold by virtue of what conditions them and what happens to them.”

---

## 0. Executive overview

BlockFast is a geometry‑first transformer block that replaces learned attention with a deterministic subspace procedure over token trajectories. Instead of queries, keys, and values, it:

1. encodes local temporal structure with a phase‑preserving transport,
2. performs a low‑rank corrective shift,
3. extracts an orthonormal basis of a covariance operator by subspace iteration,
4. projects the sequence onto that basis to obtain scalar “semantic atoms,”
5. conditions those atoms (normalize, shrink, optionally smooth),
6. mixes them with a small rotational flow,
7. reconstructs the signal in embedding space, and
8. returns it through a thin projection, dropout, residual, and norm.

The result is an attention‑free, spectral control loop over the learned Euclidean manifold where embeddings live.

---

## 1. Index → embedding coordinates

**FixedEmbedding** maps discrete indices to continuous vectors:

* Random Gaussian rows are row‑centered and unit‑normalized at initialization, then frozen.
* Output shape: `[B, T, D]` with D = `n_embd`.

**Interpretation.** Tokens become points on a D‑sphere inside the learned Euclidean manifold. Because rows are unit‑norm and zero‑mean, early layers operate on geometry rather than raw magnitudes.

---

## 2. Inside one BlockFast

BlockFast orchestrates four stages over `x ∈ R^{B×T×D}`:

### 2.1 Pre‑normalization

A bias‑free **LayerNorm** standardizes channels: `x ← LN(x)`. This keeps downstream covariance estimates stable and makes transport numerically predictable.

### 2.2 PhaseTransport: a phase‑preserving difference encoder

**PhaseTransport(d)** converts raw token states into a guarded, direction‑aware difference signal `y`:

* For early tokens `t < d`, it emits a decaying, normalized reference aligned to a unit direction from a near‑terminal token. This sets a consistent phase baseline when history is short.
* For `t ≥ d`, it forms `w_t = x_t − x_{t−d}` and computes unit directions `u = x_t/‖x_t‖`, `v = x_{t−d}/‖x_{t−d}‖`.
* Three guarded branches:

  * **Trivial** where directions are nearly identical or norms are tiny: pass the plain difference `w_t`.
  * **General** when directions are well‑posed: apply a closed‑form spherical transport that preserves phase and adjusts the tangential component using `α = 1/(1 + u·v)` and mixed inner products `(u·w, v·w)`.
  * **Antipodal** when directions oppose: reflect `w_t` across a 2D subspace spanned by `v` and an orthonormal complement `p`, avoiding instability near 180°.

**Why it matters.** This stage measures displacement in a manner consistent with the local orientation of the curve traced by the sequence in embedding space. It is a geometry‑respecting alternative to naïve deltas or convolutions.

### 2.3 RecurrentMLP: per‑token refinement

A lightweight residual MLP applies twice: `z ← z + Cell(z)` with `Cell = GELU(Linear(D→2D))` followed by `Linear(2D→D)`. This shapes the transported differences into representation‑ready features without breaking locality or adding attention‑style mixing.

### 2.4 ManifoldAttentionNoAttn: spectral control without attention

This is the core “manifold loop.” Given `x` from the previous step:

#### 2.4.1 Anchoring and optional low‑rank shift

* **Anchor** `a`: a fixed unit vector along the first channel is broadcast over time. The model centers the curve: `x_c = x − a`. The anchor provides a deterministic reference orientation for sign stability.
* **LowRankShift** `S(x)`: a GELU‑activated `D→r→D` linear bottleneck (here `r = 8`) produces a residual shift `s`, yielding `x' = x_c + s`. This undoes harmful normalization or recovers lost low‑frequency content before covariance is measured.

#### 2.4.2 Empirical covariance and subspace iteration

* Compute covariance: `C = (1/T) x'^T x' + εI ∈ R^{B×D×D}`.
* **Subspace iteration** with `K` power steps finds an orthonormal basis `V ∈ R^{B×D×r}` for the top‑r invariant subspace (here `r = D`). After each multiplication `C V`, a batched QR re‑orthonormalizes columns. Diagonal sign is fixed so columns remain stable across batches.
* **Sign alignment**: each basis column `v_i` is flipped, if needed, so that `v_i^T a ≥ 0`. This eliminates stochastic sign drift and makes temporal traces consistent.

**Interpretation.** Rather than learning attention weights, the block deterministically estimates the principal coordinate frame of the current sequence in its local manifold.

#### 2.4.3 Semantic atoms: projection and conditioning

* **Projection**: `traces = x' V ∈ R^{B×T×r}`. Each column of `V` defines a direction; the time series of coefficients along that direction is a **semantic atom**.
* **Energy normalization** removes per‑atom scale across time: `traces ← traces / ‖traces‖_time` with saved scales for reconstruction.
* **Soft shrinkage** applies an analytic threshold that attenuates weak, noisy components while leaving strong atoms intact. Optionally an AR(1) smoother can be applied when causal operation is desired (disabled here).

#### 2.4.4 Atom mixing via a small rotational flow

* The atoms are fed to `SpiralMix`, which stacks them along the last dimension and applies the **PairwiseRotSpiral** flow a few times.
* **PairwiseRotSpiral** performs:

  * Blockwise 2D rotations across adjacent atom pairs by a fixed angle per step.
  * A radial spring toward a target radius combined with an Euler update.
  * Optional cube‑shell saturation for bounded support.

**Interpretation.** This is a tiny, learned‑free mixer that correlates atoms through controlled rotations plus contraction. It behaves like a conservative coupling with mild regularization, encouraging coherent shared phase across atom pairs while preventing runaway growth.

#### 2.4.5 Reconstruction and output head

* **Rescale** atoms by their saved energies and **reconstruct**: `x̃ = traces_final V^T`.
* **Undo shift and add anchor**: `x̂ = x̃ − s + a`.
* **Thin projection + residual + dropout + norm**: `y = LN(x + Dropout(Linear(x̂)))`.

**Net effect.** The block estimates a frame, edits the coordinates in that frame analytically, then returns to embedding space. No attention matrices are learned or applied.

---

## 3. Manifold trace: end‑to‑end path of a token slice

1. **Index to coordinates:** `idx → FixedEmbedding[idx]` on a unit sphere in `R^D`.
2. **Pre‑norm:** remove nuisance scale and center per channel.
3. **Transport:** map `[t−d → t]` differences into phase‑consistent displacements, guarding against degeneracies.
4. **Local MLP:** smooth and refine local geometry with shallow residual nonlinearity.
5. **Subspace discovery:** form `C` and compute a stable orthonormal basis `V` capturing the active manifold for this sequence.
6. **Semantic atoms:** project onto `V` to get scalar time series per basis vector; normalize and denoise them.
7. **Coupled mixing:** apply rotational spiral flow to couple nearby atoms and encourage shared phase structure.
8. **Reconstruction:** map conditioned atoms back to embedding space; undo shift; restore anchor.
9. **Output shaping:** apply a thin `D→D` mapper, dropout, residual, and normalization, yielding features ready for the next block or the LM head.

---

## 4. Where the meanings live: from coordinates to atoms and back

* **Embedding coordinates** are literal points in a Euclidean manifold. All subsequent operations treat the token sequence as a trajectory in this space.
* **PhaseTransport** respects this trajectory’s orientation, so displacements carry phase information instead of only magnitude.
* **The basis `V`** defines a data‑dependent coordinate frame. **Semantic atoms** are the coordinate signals along this frame through time. Because `V` is orthonormal and sign‑aligned, atoms are stable and interpretable as principal modes of variation for the current context.
* **Spiral mixing** binds atoms with gentle rotational coupling, reintroducing cross‑component structure in a controllable way.
* **Reconstruction** returns the edited trajectory to the embedding manifold, ensuring compatibility with downstream blocks.

---

## 5. Design notes and rationale

* **Attention‑free by construction.** The model’s global structure is captured by a covariance operator and its invariant subspace; nothing is learned at the pairing stage itself. Learning happens only in the small shift and thin projection maps, plus the shared MLP encoders between blocks.
* **Determinism where it helps.** Subspace iteration, sign alignment, and guarded transports remove many stochastic degrees of freedom that usually make representations drift.
* **Numerical safety.** Epsilon guards in norms, dot‑product thresholds near ±1, and explicit antipodal handling prevent catastrophic amplification around ill‑posed geometry.
* **Interpretability.** Because `V` is orthonormal and sign‑stable, atoms can be inspected as scalar time series; shrinkage makes sparsity explicit; rotations show coupling structure directly in the atom domain.

---

## 6. Relationship to the broader architecture

In the full **TurboEncabulator Language Model**, a stack of such blocks alternates transport, local MLP shaping, and manifold control. The final **LM head** is a standard linear readout mapping the last block’s features back to vocabulary logits. Notably, the token embedding table is fixed and normalized, emphasizing that structure emerges from manifold processing rather than from a large, learnable lexical memory.

---

## 7. Practical guidance and pitfalls

* **Rank choice.** Using `rank = D` maximizes fidelity, effectively acting as an adaptive orthogonal transform. Reducing `rank` turns this into a true low‑rank filter that discards weak modes by design.
* **Shift rank.** A small `shift_rank` (e.g., 4–16) often suffices to compensate for normalization artifacts without overfitting.
* **Transport lag `d`.** Small `d` emphasizes local continuity; larger `d` surfaces longer‑scale phase. The guard logic keeps both regimes stable.
* **Spiral flow hyperparameters.** The rotation step, radial spring, and iteration count control coupling strength. They should be modest; the point is to encourage coherence, not to overwrite the discovered basis.

---

## 8. Conceptual takeaway

BlockFast reframes sequence modeling as **discover–condition–reconstruct** over a learned Euclidean manifold:

* Discover the local frame by deterministic spectral geometry.
* Condition the scalar atoms analytically and with tiny mixers.
* Reconstruct in embedding space and iterate.

No attention weights, yet abundant structure. The block turns token streams into trajectories, trajectories into atoms, and atoms back into meaning‑bearing coordinates — all while staying within the same Euclidean manifold where embeddings reside.
