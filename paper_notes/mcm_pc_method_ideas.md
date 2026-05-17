# MCM-PC Paper Notes and Method Ideas

## Current Working Title

**MCM-PC: Multi-Cache Matrix for 3D Point Cloud Test-Time Adaptation**

---

## Core Narrative

The project starts from Point-Cache and studies how to make test-time cache construction more reliable under 3D point-cloud corruptions.

The key discovery so far:

1. Entropy-only positive cache admission is insufficient.
2. Entropy-margin reliability improves some cases but is unstable.
3. Global-local consistency should not be used to promote positive cache samples.
4. Global-local conflict is highly useful for detecting wrong global pseudo-labels.
5. Therefore, the main innovation should move toward conflict-aware negative / boundary suppression.

---

## Candidate Main Innovation

**Confusion-Aware Negative Cache**

Instead of using local evidence as a positive label correction, use it to detect when the global pseudo-label should be suppressed.

Core idea:

\[
\hat y_g \ne \text{local evidence}
\Rightarrow
\hat y_g \text{ is likely unreliable}
\]

E4-DIAG shows that conflict samples have much higher global pseudo-label error probability.

---

## Method Section Draft Note

### Section: Method / Conflict-Aware Negative Cache

Global-local disagreement is not treated as a positive correction signal because the local alternative class is often unreliable. Instead, we use disagreement to identify potentially wrong global pseudo-labels and construct a conservative negative suppression signal.

---

## Experiment Section Draft Note

### Section: Ablation Study

E2-EMR and E3-GLC show that directly improving positive cache admission is not sufficient. In particular, global-local consistency as a positive reliability term fails to improve the full corruption average. This motivates using global-local conflict for negative suppression.

---

## Diagnostic Result Draft Note

### Section: Diagnostic Analysis

Across all ModelNet-C corruptions, global-local conflict significantly increases the probability of wrong global pseudo-labels. However, the local alternative class is rarely correct. This supports the use of conflict as a negative suppression signal rather than a positive pseudo-label.

---

## Important Warning

Do not write:

> local alternative class is used as the corrected pseudo-label.

Write:

> local alternative evidence is used only to suppress unreliable global predictions.

---

## Next Experiment

**E4-CANC-v0: Conservative Conflict-Aware Negative Cache**

Implementation direction:

\[
I_{neg}^{E4}(x)=I_H(x)\lor(I_M(x)\land I_D(x))
\]

Use this to update or supplement the negative cache.

