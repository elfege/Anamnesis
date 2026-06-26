# On the Bitter Lesson — Reference Notes for Anamnesis-δ²

> Personal working notes consolidating the discussion around Sutton's
> "The Bitter Lesson" (March 2019) and what it means for the δ² project.
> Source: `docs/bitter_lesson.txt` (cached copy of the original essay).

---

## 1. Sutton's Argument in One Paragraph

The biggest lesson from 70 years of AI research: **general methods that
leverage computation are ultimately the most effective, and by a large
margin.** Hand-engineered domain knowledge ("how we think we think") helps
in the short term, plateaus in the long run, and gets superseded by
brute-force search + learning at scale.

Examples Sutton walks through: chess (Deep Blue's massive search beat
hand-coded chess knowledge), Go (AlphaGo's self-play learning beat
human-pattern-encoding programs), speech recognition (statistical
HMMs + later deep learning beat phoneme/vocal-tract knowledge), computer
vision (deep convolutional nets beat SIFT, edge detection, generalized
cylinders, etc.).

The two methods that scale arbitrarily with compute are **search** and
**learning**. Everything else is local optimum.

---

## 2. Why This Threatens Anamnesis-δ²

A skeptical NeurIPS reviewer with this essay bookmarked will reach for it
within 30 seconds of reading our abstract. The threat:

- δ² is hand-engineered structure injected into the optimizer
- It bakes a **categorical taxonomy** (4 negation types) into the update rule
- The motivation is explicitly **philosophical** (Hegelian negation, dialectics)
- It is **not a compute-scaling technique** — it's a structure-imposition technique

By Sutton's lens, δ² is exactly the research pattern he warns against:
satisfying to the researcher, helpful in the short term, plateaus as
compute scales. If we claim δ² competes with GPT-5 on benchmarks, we
lose. Scale wins. That's not in dispute.

---

## 3. Three Defensive Positions

There are three honest counter-positions. Pick one and commit.

### Position #1 — "Adam also violates the Bitter Lesson"

Adaptive learning rates, momentum, weight decay, gradient clipping — all
hand-engineered structure. The ML literature is full of Bitter-Lesson
violations that got published and adopted because they **worked
empirically at scale**. δ² is in that lineage. Doesn't win the argument
in principle — wins it with WikiText-103 numbers. **Weak defense but
honest.**

### Position #2 — "δ² isn't competing on benchmark quality — it's competing on continual adaptation"

**This is the strongest position.**

**What Sutton is actually arguing:** *"To get better performance on
benchmark X, throw more compute and data at it."* Notice the problem he's
addressing: how to maximize benchmark score with unlimited compute.

**A different problem his essay does NOT address:** *"You have ONE model.
It's already trained. You cannot retrain it from scratch. New
information keeps arriving (user feedback, new documents, corrections).
How do you update this model without it forgetting what it already knows?"*

This is **continual learning** (lifelong learning). It's a well-known
hard problem because of **catastrophic forgetting**: standard SGD on new
data destroys old knowledge. Known bug since the 1980s.

GEM (2017) and EWC (2017) were published on exactly this problem. They
didn't claim to beat SOTA benchmarks — they claimed incremental update
without forgetting. Sutton's essay doesn't argue against this work,
because scale-from-scratch is not an option in the continual-learning
setting.

**δ² fits here naturally.** The bassin is *literally* a mechanism for
retaining knowledge that standard gradient descent would discard. That's
the continual-learning problem.

**The practical move:** restructure the README and the paper to lead
with continual learning. Currently we present:

1. δ² is a new optimizer (Bitter-Lesson-vulnerable)
2. Oh also it has this near-RT learning trick

Swap them:

1. δ² is a continual-learning primitive for models that adapt from streaming feedback
2. As a side benefit, it also works for standalone training

Same code. Different frame. The second frame doesn't trigger the Bitter
Lesson objection because we're not competing on "pretraining quality"
— we're competing on "adaptation quality at fixed model size," a
legitimate open problem with its own literature.

### Position #3 — "The bassin + switching controller IS search, at the optimizer level"

**What Sutton loves:** search. Chess searches moves, Go searches board
positions. The more compute, the deeper/wider the search. This is what he
means by "general methods that leverage computation."

**What traditional optimizers look like:** ONE fixed rule that updates
weights every step. Adam is a rule. SGD is a rule. δ² is a rule. The rule
never changes — same formula every step regardless of context.

**What a controller-based optimizer looks like:**

```
at every training step:
    look at the current situation (loss, gradients, entropy, etc.)
    → decide which rule to use right now
    → apply that rule
    → observe result
    → update the decision logic
```

That decision process is **search over update rules**, not a fixed rule.
Every step asks: "of the tools I have (Adam, δ², SGD, something else),
which one fits this moment?"

A controller that learns which rule to use when is *itself* a learning
system that scales with compute. More training data → better controller
decisions → better performance. Sutton-compatible architecture.

Under this framing, δ² isn't the whole system — it's **one primitive**
that the controller can dispatch to, alongside Adam. The system becomes
a **meta-optimizer**, not a fixed rule.

**Why the README currently doesn't do this:** the "Composition with Adam"
section *talks about* a controller in pseudocode, but the actual code in
`d2/` implements δ² standalone. Talk is all it is. To deploy this
defense, the controller has to be built — feature extraction from
training state, decision logic (initially `if/else`, later learnable),
dispatch to primitives.

**Recommendation:** commit to #2 first (framing change, mostly README
rewrite). Build the controller and extend to #3 in a follow-up.

---

## 4. GEM and EWC — The Closest Cousins

**Neither is a model. Both are techniques** — modifications to how you
train an existing model. Architecture stays the same. Training loop
changes.

### EWC — Elastic Weight Consolidation (Kirkpatrick et al., 2017, DeepMind)

**Plain idea:** *"After learning Task A, figure out which weights mattered
most. When you learn Task B, penalize changes to those important weights."*

**Mechanically:**

1. Finish training on Task A. Weights are now `W*`.
2. Compute the **Fisher Information Matrix** `F` — for each weight, how
   confident is the model in that weight? (Confidence ≈ "would moving
   this weight a little hurt the loss a lot?")
3. Train on Task B with a modified loss:

```
L_total = L_taskB + λ · Σᵢ Fᵢ × (Wᵢ − W*ᵢ)²
                       ^^^^^^^^^^^^^^^^^^^^^^^^
                       penalty: don't drift too far on weights
                       that mattered for Task A
```

Important-to-A weights act *elastic* — pulled back toward `W*`.
Unimportant weights are free to change. That's the whole trick. SGD on a
modified loss.

**Direct relevance:** δ₁ as currently defined (`F(W)^(1/2) ⊙ (W − W̄)`) is
**functionally close to EWC's penalty term**. A reviewer will spot this
immediately. Required move: cite EWC as the parent of δ₁ and explain
what's different (signed squaring, additive injection rather than
quadratic penalty, integration with δ₂).

### GEM — Gradient Episodic Memory (Lopez-Paz & Ranzato, 2017, FAIR)

**Plain idea:** *"Keep a small buffer of past examples. Before applying any
gradient update, check that it doesn't hurt past performance. If it would,
modify the gradient."*

**Mechanically:**

1. Maintain memory buffer `M` of examples from past tasks.
2. At each step, compute gradient `g` for the current task.
3. Compute gradients `g_k` for each old task using examples from `M`.
4. Inner-product test: if `g · g_k < 0`, the new gradient would hurt task `k`.
5. If yes, project `g` onto the closest direction that doesn't hurt any
   old task. Apply that projected gradient instead.

Gradient surgery — keep moving forward on Task B but never in a direction
that actively damages knowledge of Task A.

**Direct relevance:** the memory buffer `M` is structurally similar to
the bassin. GEM stores *raw past examples*; δ² stores *processed past
frictions*. GEM uses them as **constraints** (don't go in directions that
hurt them); δ² uses them as **injections** (add their squared magnitude
to the update). Same starting point — past gradient information shouldn't
be discarded — different consumption pattern.

### Standard Benchmarks for This Family

This is the league δ² is actually in. Not "beat GPT-5 on perplexity." It's:

- **Split-MNIST / Split-CIFAR** — divide a dataset into N tasks, train sequentially
- **Permuted-MNIST** — apply different pixel permutations as "different tasks"
- **Continual language modeling** on segmented WikiText subsets

Metrics measured:

- **Average accuracy** across all tasks after sequential training
- **Backward transfer (BWT)** — how much old-task performance dropped
  after learning new tasks (negative = forgetting)
- **Forward transfer (FWT)** — how much old-task knowledge helped new tasks

Comparison is clean, reproducible on a single GPU in hours. No claim of
new general-purpose architecture — just a better recipe for one specific
problem.

---

## 5. What's Actually New in δ² (Honest Catalog)

### Genuinely new (small but real)

1. **Signed squaring as an active update operator** — `sign(δ) ⊙ δ²`,
   direction preserved, magnitude amplified. EWC's `(W − W*)²` is inside
   a *penalty* (no sign needed). GEM doesn't square. Using signed squaring
   as the *added-to-weights* term is original. **Small detail, not paradigm.**

2. **Four-way negation taxonomy applied to gradient retention** — EWC
   uses one continuous Fisher weight; GEM uses one binary "conflicts or
   not." δ² uses four discrete categories (inessential / essential
   difference / opposition / annihilation) to structure what gets stored
   and how it's weighted. **Whether the structuring move beats simpler
   alternatives empirically — open question.**

3. **Inverting the Adam-family accumulator's purpose** — "same EMA of
   squared gradients, used to inject rather than dampen." Adam dampens.
   δ² adds. **Reframing is genuinely original; the math machinery is
   borrowed.**

### NOT new

- **The bassin, structurally** — experience replay (DQN 2013) + GEM's
  memory buffer. Around for 12+ years.
- **δ₁ (Fisher-weighted drift from baseline)** — that *is* EWC, almost
  line for line.
- **Continual learning as the problem** — entire field, decade of NeurIPS
  publications.
- **Uncertainty-gated retrieval at inference** — that's RAG with a
  different trigger.
- **Motivation from negation/contradiction** — Hinton has used "this
  neuron disagrees with that neuron" framings for decades. Philosophical
  vocabulary is ours; underlying intuition is in the air.

### Honest characterization

δ² is **a specific recombination of existing continual-learning ideas
with one new technical detail (signed-squared injection) and a structural
taxonomy on top.** Lands at a NeurIPS workshop on continual learning if
empirics hold up — *"another data point in this active subfield,
motivated through an unusual lens."*

It is **not** a new optimizer family. It is **not** a Bitter-Lesson-defying
breakthrough. It is **not** something Kirkpatrick would read and say "I
never thought of that."

What it could realistically achieve:

- Workshop paper if benchmarks are competitive with EWC/GEM
- arXiv preprint regardless
- An interesting *engineering* artifact (the bassin + controller
  infrastructure) that makes continual learning easier to deploy in practice
- A genuinely original framing that could attract ML/philosophy bridge people

What it almost certainly won't achieve:

- Main-track NeurIPS / ICML
- "Inspired by Hegel" lede in any major venue
- Winning against well-tuned EWC + replay on standard benchmarks (it
  might, but the prior is against)

---

## 6. The Real Project Value

The project's actual value isn't the optimizer alone. It's the integration:

- A working multi-machine ML infrastructure (Anamnesis app + crawler +
  trainers + workers + memory)
- A specific framing of continual learning that ties feedback → bassin
  → deferred update
- A philosophical lens that produces new questions even if not new answers
  (e.g., "does taxonomy of negation matter?" — that's testable)
- Engineering taste that ties research-grade ideas to deployable systems

That combination — engineer who ships infra + philosopher who poses
unusual questions — is genuinely rare in the field. The mistake would be
selling δ² as the headline. The headline is the **platform that contains
δ² as one component.** That framing is the one that's hardest to push back
against, because it's not a competitor to EWC/GEM — it's a context that
uses them.

---

## 7. The Trap to Avoid

**Don't write a paper titled "A Hegelian Alternative to Gradient
Descent."** That frames it as knowledge injection. Gets Bitter-Lessoned
in review.

**Write one titled "Structured Negative Retention for Continual Learning:
A Comparison with GEM and EWC on WikiText-103."** Frames it as a
continual-learning contribution with philosophical motivation tucked into
the methodology section. Different reception entirely.

---

## 8. The Failure Mode to Watch For

When an engineer challenges you on engineering, the temptation is to
reach for philosophy because that's the home turf. When a philosopher
challenges you on philosophy, you'd probably reach for engineering. It's
a mode of self-defense.

If you don't notice it, it slowly turns the philosophical work into an
unfalsifiable shield around the engineering work, and the engineering
into a weak demo for the philosophy. **Both get worse.**

The fix is mechanical: before sending anything technical to a technical
audience, scan for paragraphs drifting into Hegel/dialectics/PhD
methodology and ask "did they ask?" If no, cut. Philosophy stays in the
drawer where the audience opted in (essays, the d2 README *theory*
section, conversations with people who know the background). It does not
appear in replies to ML researchers reviewing the update rule.

---

## 9. The Bottom Line

The Bitter Lesson does apply, and it has to be answered. Not
rhetorically — with results.

- **If WikiText-103 benchmarks show δ² matches Adam on loss**: publishable
  as "alternative with comparable performance, interesting interpretability."
- **If δ² outperforms Adam on continual-learning benchmarks (streaming data,
  multiple tasks, forgetting resistance)**: publishable as "novel
  continual-learning method."
- **If δ² underperforms Adam on standard benchmarks**: don't publish the
  current framing. Fall back to position #2 (continual adaptation) and
  build for that specifically — or abandon the optimizer claim and keep
  the bassin as a RAG-for-training mechanism.

The project has to earn its structure by showing it does something compute
alone doesn't do. Until that empirical result exists, the philosophy is
a motivation, not a contribution.

---

## 10. Heuristic vs Epistemogenetic — Where δ² Sits

The project can be framed at two different levels:

**Heuristic-side** (what's actually built today): the bassin + controller
is a **heuristic for choosing when to descend vs grow**. A tool for
finding solutions inside an existing problem space. EWC, GEM, SAM are
all heuristic in this sense. So is Adam. So is most of ML. δ² as
currently scoped fits cleanly into this register and **the Bitter
Lesson question only applies here.**

**Epistemogenetic-side** (riskier, partially built): the system
recognizes the limits of *its own* knowledge and triggers an
outward-facing process (dialogue, search, tool call) to fill them. The
"good contradiction vs bad contradiction" move is genuinely
epistemological — bad contradictions (high tension, structurally
unresolvable internally) become triggers for reaching outward.

**Recommendation:** for ML audiences (Jan, NeurIPS reviewers), lead with
the heuristic frame. The epistemogenetic claim is bigger-than-ML and
deserves a different audience.

---

## 11. Fixed Taxonomy, Learned Distribution (load-bearing clarification)

A confusion to preempt: the four categories of negation
(inessential difference, essential difference, opposition, annihilation)
are **fixed structural categories**, not labels the system learns from
data. They are *a priori* in the strong Hegelian sense. **The taxonomy
itself does not evolve.**

What *does* accumulate and shift over training:

- the specific frictions stored in the bassin
- their tension scores
- their semantic context (what input region produced them)

Concretely:

> A region of weight space that initially produces "essential difference"
> frictions might, after enough accumulated bassin entries, start
> producing "annihilation" frictions in a recurring pattern.

That recurring pattern signals something to the controller — **not**
"I need a new category" but **"this region is structurally unstable,
the dialogue/search trigger should fire here."**

So the system does **not** invent new categories. It learns *where* in
its weight space each existing category tends to fire, and uses that
map to drive search/dialogue. **Categories fixed; distribution over
categories learned.** That's coherent with the Hegelian framework
(categories necessary, not contingent) while still being adaptive
behaviorally.

**Why this matters operationally:**

- The negation classifier (in `d2/bassin.py`) maps each discarded gradient
  to one of four fixed bins by signed inner-product / magnitude tests
- The bassin's index is over (region_in_weight_space, category) pairs
- The controller reads density patterns from this index — not as
  category-revision signals, but as routing signals for what to do next

**Why this matters epistemologically:**

The *fully* epistemogenetic version (system invents new categories from
its own data) is a separate, anti-Hegelian project. It would step
outside the framework entirely — empiricist rather than dialectical.
That is **not** what δ² is doing. The categories are given; their
*spatial distribution* is what the system learns about itself.

**The contrast with Piaget:**

Piaget's "categories" are mental schemas the child *constructs* through
interaction with the environment — they evolve via accommodation when
new experiences don't fit existing structures. Examples of Piagetian
schemas / categories:

- **Object permanence** — the schema that objects continue to exist when
  out of sight (constructed during the sensorimotor stage, ~8 months)
- **Conservation of liquid** — the schema that pouring water into a
  different-shaped container doesn't change the amount (concrete
  operational stage, ~7 years)
- **Class inclusion** — the schema that all dogs are animals but not all
  animals are dogs (concrete operational stage)
- **Seriation** — the schema for ordering objects by a continuous property
- **Reversibility** — the schema that mental operations can be reversed
- **Hypothetico-deductive reasoning** — the formal-operational schema
  for testing propositions against possible counterfactuals

Piaget's categories *develop* — they are constructed, contingent,
revisable through experience. **Hegel's categories of the Logic do
not.** Being, Nothing, Becoming, Quality, Quantity, Measure, etc., are
necessary derivations of pure thought. They are not learned from
experience and they do not change based on it.

The δ² taxonomy of negation is in the **Hegelian register, not the
Piagetian one.** The system does not Piagetian-accommodate to invent
new categories. It Hegelian-classifies experience into pre-given
necessary categories, while learning a contingent map of *where* in its
own state space each category fires.

This distinction needs to be in the formal addendum so a careful reader
doesn't misread δ² as a constructivist / Piagetian system. It isn't.

---

*Last updated 2026-04-23. Source: discussion with d² Claude instance,
Sutton-essay critique thread. Original essay: `bitter_lesson.txt` (this
directory). Cited prior work: EWC (Kirkpatrick et al. 2017), GEM
(Lopez-Paz & Ranzato 2017), SAM (Foret et al. 2020), Adam (Kingma & Ba
2015), Experience Replay (Lin 1992), Piaget (genetic epistemology, e.g.
*La construction du réel chez l'enfant*, 1937).*
