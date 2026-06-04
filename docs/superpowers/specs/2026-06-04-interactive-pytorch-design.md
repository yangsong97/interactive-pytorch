# Interactive PyTorch — Design Spec

- **Date:** 2026-06-04
- **Status:** Approved (pending spec review)
- **Author:** Yang Song
- **Repo (planned):** `yangsong97/interactive-pytorch`
- **Live URL (planned):** https://yangsong97.github.io/interactive-pytorch/

## 1. Goal

An interactive website that teaches PyTorch from the ground up to someone
comfortable with Python but new to machine learning. The site combines
**in-browser interactive visualizations** of core concepts with **"Open in
Colab" buttons** that let learners run the real PyTorch code for free. It ships
as a static site deployed to GitHub Pages.

## 2. Target audience

- Comfortable with Python basics (functions, loops, lists, imports).
- New to ML: tensors, gradients, neural networks are unfamiliar.
- Wants to understand the *why*, not just copy code.
- No local install required — real code runs in Google Colab.

## 3. Core decision: what "interactive" means

PyTorch is a Python library with native C++/CUDA extensions; it **cannot run in
a browser** (even Pyodide does not ship PyTorch). The chosen model:

- **Concept interactivity** is delivered by custom JavaScript/Canvas widgets
  that visualize and animate the ideas (tensors, autograd, gradient descent,
  training, convolution).
- **Real execution** is delivered by **"Open in Colab"** buttons on each lesson,
  linking to runnable notebooks with the actual PyTorch code.

This keeps the site fully static (free hosting, fast, no backend) while still
giving learners a path to run genuine PyTorch.

## 4. Tech stack & architecture

- **Astro** — static site generator. Outputs plain HTML/CSS/JS. Purpose-built
  for content-heavy sites with embedded interactivity ("islands architecture":
  prose ships as static HTML, only widgets ship JS).
- **Svelte islands** — the interactive widgets. Svelte's reactive model makes
  "slider → recompute → redraw canvas" clean and low-bug.
- **Markdown/MDX** — lesson content lives as Markdown files in an Astro content
  collection, so a large curriculum stays maintainable. MDX is used where a
  lesson embeds a widget.
- **Plain CSS design tokens** — a restrained **monochrome / muted** palette via
  CSS custom properties + Astro scoped styles. No CSS framework (avoids version
  churn, keeps source readable).
- **Pinned dependency versions** + the official Astro GitHub Pages Action for a
  robust first deploy.

### Build & deploy flow

```
push to main ──> GitHub Actions ──> astro build ──> GitHub Pages
```

Astro config sets `site: 'https://yangsong97.github.io'` and
`base: '/interactive-pytorch'` so asset paths resolve under the project-pages
subpath.

## 5. Site structure

- **Home** (`/`) — what this is, "no install needed — runs in Colab", a clear
  "Start learning" call to action, and a curriculum overview.
- **Persistent sidebar nav** — lessons grouped by module; highlights current
  lesson; collapses on mobile.
- **Lesson pages** (`/lessons/<slug>`) — the workhorse (anatomy in §7).
- **How to use this site / Colab 101** (`/start`) — 60-second guide to running
  cells in Colab.

## 6. Curriculum (comprehensive skeleton)

Bolded concepts get a bespoke interactive widget (§7). All lessons get prose +
real PyTorch code + a Colab button.

- **M1 · Tensors** — create; shape/dtype/device; indexing & slicing; reshaping
  (`view`/`reshape`, `squeeze`/`unsqueeze`); math & **broadcasting**; NumPy bridge.
- **M2 · Autograd** — the gradient idea; `requires_grad` / `backward()` /
  `.grad`; **computation graphs**; `no_grad` / `detach`.
- **M3 · Optimization** — loss functions; **gradient descent** intuition;
  optimizers (SGD, Adam); the training-loop pattern.
- **M4 · Neural networks** — neurons & **activation functions**; `nn.Module`;
  a first MLP; **training it on real data** (2D classification).
- **M5 · Data** — `Dataset` / `DataLoader`; transforms; batching & shuffling.
- **M6 · CNNs** — **convolution** intuition; pooling; an image classifier.
- **M7 · Training well** — overfitting; regularization (dropout, weight decay);
  LR scheduling; **GPU** (`.to(device)`).
- **M8 · Beyond** — saving/loading (`state_dict`); transfer learning; where to
  go next.
- **Capstone** — end-to-end image classifier in Colab.

### Depth strategy (scope control)

v1 ships the **full structure** end-to-end. The early/core modules (M1–M4 and
the M6 convolution lesson) are fully fleshed out with bespoke widgets. Advanced
lessons (M5, M7, M8) ship as strong written lessons + runnable Colab notebooks,
with widgets added later. This delivers "comprehensive" honestly without faking
depth.

## 7. Interactive "hero" widgets

1. **Tensor Playground** — reshape / slice / broadcast a tensor; watch the grid
   and shape annotation update live.
2. **Autograd graph builder** — assemble a small expression (e.g.
   `y = (x*w + b)^2`); click "backward"; watch gradient values fill in and flow
   backward through the graph.
3. **Gradient descent** — a marker descending a loss surface; tune the learning
   rate and watch it converge, crawl, or diverge.
4. **Activation explorer** — ReLU / sigmoid / tanh / LeakyReLU and their
   derivatives; drag the input to read off values.
5. **Live training loop** *(showpiece)* — a tiny MLP learns to classify 2D
   points (moons/spirals); watch the loss curve drop and the decision boundary
   form in real time; sliders for learning rate, epochs, hidden units, activation.
6. **Convolution visualizer** — slide a kernel over a small image; see each
   feature-map cell compute; switch between edge-detect / blur / sharpen kernels.

All widgets are pure JS/Canvas — no PyTorch required — and each is paired with
the equivalent real PyTorch code + Colab link in its lesson.

## 8. Lesson page anatomy

A reusable lesson layout with these blocks, top to bottom:

1. **Concept prose** — plain-language explanation of the *why*.
2. **Embedded widget** — the relevant visualization (where one exists).
3. **Real PyTorch code** — syntax-highlighted, with a copy button.
4. **"Open in Colab"** — runs that lesson's code in a notebook.
5. **Check your understanding** — 1–2 short prompts with reveal-able answers.
6. **Prev / Next** navigation.

## 9. Learning-mode contribution points

Two spots are scaffolded with context + signature + TODO for the user to fill
(~5–10 lines each). They represent genuine design choices, not busywork. The
user may also defer and have Claude fill them.

- **A. Gradient-descent update rule** (widget #3): implement the parameter
  update step. Choice that shapes behavior: plain SGD (`p -= lr * grad`) vs.
  adding momentum. Affects how the marker moves on the loss surface.
- **B. Forward pass of the tiny net** (widget #5): implement the MLP forward
  computation (linear → activation → linear) that the training loop optimizes.

## 10. Deployment

- Public repo **`yangsong97/interactive-pytorch`**, created via `gh`.
- **GitHub Actions** workflow (`.github/workflows/deploy.yml`) using the
  official Astro Pages steps: checkout → setup Node → `astro build` →
  upload artifact → deploy to Pages. Triggered on push to `main`.
- GitHub Pages source set to **GitHub Actions**.
- Live at **https://yangsong97.github.io/interactive-pytorch/**.

## 11. Non-goals (YAGNI)

- No real PyTorch execution in the browser, and no hosted execution backend.
- No user accounts, progress saving, or analytics in v1.
- No bespoke widgets for M5/M7/M8 in v1 (prose + Colab instead).
- No search, comments, or i18n in v1.

## 12. Success criteria

- Site builds clean and is live at the Pages URL.
- A newcomer can go Home → M1 → … and learn tensors → autograd → training a net
  with concepts that are *seen*, not just read.
- Every code lesson has working, runnable Colab links.
- Lighthouse: prose pages ship little/no JS; widgets hydrate only where used.
- Monochrome/muted theme, responsive, keyboard-navigable.

## 13. Risks & mitigations

- **Astro/Svelte/Pages config friction** → pin versions; use the official Pages
  Action; set `site`/`base` correctly; verify the build locally before pushing.
- **Scope creep ("comprehensive")** → depth strategy in §6; widgets limited to
  the §7 list for v1.
- **Colab links rot / wrong path** → keep notebooks in-repo under `notebooks/`
  and link via the `colab.research.google.com/github/...` URL form so they track
  the repo.
