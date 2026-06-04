# Interactive PyTorch Site — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy a static, interactive website that teaches PyTorch to ML beginners, combining in-browser concept visualizations with "Open in Colab" buttons for real code.

**Architecture:** Astro static site (islands architecture) with Svelte components for interactive widgets and MDX for lesson content. Widget *logic* lives in framework-free, unit-tested TypeScript modules (including a from-scratch scalar autograd engine); Svelte components handle presentation/canvas. Deployed to GitHub Pages via GitHub Actions on push to `main`.

**Tech Stack:** Astro 5, Svelte 5, `@astrojs/mdx`, Vitest (logic tests), plain CSS design tokens (monochrome theme), GitHub Actions + Pages, Python `nbformat` for Colab notebooks.

---

## Conventions

- **Base path:** the site is served from `/interactive-pytorch/`. ALL internal links and asset URLs must be prefixed with Astro's base. Use the helper `withBase()` (Task 1.3) — never hardcode `/lessons/...`.
- **Commits:** no AI attribution lines. Use `git -c user.name="Yang Song" -c user.email="yangsong@alumni.princeton.edu" commit`.
- **Verification:** logic tasks verify via `npx vitest run`. Presentation/content tasks verify via `npm run build` (must exit 0) and the local preview (`npm run dev`).
- **TDD note:** pure-logic modules (`src/lib/*`) are built test-first. Svelte/canvas/content is verified by build success + visual preview, not unit tests.

---

## File Structure

```
interactive-pytorch/
├── .github/workflows/deploy.yml      # CI: build + deploy to Pages
├── astro.config.mjs                  # site/base/integrations
├── vitest.config.ts                  # test runner
├── tsconfig.json
├── package.json
├── public/                           # static assets (favicon, sample image)
├── notebooks/                        # Colab .ipynb files (one per module)
│   └── build_notebooks.py            # generator script (nbformat)
├── src/
│   ├── styles/global.css             # design tokens + base styles (monochrome)
│   ├── lib/                          # FRAMEWORK-FREE, TESTED logic
│   │   ├── tensor.ts                 # shape/broadcast/reshape helpers
│   │   ├── autograd.ts               # scalar autograd Value engine
│   │   ├── activations.ts            # relu/sigmoid/tanh + derivatives
│   │   ├── gradientDescent.ts        # update step  ← CONTRIBUTION POINT A
│   │   ├── mlp.ts                    # tiny MLP forward/train ← CONTRIBUTION POINT B
│   │   ├── conv.ts                   # 2D convolution
│   │   └── data.ts                   # toy datasets (moons) + seeded RNG
│   ├── components/
│   │   ├── ColabButton.astro
│   │   ├── CodeBlock.astro           # (copy button wrapper)
│   │   ├── Sidebar.astro
│   │   ├── Callout.astro
│   │   ├── Quiz.svelte               # "check understanding" reveal
│   │   └── widgets/                  # Svelte islands (presentation)
│   │       ├── TensorPlayground.svelte
│   │       ├── AutogradGraph.svelte
│   │       ├── GradientDescent.svelte
│   │       ├── ActivationExplorer.svelte
│   │       ├── TrainingLoop.svelte
│   │       └── ConvVisualizer.svelte
│   ├── layouts/
│   │   ├── Base.astro                # html shell, theme, fonts
│   │   └── Lesson.astro              # sidebar + lesson chrome + prev/next
│   ├── content.config.ts             # lessons collection schema
│   ├── content/lessons/              # MDX lessons, NN-slug.mdx (ordered)
│   └── pages/
│       ├── index.astro               # home
│       ├── start.astro               # "Colab 101"
│       └── lessons/[...slug].astro   # renders a lesson
└── docs/superpowers/                 # spec + this plan
```

---

## Phase 0 — Walking skeleton: scaffold + deploy an (almost) empty site

> Goal: prove the entire build→deploy→Pages pipeline end-to-end FIRST, so deployment is never a big-bang risk at the end.

### Task 0.1: Minimal Astro scaffold

**Files:**
- Create: `package.json`, `astro.config.mjs`, `tsconfig.json`, `src/pages/index.astro`, `.gitignore`

- [ ] **Step 1: Write `.gitignore`**

```gitignore
node_modules/
dist/
.astro/
.DS_Store
*.log
.vitest/
```

- [ ] **Step 2: Write `package.json`**

```json
{
  "name": "interactive-pytorch",
  "type": "module",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "astro dev",
    "build": "astro build",
    "preview": "astro preview",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "dependencies": {
    "astro": "^5.0.0"
  },
  "devDependencies": {
    "vitest": "^2.0.0"
  }
}
```

- [ ] **Step 3: Write `tsconfig.json`**

```json
{
  "extends": "astro/tsconfigs/strict",
  "include": [".astro/types.d.ts", "**/*"],
  "exclude": ["dist"]
}
```

- [ ] **Step 4: Write `src/pages/index.astro`** (placeholder)

```astro
---
const base = import.meta.env.BASE_URL;
---
<html lang="en">
  <head><meta charset="utf-8" /><title>Interactive PyTorch</title></head>
  <body>
    <h1>Interactive PyTorch</h1>
    <p>Coming soon. Base path: {base}</p>
  </body>
</html>
```

- [ ] **Step 5: Install deps**

Run: `cd ~/interactive-pytorch && npm install`
Expected: installs astro + vitest, creates `package-lock.json`, exit 0.

- [ ] **Step 6: Add Svelte + MDX integrations (auto-installs compatible versions)**

Run: `npx astro add svelte mdx --yes`
Expected: installs `@astrojs/svelte`, `svelte`, `@astrojs/mdx`; rewrites `astro.config.mjs` with the integrations; exit 0.

- [ ] **Step 7: Set `site` and `base` in `astro.config.mjs`**

Edit the generated config so it reads (keep the integrations array `astro add` produced):

```js
// @ts-check
import { defineConfig } from 'astro/config';
import svelte from '@astrojs/svelte';
import mdx from '@astrojs/mdx';

export default defineConfig({
  site: 'https://yangsong97.github.io',
  base: '/interactive-pytorch',
  integrations: [svelte(), mdx()],
});
```

- [ ] **Step 8: Verify the build**

Run: `npm run build`
Expected: exit 0; output in `dist/`. Then `npm run dev` and confirm the page renders at the printed localhost URL.

- [ ] **Step 9: Commit**

```bash
git add -A
git -c user.name="Yang Song" -c user.email="yangsong@alumni.princeton.edu" commit -m "Scaffold Astro project with Svelte + MDX"
```

### Task 0.2: GitHub Actions deploy workflow

**Files:**
- Create: `.github/workflows/deploy.yml`

- [ ] **Step 1: Write the workflow** (official Astro → Pages recipe)

```yaml
name: Deploy to GitHub Pages
on:
  push:
    branches: [main]
  workflow_dispatch:
permissions:
  contents: read
  pages: write
  id-token: write
concurrency:
  group: pages
  cancel-in-progress: false
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      - name: Build with Astro
        uses: withastro/action@v3
  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: Commit**

```bash
git add .github/
git -c user.name="Yang Song" -c user.email="yangsong@alumni.princeton.edu" commit -m "Add GitHub Pages deploy workflow"
```

### Task 0.3: Create the GitHub repo, push, enable Pages, verify live

**Files:** none (uses `gh`)

- [ ] **Step 1: Create the public repo and push**

Run:
```bash
cd ~/interactive-pytorch
gh repo create interactive-pytorch --public --source=. --remote=origin --description "Learn PyTorch interactively" --push
```
Expected: repo created at `github.com/yangsong97/interactive-pytorch`, `main` pushed.

- [ ] **Step 2: Set Pages source to GitHub Actions**

Run:
```bash
gh api -X POST repos/yangsong97/interactive-pytorch/pages -f build_type=workflow || \
gh api -X PUT  repos/yangsong97/interactive-pytorch/pages -f build_type=workflow
```
Expected: HTTP 201/204. (POST creates; PUT updates if it already exists.)

- [ ] **Step 3: Watch the deploy**

Run: `gh run watch $(gh run list --limit 1 --json databaseId -q '.[0].databaseId')`
Expected: workflow succeeds (build + deploy jobs green).

- [ ] **Step 4: Verify live URL**

Run: `curl -sSL -o /dev/null -w "%{http_code}\n" https://yangsong97.github.io/interactive-pytorch/`
Expected: `200` (may take 1–2 min after first deploy). Confirms the pipeline works end-to-end.

---

## Phase 1 — Design system, layouts, shared components

### Task 1.1: Monochrome design tokens + base styles

**Files:**
- Create: `src/styles/global.css`

- [ ] **Step 1: Write `global.css`** with CSS custom properties (muted monochrome), base typography, code styling, and a `.container` width.

```css
:root {
  --bg: #fafafa;
  --surface: #ffffff;
  --border: #e5e5e5;
  --text: #1a1a1a;
  --text-muted: #666666;
  --accent: #333333;          /* monochrome accent */
  --accent-soft: #f0f0f0;
  --code-bg: #f6f6f6;
  --radius: 8px;
  --maxw: 760px;
  --font-sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --font-mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #121212; --surface: #1b1b1b; --border: #2a2a2a;
    --text: #ececec; --text-muted: #9a9a9a; --accent: #d0d0d0;
    --accent-soft: #242424; --code-bg: #1e1e1e;
  }
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin: 0; background: var(--bg); color: var(--text);
  font-family: var(--font-sans); line-height: 1.65; }
a { color: var(--accent); text-underline-offset: 3px; }
h1,h2,h3 { line-height: 1.25; }
pre, code { font-family: var(--font-mono); }
:not(pre) > code { background: var(--code-bg); padding: 0.15em 0.4em;
  border-radius: 4px; font-size: 0.9em; }
pre { background: var(--code-bg); padding: 1rem; border-radius: var(--radius);
  overflow-x: auto; border: 1px solid var(--border); }
.container { max-width: var(--maxw); margin: 0 auto; padding: 0 1rem; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
```

- [ ] **Step 2: Verify** — import into the placeholder page and `npm run build` (exit 0). Commit.

```bash
git add src/styles/global.css
git -c user.name="Yang Song" -c user.email="yangsong@alumni.princeton.edu" commit -m "Add monochrome design tokens and base styles"
```

### Task 1.2: `Base.astro` layout

**Files:**
- Create: `src/layouts/Base.astro`

- [ ] **Step 1: Write the HTML shell** — accepts `title` + `description` props, imports `global.css`, renders a header (site title links home, link to `/start`) and a `<slot/>`, plus a footer with a GitHub link.

```astro
---
import '../styles/global.css';
const base = import.meta.env.BASE_URL.replace(/\/$/, '');
interface Props { title: string; description?: string; }
const { title, description = 'Learn PyTorch interactively.' } = Astro.props;
---
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="description" content={description} />
    <title>{title}</title>
  </head>
  <body>
    <header class="site-header">
      <a href={`${base}/`} class="brand">Interactive&nbsp;PyTorch</a>
      <nav><a href={`${base}/start`}>How to use</a>
        <a href="https://github.com/yangsong97/interactive-pytorch">GitHub</a></nav>
    </header>
    <slot />
    <footer class="site-footer container">
      <p>Built to learn PyTorch by seeing it. Code runs in Google Colab.</p>
    </footer>
    <style>
      .site-header { display:flex; justify-content:space-between; align-items:center;
        padding:1rem; border-bottom:1px solid var(--border); background:var(--surface); }
      .brand { font-weight:700; text-decoration:none; }
      .site-header nav a { margin-left:1rem; color:var(--text-muted); text-decoration:none; }
      .site-footer { color:var(--text-muted); border-top:1px solid var(--border);
        margin-top:3rem; padding-top:1.5rem; padding-bottom:2rem; font-size:.9rem; }
    </style>
  </body>
</html>
```

- [ ] **Step 2: Verify** build (exit 0) and commit.

```bash
git add src/layouts/Base.astro
git -c user.name="Yang Song" -c user.email="yangsong@alumni.princeton.edu" commit -m "Add Base layout"
```

### Task 1.3: `withBase()` link helper

**Files:**
- Create: `src/lib/url.ts`
- Test: `src/lib/url.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
import { describe, it, expect } from 'vitest';
import { withBase } from './url';

describe('withBase', () => {
  it('prefixes a root-relative path with the base', () => {
    expect(withBase('/lessons/intro', '/interactive-pytorch')).toBe('/interactive-pytorch/lessons/intro');
  });
  it('avoids double slashes', () => {
    expect(withBase('/x', '/interactive-pytorch/')).toBe('/interactive-pytorch/x');
  });
  it('handles empty base', () => {
    expect(withBase('/x', '')).toBe('/x');
  });
});
```

- [ ] **Step 2: Create `vitest.config.ts`**

```ts
import { defineConfig } from 'vitest/config';
export default defineConfig({ test: { environment: 'node', include: ['src/**/*.test.ts'] } });
```

- [ ] **Step 3: Run test → fails**

Run: `npx vitest run src/lib/url.test.ts`
Expected: FAIL (`withBase` not defined).

- [ ] **Step 4: Implement `src/lib/url.ts`**

```ts
/** Join Astro's BASE_URL with a root-relative path, collapsing duplicate slashes. */
export function withBase(path: string, base = import.meta.env.BASE_URL): string {
  const b = base.replace(/\/$/, '');
  const p = path.startsWith('/') ? path : `/${path}`;
  return `${b}${p}`;
}
```

- [ ] **Step 5: Run test → passes.** Run: `npx vitest run src/lib/url.test.ts` → PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lib/url.ts src/lib/url.test.ts vitest.config.ts
git -c user.name="Yang Song" -c user.email="yangsong@alumni.princeton.edu" commit -m "Add withBase URL helper with tests"
```

### Task 1.4: `content.config.ts` lessons collection

**Files:**
- Create: `src/content.config.ts`

- [ ] **Step 1: Define the collection schema** (Astro 5 content layer, glob loader)

```ts
import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const lessons = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/lessons' }),
  schema: z.object({
    title: z.string(),
    module: z.string(),          // e.g. "M1 · Tensors"
    order: z.number(),           // global ordering for prev/next + sidebar
    summary: z.string(),
    colab: z.string().optional(),// notebook filename under notebooks/
    draft: z.boolean().default(false),
  }),
});
export const collections = { lessons };
```

- [ ] **Step 2:** Add a temporary `src/content/lessons/intro.mdx` with valid frontmatter (title/module/order/summary), `npm run build` (exit 0) to confirm the schema loads. Commit both.

```bash
git add src/content.config.ts src/content/lessons/intro.mdx
git -c user.name="Yang Song" -c user.email="yangsong@alumni.princeton.edu" commit -m "Add lessons content collection schema"
```

### Task 1.5: `Sidebar.astro`, `Lesson.astro`, `ColabButton.astro`, `Callout.astro`

**Files:**
- Create: `src/components/Sidebar.astro`, `src/components/ColabButton.astro`, `src/components/Callout.astro`, `src/layouts/Lesson.astro`

- [ ] **Step 1: `ColabButton.astro`** — props `notebook: string`; renders an anchor to the GitHub-Colab URL, opening in a new tab.

```astro
---
interface Props { notebook: string; }
const { notebook } = Astro.props;
const url = `https://colab.research.google.com/github/yangsong97/interactive-pytorch/blob/main/notebooks/${notebook}`;
---
<a class="colab-btn" href={url} target="_blank" rel="noopener">▶ Run this in Google Colab</a>
<style>
  .colab-btn { display:inline-block; border:1px solid var(--border); background:var(--surface);
    color:var(--text); padding:.5rem .9rem; border-radius:var(--radius); text-decoration:none;
    font-weight:600; font-size:.9rem; }
  .colab-btn:hover { background:var(--accent-soft); }
</style>
```

- [ ] **Step 2: `Callout.astro`** — `type: 'note' | 'tip' | 'warn'`; slotted body; muted bordered box.

```astro
---
interface Props { type?: 'note' | 'tip' | 'warn'; title?: string; }
const { type = 'note', title } = Astro.props;
const icon = { note: 'ℹ︎', tip: '✓', warn: '⚠︎' }[type];
---
<aside class={`callout ${type}`}>
  <p class="callout-title">{icon} {title ?? type.toUpperCase()}</p>
  <div class="callout-body"><slot /></div>
</aside>
<style>
  .callout { border:1px solid var(--border); border-left:3px solid var(--accent);
    background:var(--surface); padding:.75rem 1rem; border-radius:var(--radius); margin:1.25rem 0; }
  .callout-title { font-weight:700; margin:.1rem 0 .4rem; font-size:.85rem; color:var(--text-muted); }
  .callout-body :global(p) { margin:.3rem 0; }
</style>
```

- [ ] **Step 3: `Sidebar.astro`** — loads lessons via `getCollection('lessons')`, filters out drafts, sorts by `order`, groups by `module`, highlights the active slug (prop `currentSlug`). Use `withBase` for links.

```astro
---
import { getCollection } from 'astro:content';
import { withBase } from '../lib/url';
interface Props { currentSlug: string; }
const { currentSlug } = Astro.props;
const lessons = (await getCollection('lessons', ({ data }) => !data.draft))
  .sort((a, b) => a.data.order - b.data.order);
const groups = new Map<string, typeof lessons>();
for (const l of lessons) {
  const g = groups.get(l.data.module) ?? [];
  g.push(l); groups.set(l.data.module, g);
}
---
<nav class="sidebar" aria-label="Lessons">
  {[...groups.entries()].map(([mod, items]) => (
    <details open class="mod">
      <summary>{mod}</summary>
      <ul>{items.map((l) => (
        <li class={l.id === currentSlug ? 'active' : ''}>
          <a href={withBase(`/lessons/${l.id}`)}>{l.data.title}</a>
        </li>))}</ul>
    </details>))}
</nav>
<style>
  .sidebar { font-size:.9rem; }
  .mod summary { font-weight:700; cursor:pointer; padding:.4rem 0; color:var(--text-muted); }
  .mod ul { list-style:none; margin:0 0 .5rem; padding:0 0 0 .25rem; }
  .mod li { padding:.2rem 0; }
  .mod li a { text-decoration:none; color:var(--text); }
  .mod li.active a { font-weight:700; border-left:2px solid var(--accent); padding-left:.5rem; }
</style>
```

- [ ] **Step 4: `Lesson.astro`** — wraps `Base`, renders a two-column grid (sidebar + article), the lesson title/summary, `<slot/>`, optional Colab button, and prev/next links computed from the sorted collection. Props: `title`, `summary`, `slug`, `colab?`.

```astro
---
import Base from './Base.astro';
import Sidebar from '../components/Sidebar.astro';
import ColabButton from '../components/ColabButton.astro';
import { getCollection } from 'astro:content';
import { withBase } from '../lib/url';
interface Props { title: string; summary: string; slug: string; colab?: string; }
const { title, summary, slug, colab } = Astro.props;
const all = (await getCollection('lessons', ({ data }) => !data.draft))
  .sort((a, b) => a.data.order - b.data.order);
const i = all.findIndex((l) => l.id === slug);
const prev = i > 0 ? all[i - 1] : null;
const next = i >= 0 && i < all.length - 1 ? all[i + 1] : null;
---
<Base title={`${title} · Interactive PyTorch`} description={summary}>
  <div class="layout">
    <aside class="rail"><Sidebar currentSlug={slug} /></aside>
    <article class="article">
      <h1>{title}</h1>
      <p class="summary">{summary}</p>
      <slot />
      {colab && <div class="colab-wrap"><ColabButton notebook={colab} /></div>}
      <nav class="prevnext">
        {prev ? <a href={withBase(`/lessons/${prev.id}`)}>← {prev.data.title}</a> : <span/>}
        {next ? <a href={withBase(`/lessons/${next.id}`)}>{next.data.title} →</a> : <span/>}
      </nav>
    </article>
  </div>
</Base>
<style>
  .layout { display:grid; grid-template-columns:240px 1fr; gap:2rem; max-width:1100px;
    margin:0 auto; padding:1.5rem 1rem; }
  .rail { position:sticky; top:1rem; align-self:start; max-height:90vh; overflow:auto; }
  .article { min-width:0; }
  .summary { color:var(--text-muted); font-size:1.05rem; }
  .colab-wrap { margin:1.5rem 0; }
  .prevnext { display:flex; justify-content:space-between; margin-top:2.5rem;
    border-top:1px solid var(--border); padding-top:1rem; gap:1rem; }
  .prevnext a { text-decoration:none; }
  @media (max-width:780px){ .layout{ grid-template-columns:1fr; } .rail{ position:static; } }
</style>
```

- [ ] **Step 5: Verify** build (exit 0). Commit all four files.

```bash
git add src/components/ src/layouts/Lesson.astro
git -c user.name="Yang Song" -c user.email="yangsong@alumni.princeton.edu" commit -m "Add sidebar, lesson layout, Colab button, callout"
```

### Task 1.6: Lesson renderer page `[...slug].astro`

**Files:**
- Create: `src/pages/lessons/[...slug].astro`

- [ ] **Step 1: Write the dynamic route** — `getStaticPaths` from the collection; render the MDX body inside `Lesson.astro`, passing frontmatter through.

```astro
---
import { getCollection, render } from 'astro:content';
import Lesson from '../../layouts/Lesson.astro';
export async function getStaticPaths() {
  const lessons = await getCollection('lessons', ({ data }) => !data.draft);
  return lessons.map((l) => ({ params: { slug: l.id }, props: { lesson: l } }));
}
const { lesson } = Astro.props;
const { Content } = await render(lesson);
---
<Lesson title={lesson.data.title} summary={lesson.data.summary}
        slug={lesson.id} colab={lesson.data.colab}>
  <Content />
</Lesson>
```

- [ ] **Step 2: Verify** — update the temp `intro.mdx` to real frontmatter; `npm run build`; `npm run dev` and confirm `/interactive-pytorch/lessons/intro` renders with the sidebar. Commit.

```bash
git add src/pages/lessons/
git -c user.name="Yang Song" -c user.email="yangsong@alumni.princeton.edu" commit -m "Add lesson renderer route"
```

---

## Phase 2 — Widget logic library (test-first, framework-free)

> All modules in `src/lib/` are pure TypeScript with Vitest tests. This is where TDD applies. CONTRIBUTION POINTS A and B live here.

### Task 2.1: `tensor.ts` — shapes, broadcasting, reshape

**Files:**
- Create: `src/lib/tensor.ts`, `src/lib/tensor.test.ts`

- [ ] **Step 1: Write failing tests**

```ts
import { describe, it, expect } from 'vitest';
import { broadcastShape, canReshape, numel } from './tensor';

describe('tensor shape utils', () => {
  it('numel multiplies dims', () => { expect(numel([2,3,4])).toBe(24); });
  it('broadcasts trailing-aligned shapes', () => {
    expect(broadcastShape([3,1],[1,4])).toEqual([3,4]);
    expect(broadcastShape([5],[3,5])).toEqual([3,5]);
  });
  it('returns null for incompatible shapes', () => {
    expect(broadcastShape([2,3],[4,3,2])).toBeNull(); // 2 vs 3 mismatch
  });
  it('canReshape only when element counts match', () => {
    expect(canReshape([2,6],[3,4])).toBe(true);
    expect(canReshape([2,6],[5,2])).toBe(false);
  });
});
```

- [ ] **Step 2: Run → fails.** `npx vitest run src/lib/tensor.test.ts` → FAIL.

- [ ] **Step 3: Implement `tensor.ts`**

```ts
export function numel(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}
/** NumPy/PyTorch broadcasting: align from the right; dims must be equal or one is 1. */
export function broadcastShape(a: number[], b: number[]): number[] | null {
  const out: number[] = [];
  const n = Math.max(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const da = a[a.length - 1 - i] ?? 1;
    const db = b[b.length - 1 - i] ?? 1;
    if (da !== db && da !== 1 && db !== 1) return null;
    out.unshift(Math.max(da, db));
  }
  return out;
}
export function canReshape(from: number[], to: number[]): boolean {
  return numel(from) === numel(to);
}
```

- [ ] **Step 4: Run → passes.** Commit.

```bash
git add src/lib/tensor.ts src/lib/tensor.test.ts
git -c user.name="Yang Song" -c user.email="yangsong@alumni.princeton.edu" commit -m "Add tensor shape/broadcast utils with tests"
```

### Task 2.2: `autograd.ts` — scalar reverse-mode autodiff (the engine)

**Files:**
- Create: `src/lib/autograd.ts`, `src/lib/autograd.test.ts`

- [ ] **Step 1: Write failing tests** (verify values AND gradients against hand computation)

```ts
import { describe, it, expect } from 'vitest';
import { V } from './autograd';

describe('scalar autograd', () => {
  it('computes d/dx of x^2 at x=3 is 6', () => {
    const x = V(3); const y = x.mul(x); y.backward();
    expect(y.data).toBe(9); expect(x.grad).toBeCloseTo(6);
  });
  it('chains add/mul: f = (x*w+b), df/dw = x', () => {
    const x = V(2), w = V(-1), b = V(0.5);
    const f = x.mul(w).add(b); f.backward();
    expect(f.data).toBeCloseTo(-1.5);
    expect(w.grad).toBeCloseTo(2);  // df/dw = x
    expect(x.grad).toBeCloseTo(-1); // df/dx = w
    expect(b.grad).toBeCloseTo(1);
  });
  it('relu blocks gradient when input < 0', () => {
    const a = V(-2); const r = a.relu(); r.backward();
    expect(r.data).toBe(0); expect(a.grad).toBe(0);
  });
});
```

- [ ] **Step 2: Run → fails.**

- [ ] **Step 3: Implement `autograd.ts`** (micrograd-style `Value`)

```ts
export class Value {
  data: number; grad = 0;
  _backward: () => void = () => {};
  _prev: Value[];
  constructor(data: number, _children: Value[] = []) {
    this.data = data; this._prev = _children;
  }
  add(o: Value | number): Value {
    const ot = o instanceof Value ? o : new Value(o);
    const out = new Value(this.data + ot.data, [this, ot]);
    out._backward = () => { this.grad += out.grad; ot.grad += out.grad; };
    return out;
  }
  mul(o: Value | number): Value {
    const ot = o instanceof Value ? o : new Value(o);
    const out = new Value(this.data * ot.data, [this, ot]);
    out._backward = () => { this.grad += ot.data * out.grad; ot.grad += this.data * out.grad; };
    return out;
  }
  relu(): Value {
    const out = new Value(this.data < 0 ? 0 : this.data, [this]);
    out._backward = () => { this.grad += (out.data > 0 ? 1 : 0) * out.grad; };
    return out;
  }
  tanh(): Value {
    const t = Math.tanh(this.data);
    const out = new Value(t, [this]);
    out._backward = () => { this.grad += (1 - t * t) * out.grad; };
    return out;
  }
  sub(o: Value | number): Value { const ot = o instanceof Value ? o : new Value(o);
    return this.add(ot.mul(-1)); }
  pow(p: number): Value {
    const out = new Value(Math.pow(this.data, p), [this]);
    out._backward = () => { this.grad += p * Math.pow(this.data, p - 1) * out.grad; };
    return out;
  }
  backward(): void {
    const topo: Value[] = []; const seen = new Set<Value>();
    const build = (v: Value) => { if (!seen.has(v)) { seen.add(v);
      for (const c of v._prev) build(c); topo.push(v); } };
    build(this);
    for (const v of topo) v.grad = 0;
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
  }
}
export const V = (x: number) => new Value(x);
```

- [ ] **Step 4: Run → passes.** Commit.

```bash
git add src/lib/autograd.ts src/lib/autograd.test.ts
git -c user.name="Yang Song" -c user.email="yangsong@alumni.princeton.edu" commit -m "Add scalar autograd engine with tests"
```

### Task 2.3: `activations.ts`

**Files:**
- Create: `src/lib/activations.ts`, `src/lib/activations.test.ts`

- [ ] **Step 1: Failing tests**

```ts
import { describe, it, expect } from 'vitest';
import { relu, sigmoid, tanh, leakyRelu, dRelu, dSigmoid } from './activations';
describe('activations', () => {
  it('relu', () => { expect(relu(-1)).toBe(0); expect(relu(2)).toBe(2); });
  it('sigmoid(0)=0.5', () => { expect(sigmoid(0)).toBeCloseTo(0.5); });
  it('tanh(0)=0', () => { expect(tanh(0)).toBeCloseTo(0); });
  it('leakyRelu negative slope', () => { expect(leakyRelu(-10, 0.1)).toBeCloseTo(-1); });
  it('derivatives', () => { expect(dRelu(3)).toBe(1); expect(dRelu(-3)).toBe(0);
    expect(dSigmoid(0)).toBeCloseTo(0.25); });
});
```

- [ ] **Step 2: Run → fails. Step 3: Implement**

```ts
export const relu = (x: number) => (x < 0 ? 0 : x);
export const dRelu = (x: number) => (x < 0 ? 0 : 1);
export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
export const dSigmoid = (x: number) => { const s = sigmoid(x); return s * (1 - s); };
export const tanh = (x: number) => Math.tanh(x);
export const dTanh = (x: number) => 1 - Math.tanh(x) ** 2;
export const leakyRelu = (x: number, slope = 0.01) => (x < 0 ? slope * x : x);
```

- [ ] **Step 4: Run → passes. Commit.**

```bash
git add src/lib/activations.ts src/lib/activations.test.ts
git -c user.name="Yang Song" -c user.email="yangsong@alumni.princeton.edu" commit -m "Add activation functions with tests"
```

### Task 2.4: `gradientDescent.ts` — ⭐ CONTRIBUTION POINT A

**Files:**
- Create: `src/lib/gradientDescent.ts`, `src/lib/gradientDescent.test.ts`

- [ ] **Step 1: Write failing tests** (these define the contract the user implements against)

```ts
import { describe, it, expect } from 'vitest';
import { sgdStep, type Optimizer } from './gradientDescent';

describe('sgdStep', () => {
  it('moves params downhill: p -= lr*grad', () => {
    const s: Optimizer = { params: [1, -2], velocity: [0, 0] };
    sgdStep(s, [2, 4], 0.1, 0);            // momentum=0 → plain SGD
    expect(s.params[0]).toBeCloseTo(0.8);  // 1 - 0.1*2
    expect(s.params[1]).toBeCloseTo(-2.4); // -2 - 0.1*4
  });
  it('with momentum, velocity accumulates', () => {
    const s: Optimizer = { params: [0], velocity: [0] };
    sgdStep(s, [1], 0.1, 0.9);  // v = 0.9*0 + 1 = 1 ; p = 0 - 0.1*1 = -0.1
    sgdStep(s, [1], 0.1, 0.9);  // v = 0.9*1 + 1 = 1.9 ; p = -0.1 - 0.1*1.9 = -0.29
    expect(s.params[0]).toBeCloseTo(-0.29);
  });
});
```

- [ ] **Step 2: Run → fails.**

- [ ] **Step 3: Create the file with scaffolding + TODO for the user** (the `sgdStep` body is the contribution)

```ts
export interface Optimizer {
  params: number[];
  velocity: number[]; // per-param momentum buffer (zeros when momentum unused)
}

/**
 * One in-place gradient-descent update over all params.
 * Contract (must satisfy gradientDescent.test.ts):
 *   - momentum === 0 → plain SGD:      p_i -= lr * grad_i
 *   - momentum  >  0 → SGD + momentum:  v_i = momentum * v_i + grad_i ;  p_i -= lr * v_i
 *
 * CONTRIBUTION POINT A — implement the loop below (≈5–8 lines).
 * Design choice this encodes: momentum lets the "ball" build speed across
 * consistent gradient directions and smooth past small bumps, but can overshoot.
 */
export function sgdStep(opt: Optimizer, grads: number[], lr: number, momentum = 0): void {
  // TODO(you): for each i, update opt.velocity[i] and opt.params[i] per the contract above.
  throw new Error('sgdStep not implemented');
}
```

- [ ] **Step 4 (USER or fallback): implement the body.** Reference solution (used if the user defers):

```ts
  for (let i = 0; i < opt.params.length; i++) {
    opt.velocity[i] = momentum * opt.velocity[i] + grads[i];
    opt.params[i] -= lr * opt.velocity[i];
  }
```

- [ ] **Step 5: Run → passes.** `npx vitest run src/lib/gradientDescent.test.ts` → PASS.

- [ ] **Step 6: Commit.**

```bash
git add src/lib/gradientDescent.ts src/lib/gradientDescent.test.ts
git -c user.name="Yang Song" -c user.email="yangsong@alumni.princeton.edu" commit -m "Add gradient descent step with tests"
```

### Task 2.5: `data.ts` — seeded RNG + `makeMoons`

**Files:**
- Create: `src/lib/data.ts`, `src/lib/data.test.ts`

- [ ] **Step 1: Failing tests**

```ts
import { describe, it, expect } from 'vitest';
import { mulberry32, makeMoons } from './data';
describe('data', () => {
  it('mulberry32 is deterministic for a seed', () => {
    const a = mulberry32(42), b = mulberry32(42);
    expect(a()).toBeCloseTo(b());
  });
  it('makeMoons returns n points with binary labels in range', () => {
    const { X, y } = makeMoons(50, 0.1, 7);
    expect(X.length).toBe(50); expect(y.length).toBe(50);
    expect(y.every((v) => v === 0 || v === 1)).toBe(true);
    expect(X.every(([a, b]) => Number.isFinite(a) && Number.isFinite(b))).toBe(true);
  });
});
```

- [ ] **Step 2: Run → fails. Step 3: Implement**

```ts
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => { a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export interface Dataset { X: [number, number][]; y: number[]; }
/** Two interleaving half-circles ("moons"), a classic non-linear 2D toy set. */
export function makeMoons(n: number, noise = 0.1, seed = 1): Dataset {
  const rnd = mulberry32(seed); const X: [number, number][] = []; const y: number[] = [];
  const half = Math.floor(n / 2);
  for (let i = 0; i < n; i++) {
    const top = i < half; const t = Math.PI * (i % half) / (half - 1 || 1);
    const nx = (rnd() - 0.5) * 2 * noise, ny = (rnd() - 0.5) * 2 * noise;
    if (top) { X.push([Math.cos(t) + nx, Math.sin(t) + ny]); y.push(0); }
    else { X.push([1 - Math.cos(t) + nx, 0.5 - Math.sin(t) + ny]); y.push(1); }
  }
  return { X, y };
}
```

- [ ] **Step 4: Run → passes. Commit.**

```bash
git add src/lib/data.ts src/lib/data.test.ts
git -c user.name="Yang Song" -c user.email="yangsong@alumni.princeton.edu" commit -m "Add seeded RNG and makeMoons dataset with tests"
```

### Task 2.6: `mlp.ts` — tiny MLP on the autograd engine — ⭐ CONTRIBUTION POINT B

**Files:**
- Create: `src/lib/mlp.ts`, `src/lib/mlp.test.ts`

- [ ] **Step 1: Write failing tests** (forward shape + that training reduces loss)

```ts
import { describe, it, expect } from 'vitest';
import { MLP, mseLoss } from './mlp';
import { makeMoons } from './data';

describe('MLP', () => {
  it('forward returns one output Value per input', () => {
    const net = new MLP(2, 4, 1234);
    const out = net.forward([0.5, -0.3]);
    expect(typeof out.data).toBe('number');
  });
  it('training on moons reduces loss', () => {
    const { X, y } = makeMoons(80, 0.1, 3);
    const net = new MLP(2, 8, 7);
    const before = epochLoss(net, X, y);
    for (let e = 0; e < 40; e++) net.trainEpoch(X, y, 0.1);
    const after = epochLoss(net, X, y);
    expect(after).toBeLessThan(before * 0.7); // at least 30% lower
  });
});
function epochLoss(net: MLP, X: [number,number][], y: number[]): number {
  let s = 0; for (let i = 0; i < X.length; i++) s += mseLoss(net.forward(X[i]), y[i]).data;
  return s / X.length;
}
```

- [ ] **Step 2: Run → fails.**

- [ ] **Step 3: Create `mlp.ts` with `forward` as the contribution point**

```ts
import { Value } from './autograd';
import { mulberry32 } from './data';

export function mseLoss(pred: Value, target: number): Value {
  return pred.sub(target).pow(2);
}

/** A 2-layer MLP: input → hidden (tanh) → 1 output. Scalar-valued via the autograd engine. */
export class MLP {
  W1: Value[][]; b1: Value[]; W2: Value[]; b2: Value; hidden: number;
  constructor(inDim: number, hidden: number, seed = 1) {
    const r = mulberry32(seed); const rn = () => (r() - 0.5) * 2;
    this.hidden = hidden;
    this.W1 = Array.from({ length: hidden }, () =>
      Array.from({ length: inDim }, () => new Value(rn())));
    this.b1 = Array.from({ length: hidden }, () => new Value(0));
    this.W2 = Array.from({ length: hidden }, () => new Value(rn()));
    this.b2 = new Value(0);
  }

  /**
   * Forward pass for a single 2D input `x`.
   * Compute: h_j = tanh( sum_k W1[j][k]*x[k] + b1[j] );  out = sum_j W2[j]*h_j + b2.
   *
   * CONTRIBUTION POINT B — implement using the Value API
   * (.mul/.add/.tanh). ≈6–9 lines. Return the output Value.
   */
  forward(x: number[]): Value {
    // TODO(you): build the hidden layer with tanh, then the linear output.
    throw new Error('MLP.forward not implemented');
  }

  parameters(): Value[] {
    return [...this.W1.flat(), ...this.b1, ...this.W2, this.b2];
  }

  trainEpoch(X: number[][], y: number[], lr: number): number {
    const params = this.parameters();
    let total = 0;
    for (let i = 0; i < X.length; i++) {
      const loss = mseLoss(this.forward(X[i]), y[i]);
      loss.backward();
      for (const p of params) p.data -= lr * p.grad; // plain SGD over Values
      total += loss.data;
    }
    return total / X.length;
  }
}
```

- [ ] **Step 4 (USER or fallback): implement `forward`.** Reference solution:

```ts
    const h = this.W1.map((row, j) => {
      let s: Value = this.b1[j];
      for (let k = 0; k < x.length; k++) s = s.add(row[k].mul(x[k]));
      return s.tanh();
    });
    let out: Value = this.b2;
    for (let j = 0; j < this.hidden; j++) out = out.add(this.W2[j].mul(h[j]));
    return out;
```

- [ ] **Step 5: Run → passes** (loss drops ≥30%). Commit.

```bash
git add src/lib/mlp.ts src/lib/mlp.test.ts
git -c user.name="Yang Song" -c user.email="yangsong@alumni.princeton.edu" commit -m "Add tiny MLP on autograd engine with tests"
```

### Task 2.7: `conv.ts` — 2D valid convolution

**Files:**
- Create: `src/lib/conv.ts`, `src/lib/conv.test.ts`

- [ ] **Step 1: Failing tests**

```ts
import { describe, it, expect } from 'vitest';
import { convolve2d, KERNELS } from './conv';
describe('convolve2d', () => {
  it('valid conv shrinks by kernel-1', () => {
    const img = [[1,2,3],[4,5,6],[7,8,9]];
    const k = [[0,0,0],[0,1,0],[0,0,0]]; // identity
    const out = convolve2d(img, k);
    expect(out.length).toBe(1); expect(out[0].length).toBe(1);
    expect(out[0][0]).toBe(5); // center pixel
  });
  it('ships named kernels', () => {
    expect(KERNELS.edge.length).toBe(3);
  });
});
```

- [ ] **Step 2: Run → fails. Step 3: Implement**

```ts
export function convolve2d(img: number[][], kernel: number[][]): number[][] {
  const kh = kernel.length, kw = kernel[0].length;
  const oh = img.length - kh + 1, ow = img[0].length - kw + 1;
  const out: number[][] = [];
  for (let i = 0; i < oh; i++) {
    const row: number[] = [];
    for (let j = 0; j < ow; j++) {
      let s = 0;
      for (let a = 0; a < kh; a++) for (let b = 0; b < kw; b++) s += img[i+a][j+b] * kernel[a][b];
      row.push(s);
    }
    out.push(row);
  }
  return out;
}
export const KERNELS: Record<string, number[][]> = {
  identity: [[0,0,0],[0,1,0],[0,0,0]],
  edge:     [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],
  blur:     [[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]],
  sharpen:  [[0,-1,0],[-1,5,-1],[0,-1,0]],
};
```

- [ ] **Step 4: Run → passes. Commit.**

```bash
git add src/lib/conv.ts src/lib/conv.test.ts
git -c user.name="Yang Song" -c user.email="yangsong@alumni.princeton.edu" commit -m "Add 2D convolution with named kernels and tests"
```

- [ ] **Step 5: Run the FULL suite.** `npx vitest run` → all green.

---

## Phase 3 — Svelte widget components (presentation)

> Each widget imports tested logic from `src/lib/` and renders interactive controls + a canvas/SVG. Verified by `npm run build` + visual preview (no unit tests for canvas). Each uses a shared `<canvas>` pattern: a `draw()` function called on mount and whenever reactive state changes.

### Task 3.1: `TensorPlayground.svelte`

**Files:** Create `src/components/widgets/TensorPlayground.svelte`

- [ ] **Step 1:** Implement a widget with: number inputs for a source shape (e.g. `[2,6]`), a target reshape input, a live grid drawing `numel` cells numbered 0..N-1 arranged by the target shape, and an inline validity message using `canReshape`/`numel`. A second panel shows two shapes and the `broadcastShape` result (or "incompatible"). Use Svelte `$state`/`$derived` (Svelte 5 runes). Draw the grid with HTML/CSS (divs), not canvas, for crisp labels.
- [ ] **Step 2:** `npm run build` (exit 0). Manually verify in `npm run dev` via an MDX test embed. Commit.

```bash
git add src/components/widgets/TensorPlayground.svelte
git -c user.name="Yang Song" -c user.email="yangsong@alumni.princeton.edu" commit -m "Add TensorPlayground widget"
```

### Task 3.2: `ActivationExplorer.svelte`

**Files:** Create `src/components/widgets/ActivationExplorer.svelte`

- [ ] **Step 1:** Canvas plot (width 480, height 280) of the selected activation over x∈[-6,6]; a `<select>` for relu/sigmoid/tanh/leakyRelu; a draggable vertical guide (range input for x) showing f(x) and f′(x) read off `activations.ts`. Redraw on state change.
- [ ] **Step 2:** Build + preview verify. Commit.

```bash
git add src/components/widgets/ActivationExplorer.svelte
git -c user.name="Yang Song" -c user.email="yangsong@alumni.princeton.edu" commit -m "Add ActivationExplorer widget"
```

### Task 3.3: `GradientDescent.svelte`

**Files:** Create `src/components/widgets/GradientDescent.svelte`

- [ ] **Step 1:** Visualize descent on a fixed 1D loss `L(p) = p^2` (grad `2p`). Controls: learning-rate slider (0.01–1.1), momentum slider (0–0.95), "step", "run", "reset". Maintain an `Optimizer` from `gradientDescent.ts`; each step calls `sgdStep(opt, [2*opt.params[0]], lr, momentum)` and plots the marker on the parabola + a small loss-vs-step trace. With lr>1 it visibly diverges — call that out in the lesson. Use canvas.
- [ ] **Step 2:** Build + preview verify (try lr=0.1 converges, lr=1.05 diverges). Commit.

```bash
git add src/components/widgets/GradientDescent.svelte
git -c user.name="Yang Song" -c user.email="yangsong@alumni.princeton.edu" commit -m "Add GradientDescent widget"
```

### Task 3.4: `AutogradGraph.svelte`

**Files:** Create `src/components/widgets/AutogradGraph.svelte`

- [ ] **Step 1:** Build the fixed expression `y = (x*w + b)^2` using the `Value` API with sliders for x, w, b. On "Run backward", call `y.backward()` and render a small left-to-right node diagram (SVG): each node shows its op, `data`, and `grad`. Recompute the graph whenever a slider changes (rebuild Values; values update, grads show after backward). This makes the chain rule visible.
- [ ] **Step 2:** Build + preview verify (e.g. x=2,w=-1,b=0.5 → check grads match Task 2.2 reasoning). Commit.

```bash
git add src/components/widgets/AutogradGraph.svelte
git -c user.name="Yang Song" -c user.email="yangsong@alumni.princeton.edu" commit -m "Add AutogradGraph widget"
```

### Task 3.5: `TrainingLoop.svelte` (showpiece)

**Files:** Create `src/components/widgets/TrainingLoop.svelte`

- [ ] **Step 1:** Combine `makeMoons` + `MLP`. Canvas 1 (≈360²): scatter the dataset (two classes, monochrome shapes: filled vs hollow) and shade the decision boundary by sampling `net.forward` on a grid (threshold 0.5). Canvas 2: loss-vs-epoch line. Controls: learning rate, hidden units (rebuilds net), "train" (animation loop calling `net.trainEpoch` once per frame via `requestAnimationFrame`), "pause", "reset (reseed)". Show current epoch + loss. Clean up the rAF on unmount.
- [ ] **Step 2:** Build + preview verify (watch boundary form, loss drop). Commit.

```bash
git add src/components/widgets/TrainingLoop.svelte
git -c user.name="Yang Song" -c user.email="yangsong@alumni.princeton.edu" commit -m "Add TrainingLoop showpiece widget"
```

### Task 3.6: `ConvVisualizer.svelte`

**Files:** Create `src/components/widgets/ConvVisualizer.svelte`

- [ ] **Step 1:** A small grayscale image (e.g. 8×8 hardcoded or a simple generated shape). `<select>` over `KERNELS`. Render the input grid, the kernel, and the output feature map (`convolve2d`). On hover/step, highlight the current 3×3 receptive field on the input and the produced output cell, showing the multiply-accumulate sum. Use CSS-grid of cells with numeric labels.
- [ ] **Step 2:** Build + preview verify. Commit.

```bash
git add src/components/widgets/ConvVisualizer.svelte
git -c user.name="Yang Song" -c user.email="yangsong@alumni.princeton.edu" commit -m "Add ConvVisualizer widget"
```

### Task 3.7: `Quiz.svelte` ("check your understanding")

**Files:** Create `src/components/Quiz.svelte`

- [ ] **Step 1:** Props: `question: string`, `answer: string`. Renders the question and a "Show answer" toggle revealing the answer. Keyboard accessible (a real `<button>`).
- [ ] **Step 2:** Build + preview verify. Commit.

```bash
git add src/components/Quiz.svelte
git -c user.name="Yang Song" -c user.email="yangsong@alumni.princeton.edu" commit -m "Add Quiz reveal component"
```

---

## Phase 4 — Content: home, start, lessons, Colab notebooks

### Task 4.1: Home page

**Files:** Modify `src/pages/index.astro`

- [ ] **Step 1:** Rebuild `index.astro` on `Base.astro`: a hero ("Learn PyTorch by seeing it work"), one sentence on the visualize-then-run-in-Colab model, a "Start learning" button (→ first lesson, via `withBase`), and an auto-generated module overview from `getCollection('lessons')` grouped by module. Monochrome, centered, `.container`.
- [ ] **Step 2:** Build + preview verify. Commit.

```bash
git add src/pages/index.astro
git -c user.name="Yang Song" -c user.email="yangsong@alumni.princeton.edu" commit -m "Build home page"
```

### Task 4.2: `start` page (Colab 101)

**Files:** Create `src/pages/start.astro`

- [ ] **Step 1:** On `Base.astro`: short guide — "No install needed", how to open a Colab notebook, run a cell (Shift+Enter), and that the first PyTorch import may take ~20s. Link back to lessons.
- [ ] **Step 2:** Build + preview verify. Commit.

```bash
git add src/pages/start.astro
git -c user.name="Yang Song" -c user.email="yangsong@alumni.princeton.edu" commit -m "Add Colab 101 start page"
```

### Task 4.3: Colab notebook generator

**Files:** Create `notebooks/build_notebooks.py`, generate `notebooks/*.ipynb`

- [ ] **Step 1:** Write a Python script using `nbformat` that emits one notebook per module (filenames referenced by lesson frontmatter `colab`, see Task 4.5 table). Each notebook: a title markdown cell, a `!pip -q install torch` cell, then the runnable PyTorch snippets for that module's lessons. Run with `uv`-managed env if available, else `pip3 install nbformat`.

```python
# notebooks/build_notebooks.py  (excerpt pattern; one entry per module)
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def build(path, title, cells):
    nb = new_notebook()
    nb.cells = [new_markdown_cell(f"# {title}\n\nRun cells with Shift+Enter."),
                new_code_cell("!pip -q install torch")]
    for kind, src in cells:
        nb.cells.append(new_markdown_cell(src) if kind == "md" else new_code_cell(src))
    nbf.write(nb, path)

build("notebooks/m1_tensors.ipynb", "M1 · Tensors", [
    ("code", "import torch\nx = torch.tensor([[1.,2.,3.],[4.,5.,6.]])\nprint(x, x.shape, x.dtype)"),
    ("md",   "## Broadcasting"),
    ("code", "a = torch.arange(3); b = torch.ones(2,3); print((a+b))"),
])
# ... build(...) for m2..m8 + capstone, mirroring each lesson's code snippet.
```

- [ ] **Step 2:** Run `python3 notebooks/build_notebooks.py` (after `pip3 install nbformat`); verify `.ipynb` files are valid JSON (`python3 -c "import nbformat,glob;[nbformat.read(f,4) for f in glob.glob('notebooks/*.ipynb')]"`). Commit script + notebooks.

```bash
git add notebooks/
git -c user.name="Yang Song" -c user.email="yangsong@alumni.princeton.edu" commit -m "Add Colab notebooks and generator"
```

### Task 4.4: Lesson template (the canonical example) + authoring guide

**Files:** Create `src/content/lessons/01-what-is-pytorch.mdx`; replace temp `intro.mdx`

- [ ] **Step 1:** Author the first lesson fully as the template others copy. Structure:

```mdx
---
title: What is PyTorch?
module: M0 · Orientation
order: 1
summary: PyTorch in one minute, and how to use this site.
colab: m1_tensors.ipynb
---
import Callout from '../../components/Callout.astro';

PyTorch is a Python library for building and training neural networks...

<Callout type="tip" title="No install needed">
Every code block here can be run in Google Colab — click the button at the
bottom of each lesson. The first `import torch` takes ~20 seconds.
</Callout>

## The mental model
... prose ...

```python
import torch
x = torch.tensor([1.0, 2.0, 3.0])
print(x)
```
```

- [ ] **Step 2:** Delete the temporary `intro.mdx`. Build + preview verify (renders with sidebar, Colab button, prev/next). Commit.

```bash
git rm src/content/lessons/intro.mdx
git add src/content/lessons/01-what-is-pytorch.mdx
git -c user.name="Yang Song" -c user.email="yangsong@alumni.princeton.edu" commit -m "Add first lesson as content template"
```

### Task 4.5: Author all lessons

**Files:** Create one MDX file per row below in `src/content/lessons/`.

Author each lesson following the Task 4.4 template: frontmatter (title/module/order/summary/colab), concept prose, the embedded widget (where listed — import from `../../components/widgets/X.svelte` and use `<X client:visible />`), a `python` code block with the listed APIs, and 1–2 `<Quiz/>` items. Commit after each module (8 commits) to keep changes bite-sized.

| order | slug | module | widget | key PyTorch APIs in code block | colab |
|------:|------|--------|--------|-------------------------------|-------|
| 1 | 01-what-is-pytorch | M0 · Orientation | — | `torch.tensor` | m1_tensors.ipynb |
| 2 | 02-creating-tensors | M1 · Tensors | TensorPlayground | `tensor`,`zeros`,`ones`,`arange`,`rand` | m1_tensors.ipynb |
| 3 | 03-shape-dtype-device | M1 · Tensors | — | `.shape`,`.dtype`,`.device`,`.to` | m1_tensors.ipynb |
| 4 | 04-indexing-slicing | M1 · Tensors | TensorPlayground | slicing, boolean masks | m1_tensors.ipynb |
| 5 | 05-reshaping | M1 · Tensors | TensorPlayground | `view`,`reshape`,`squeeze`,`unsqueeze` | m1_tensors.ipynb |
| 6 | 06-math-broadcasting | M1 · Tensors | TensorPlayground | elementwise ops, broadcasting, `@` | m1_tensors.ipynb |
| 7 | 07-numpy-bridge | M1 · Tensors | — | `.numpy()`,`from_numpy` | m1_tensors.ipynb |
| 8 | 08-gradients-idea | M2 · Autograd | AutogradGraph | conceptual | m2_autograd.ipynb |
| 9 | 09-requires-grad-backward | M2 · Autograd | AutogradGraph | `requires_grad_`,`backward`,`.grad` | m2_autograd.ipynb |
| 10 | 10-computation-graphs | M2 · Autograd | AutogradGraph | graph, `retain_graph` | m2_autograd.ipynb |
| 11 | 11-no-grad-detach | M2 · Autograd | — | `torch.no_grad`,`detach` | m2_autograd.ipynb |
| 12 | 12-loss-functions | M3 · Optimization | — | `nn.MSELoss`,`nn.CrossEntropyLoss` | m3_optim.ipynb |
| 13 | 13-gradient-descent | M3 · Optimization | GradientDescent | manual update | m3_optim.ipynb |
| 14 | 14-optimizers | M3 · Optimization | GradientDescent | `optim.SGD`,`optim.Adam`,`zero_grad`,`step` | m3_optim.ipynb |
| 15 | 15-training-loop | M3 · Optimization | TrainingLoop | full loop pattern | m3_optim.ipynb |
| 16 | 16-neurons-activations | M4 · Neural networks | ActivationExplorer | `nn.ReLU`,`nn.Sigmoid`,`nn.Tanh` | m4_nn.ipynb |
| 17 | 17-nn-module | M4 · Neural networks | — | `nn.Module`,`nn.Linear`,`forward` | m4_nn.ipynb |
| 18 | 18-first-mlp | M4 · Neural networks | TrainingLoop | `nn.Sequential` MLP | m4_nn.ipynb |
| 19 | 19-train-on-data | M4 · Neural networks | TrainingLoop | end-to-end classification | m4_nn.ipynb |
| 20 | 20-dataset-dataloader | M5 · Data | — | `Dataset`,`DataLoader`,`__getitem__` | m5_data.ipynb |
| 21 | 21-transforms-batching | M5 · Data | — | `torchvision.transforms`, batching | m5_data.ipynb |
| 22 | 22-convolution | M6 · CNNs | ConvVisualizer | `nn.Conv2d` | m6_cnn.ipynb |
| 23 | 23-pooling | M6 · CNNs | ConvVisualizer | `nn.MaxPool2d` | m6_cnn.ipynb |
| 24 | 24-cnn-classifier | M6 · CNNs | — | CNN on MNIST | m6_cnn.ipynb |
| 25 | 25-overfitting | M7 · Training well | — | train/val split | m7_training.ipynb |
| 26 | 26-regularization | M7 · Training well | — | `nn.Dropout`, `weight_decay` | m7_training.ipynb |
| 27 | 27-lr-scheduling | M7 · Training well | — | `lr_scheduler.StepLR` | m7_training.ipynb |
| 28 | 28-gpu | M7 · Training well | — | `cuda`,`.to(device)` | m7_training.ipynb |
| 29 | 29-save-load | M8 · Beyond | — | `state_dict`,`save`,`load_state_dict` | m8_beyond.ipynb |
| 30 | 30-transfer-learning | M8 · Beyond | — | `torchvision.models`, freezing | m8_beyond.ipynb |
| 31 | 31-where-next | M8 · Beyond | — | conceptual + links | m8_beyond.ipynb |
| 32 | 32-capstone | Capstone | — | full image classifier | capstone.ipynb |

- [ ] **Step 1 (M1):** Author lessons order 2–7. Build + preview. Commit `feat: M1 tensor lessons`.
- [ ] **Step 2 (M2):** Author order 8–11. Build + preview. Commit `feat: M2 autograd lessons`.
- [ ] **Step 3 (M3):** Author order 12–15. Build + preview. Commit `feat: M3 optimization lessons`.
- [ ] **Step 4 (M4):** Author order 16–19. Build + preview. Commit `feat: M4 neural network lessons`.
- [ ] **Step 5 (M5):** Author order 20–21. Build + preview. Commit `feat: M5 data lessons`.
- [ ] **Step 6 (M6):** Author order 22–24. Build + preview. Commit `feat: M6 CNN lessons`.
- [ ] **Step 7 (M7):** Author order 25–28. Build + preview. Commit `feat: M7 training lessons`.
- [ ] **Step 8 (M8 + Capstone):** Author order 29–32. Build + preview. Commit `feat: M8 + capstone lessons`.

(Each commit uses the full `git -c user.name=... -c user.email=...` form.)

---

## Phase 5 — Polish, accessibility, README, final deploy

### Task 5.1: README

**Files:** Create `README.md`

- [ ] **Step 1:** Write README: what the site is, live URL, local dev (`npm install`, `npm run dev`, `npm test`), how content is structured, how to add a lesson, and how deploy works. Commit.

### Task 5.2: Accessibility + responsive pass

- [ ] **Step 1:** Verify: every widget control is a real labeled form element; canvases have a text `aria-label` or adjacent caption; color is never the only signal (use shape/labels in scatter + decision boundary); sidebar collapses < 780px; focus styles visible. Fix issues found. Build. Commit `fix: accessibility and responsive polish`.

### Task 5.3: Full verification + deploy

- [ ] **Step 1:** `npx vitest run` → all green.
- [ ] **Step 2:** `npm run build` → exit 0; note the page count.
- [ ] **Step 3:** `npm run preview` and click through home → each module's first lesson → a Colab button (opens correct notebook URL) → prev/next at the boundaries.
- [ ] **Step 4:** Push to `main`; `gh run watch` the deploy; confirm `curl -o /dev/null -w "%{http_code}" https://yangsong97.github.io/interactive-pytorch/` returns 200 and spot-check a lesson URL live.
- [ ] **Step 5:** Final commit if needed; report the live URL.

---

## Self-Review (completed during planning)

**Spec coverage:** §3 interactivity model → Phases 2–3 widgets + Task 0.x Colab plumbing + Task 4.3 notebooks. §4 stack → Phase 0. §5 site structure → Tasks 1.5, 4.1, 4.2, 1.6. §6 curriculum → Task 4.5 table (all modules M1–M8 + capstone). §7 widgets (all 6) → Tasks 3.1–3.6. §8 lesson anatomy → Lesson.astro + CodeBlock/Colab/Quiz. §9 contribution points → Tasks 2.4 (A) and 2.6 (B), flagged ⭐. §10 deployment → Tasks 0.2, 0.3, 5.3. §11 non-goals respected (no backend; M5/M7/M8 prose+Colab, no widgets). §12 success criteria → Task 5.3 + 5.2. §13 risks → walking skeleton (Phase 0), pinned-via-`astro add`, `withBase` helper, in-repo notebooks.

**Placeholder scan:** No "TBD/implement later" left undefined. The two `throw new Error('not implemented')` bodies are intentional contribution scaffolds with tests + reference solutions provided (Tasks 2.4, 2.6).

**Type consistency:** `Optimizer {params, velocity}` used consistently (2.4 ↔ 3.3). `Value` API (`add/mul/sub/pow/relu/tanh/backward`) defined in 2.2, used in 2.6 and 3.4. `MLP.forward/parameters/trainEpoch` and `mseLoss` consistent (2.6 ↔ 3.5). `convolve2d`/`KERNELS` (2.7 ↔ 3.6). `withBase` (1.3) used in 1.5, 4.1. `makeMoons`/`mulberry32` (2.5) used in 2.6, 3.5. Frontmatter fields match `content.config.ts` (1.4 ↔ 4.4/4.5).

**Note:** `CodeBlock.astro` is listed in the file structure for an optional copy-button wrapper; Astro's built-in Shiki highlighting already styles ```python blocks, so it is optional polish, not on the critical path.
