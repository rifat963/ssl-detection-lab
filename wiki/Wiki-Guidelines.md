# Wiki Guidelines

How to write and maintain these pages. Read this before adding one.

## What belongs where

| Location | Purpose |
|---|---|
| `README.md` | What the project is, why you'd use it, how to start. The shop window. |
| `wiki/` | Task-oriented "how do I…" guides and reference detail. |
| Docstrings | Why a specific piece of code is the way it is. |
| Notebooks | Teaching a concept interactively, with plots and interpretation. |

If a README section grows past roughly a screen of detail, move the detail to a wiki page and
leave a link. The README should stay skimmable.

Do not duplicate content across pages. Link instead — duplicated text drifts out of sync, and the
stale copy is the one people find.

## Audience

Students who know basic Python and have seen a PyTorch training loop, plus researchers evaluating
the package. Assume neither knows this codebase.

Explain the *why*, not just the *what*. "Set `teacher_temperature` below `student_temperature`" is
a rule to memorize; "the gap is what sharpens the target distribution" is something a reader can
reason from later.

## Page structure

1. `# Title` matching the filename
2. One or two sentences on what the page covers
3. The most common task first — a reader should be able to copy something useful within a screen
4. Reference detail after that
5. A `## Next` line linking onward

## Writing rules

- **Every code sample must run** as written, against the current API. Untested samples are the
  fastest way to lose a reader's trust.
- **Tables for reference, prose for reasoning.** Do not narrate a parameter list.
- **State constraints with their reason.** `projection_dim` must be divisible by 4 *because the
  2-D sinusoidal embedding splits it four ways*.
- **Prefer the specific.** "Lower `batch_size` first — it is per GPU" beats "adjust settings".
- **No marketing.** No "powerful", "seamless", "cutting-edge".
- **Warn where people actually get hurt.** Blockquote the traps: per-GPU batch size, gradient
  accumulation not creating negatives, SSL loss not being a detection metric, the `yolov12`
  spelling.

## Naming

- Filenames use `Title-Case-With-Hyphens.md` — this is the GitHub wiki convention and the hyphens
  become spaces in the page title.
- `_Sidebar.md` and `_Footer.md` are special GitHub wiki filenames; keep the underscore.
- Link with relative paths (`[Evaluation](Evaluation.md)`) so links work both in the rendered wiki
  and when browsing the repo on GitHub.

## Accuracy obligations

These carry across the whole project — see
[Contributing → Honesty requirements](Contributing.md#honesty-requirements):

- Keep the **adaptation notices** on MAE, DINOv2, and I-JEPA. Never call them reproductions.
- Never imply SSL loss measures detection quality.
- Keep the labelled/unlabelled metric boundary explicit on evaluation and video pages.
- Note licence obligations wherever DINOv3 weights or Ultralytics are mentioned.

## Maintenance

Update the wiki in the **same PR** as the behaviour change. A page that describes last release's
behaviour actively misleads.

When you change a public API, grep the wiki for it:

```bash
grep -rn "transfer_ssl_backbone_to_yolo" wiki/
```

Check links after renaming or moving anything — a folder rename silently breaks every relative
link pointing into it:

```bash
grep -rEo "\]\([^)#][^)]*\)" wiki/ README.md
```

Version-specific claims (dependency floors, `0.9.0`, PyTorch 2.7.1 for DINOv3) live in
[Installation](Installation.md) and the README baseline section. Change them in those two places
only, and do not restate them elsewhere.

## Adding a page

1. Create `wiki/Your-Page.md`
2. Add it to `_Sidebar.md` under the right group
3. Link it from at least one related page's `## Next`
4. Add it to the table on [Home](Home.md) if it is a top-level topic
