# Git LFS setup

Large checkpoints (~50–250 MB each) live under `checkpoints/` and are
currently `.gitignore`d. When you eventually want to publish a few
reference checkpoints with the repo (e.g. KoWiki 50M @ 30K steps, the
distilled student) use Git LFS so they don't bloat the pack.

## One-time per machine

```bash
# Ubuntu / Debian
sudo apt install git-lfs

# macOS
brew install git-lfs

# Activate the smudge/clean filters in your global config:
git lfs install
```

## In a fresh clone

```bash
git lfs pull        # fetch the LFS payloads after the regular clone
```

## Adding a new checkpoint

`.gitattributes` already routes `*.safetensors`, `*.bz2`, `*.bin`,
`*.pt`, and `*.gguf` through LFS, so a normal `git add` is enough:

```bash
# remove the gitignore entry first if you want to actually commit it
git add checkpoints/kowiki_50m_30k.safetensors
git commit -m "ship 30K-step KoWiki checkpoint"
git push
```

The pre-receive hook on the remote (GitHub) will accept the pointer
file and store the payload in LFS.

## Migrating already-committed binaries to LFS

If a `.safetensors` slips in pre-LFS (or you want to retroactively move
existing binaries):

```bash
# DESTRUCTIVE: rewrites history on the current branch.
git lfs migrate import --include="*.safetensors,*.bz2" --everything
git push --force-with-lease origin <branch>
```

`--force-with-lease` is the safe variant of `--force` — it refuses to
overwrite if someone else pushed in the meantime.

## Quotas

GitHub free tier: **1 GB storage + 1 GB/month bandwidth** per account.
A single 50M checkpoint is ~190 MB, the 50M-long one is the same, the
12M student is ~48 MB, the bz2 dump is ~1.2 GB (would blow the free
tier on its own — keep it gitignored, fetch from upstream instead).

For coreonai/core specifically: don't commit the raw `kowiki-latest-pages-articles.xml.bz2`.
The dump is reproducibly downloadable from
`https://dumps.wikimedia.org/kowiki/latest/`.

## What's currently gitignored vs. LFS-eligible

`.gitignore` excludes the whole `checkpoints/` and `data/` trees today.
That's the right default — they're regenerable. LFS is the escape
hatch for the few you want to publish with the repo (e.g. the
"reference" 30K-step checkpoint that backs the README results table).
When you decide to publish one, remove that single path from
`.gitignore` and `git add` it; LFS does the rest.
