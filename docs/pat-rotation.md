# PAT rotation + git credential cleanup

The personal access token used to push to `coreonai/core` was passed
in plaintext during a Claude Code session. Treat that token as
**potentially leaked** even if no one besides you saw the transcript:
prompts get logged, terminals scroll back, conversation summaries
persist. Rotate now, then move to SSH so no PAT lives in plaintext on
this machine again.

## 1. Revoke the leaked token (do this first, ~30s)

1. Open https://github.com/settings/tokens (you must be signed in as
   the GitHub user that owns `coreonai`).
2. Find the token you used for workLLM pushes — usually labelled
   "workLLM" or similar, value starting `ghp_...`.
3. Click **Delete**. The token is invalidated immediately; any
   process still using it will start failing on the next push.

If you can't remember which token: in the same Tokens page, the
**Last used** column tells you which token authenticated the most
recent `coreonai/core` pushes.

## 2. Issue a replacement PAT (only if you don't want SSH — see §4)

If you're staying on HTTPS:

1. https://github.com/settings/tokens → **Generate new token (classic)**.
2. **Note**: `workLLM coreonai push (rotated YYYY-MM-DD)` — datestamping
   helps the next rotation.
3. **Expiration**: 90 days. Pick the shortest expiration that doesn't
   inconvenience you; PATs can't be safely "permanent."
4. **Scopes**: only `repo`. You don't need `workflow`, `admin:org`, or
   anything else for normal push/pull. Smaller scope = smaller blast
   radius if leaked again.
5. Click **Generate token**, copy the value once (GitHub shows it
   exactly once), then update local storage in §3.

## 3. Replace the cached credential

The repo currently uses the `store` credential helper, which stashes
tokens at `~/.git-credentials` in plaintext.

**Option A — replace in place:**

```bash
# Edit ~/.git-credentials with your new PAT.
# Find the line:
#     https://<user>:<old-token>@github.com
# Replace <old-token> with the new value. Save.
$EDITOR ~/.git-credentials

# Verify:
git -C /raid/users/paul/workLLM ls-remote origin >/dev/null && echo "auth OK"
```

**Option B — drop the entry and re-prompt:**

```bash
# Reject the cached entry (works for `store` helper).
printf 'protocol=https\nhost=github.com\n\n' | git credential-store --file=$HOME/.git-credentials erase

# Next push will prompt for username + token. Enter `paul-yu` (or your
# coreonai login name) and the new PAT as the password. The store helper
# will save it back, so subsequent pushes are silent.
```

## 4. Better: switch to SSH (recommended)

SSH keys live in your `ssh-agent` (or on disk encrypted with a
passphrase) — there's no plaintext token in `~/.git-credentials` and
no expiration to chase. One-time setup:

```bash
# Generate a key dedicated to GitHub coreonai (not your default key).
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_coreonai -C "paul.yu@coreon.ai"

# Print the public key:
cat ~/.ssh/id_ed25519_coreonai.pub
# → ssh-ed25519 AAAA... paul.yu@coreon.ai
```

Then on GitHub:

1. https://github.com/settings/keys → **New SSH key**.
2. Title: `paul-workstation`. Key type: `Authentication`. Paste the
   public key. Save.

Configure ssh and switch the remote:

```bash
# Add to ~/.ssh/config — only needed if you have multiple GitHub keys
# or want a non-default identity for this account.
cat >>~/.ssh/config <<'EOF'

Host github.com-coreonai
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_coreonai
    IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config

# Switch remote (drop the HTTPS URL with embedded credentials, if any):
git -C /raid/users/paul/workLLM remote set-url origin \
    git@github.com-coreonai:coreonai/core.git

# Verify:
git -C /raid/users/paul/workLLM fetch origin
```

After SSH works, you can also delete the now-unused entry from
`~/.git-credentials` to make sure nothing falls back to HTTPS+PAT.

## 5. Verify nothing leaked into commits

```bash
# Search the whole history for the leaked-token prefix.
git -C /raid/users/paul/workLLM log -p --all | grep -E 'ghp_[A-Za-z0-9]{20,}' && \
    echo "FOUND PAT in history — rewrite required" || \
    echo "history clean"
```

For workLLM specifically: as of the last verification this returned
**clean**. Pushes used PATs only as URL credentials at runtime; they
never landed in a tracked file.

If a future leak does land in a commit, the only correct response is
`git filter-repo` (or `BFG Repo-Cleaner`) to rewrite history, force-push,
*and* still rotate the PAT — rewriting history doesn't unleak a token
that's already been seen.

## Recap

- **Now (you):** revoke the leaked PAT (§1).
- **Within 5 minutes (you):** either replace it (§2 + §3) or move to
  SSH (§4). SSH is the recommended path.
- **Anytime:** verify history is clean (§5).
- **Going forward:** never paste a PAT into a chat, terminal scrollback,
  or shared logs. If you need to share a PAT with an automation agent,
  scope it minimally + give it a short expiration + rotate after use.
