# Design note: auto-updater (kernel-start, window-only prompt)

**Status:** **Proposed — not implemented.** Depends on the single-release-location
change (PR #47 / `release_bundle.yml`): all three wheels — `biopb-mcp`, `biopb`,
`biopb-tensor-server` — now ship attached to one biopb-mcp GitHub Release per
`v*` tag. That is the precondition that makes an updater tractable: one place to
check, one version-consistent set to install, no PyPI-vs-server-release split to
reconcile.

This note fixes the hard constraints the updater must honor and works through the
design they imply:

1. **The update prompt is a popup shown once, at kernel start.**
2. **It appears only when a napari window exists** (non-headless).
3. **A consistent update means restarting all three processes together** — the
   updater stages an install and applies it on a clean full restart; a partial
   restart is allowed only behind an explicit compatibility warning.

## Why the prompt is kernel-start-only and window-only

**Kernel-start-only.** The kernel is the long-lived process that hosts the
viewer; `_bootstrap.py` runs once via `exec_lines` before the kernel services any
tool call, and the daemon now *outlives the client and is shared across clients*
(one kernel, one viewer — see the daemon-migration note). "Once at kernel start"
is therefore the lowest-noise cadence: one check per kernel lifetime, no polling
thread to own, no mid-session interruption of the agent or a running job, and a
`restart_kernel` (or window-close → idle → `start_kernel`) re-checks for free.
Checking more often buys nothing — the install cannot be *applied* without a full
restart (below), so there is no value in noticing a release sooner than the next
start.

**Window-only.** Accepting an update is a *human* decision — it replaces the code
that will run and forces a restart — so the prompt must reach a human at a
screen. A headless kernel has no human at a window: it is agent-driven over MCP,
`viewer` is the `_HeadlessViewer` sentinel, and a popup would be invisible or
raise off the sentinel; the agent cannot consent to replacing its own runtime.
The deployment an auto-updater actually *serves* is the windowed installer /
bundled app anyway — headless deployments are source/CI checkouts that update
through `uv sync` / `pip` against `pyproject.toml` pins, which a runtime updater
must not fight. The gate is the `headless` boolean bootstrap already computes.

## The hard part: three processes, one version

A biopb-mcp session is **three processes with independent lifecycles**, and a
correct update has to move all three to the same release at once:

| Process | Code it runs | Lifecycle | Brought up by |
|---|---|---|---|
| **MCP server / launcher (daemon)** | `biopb-mcp` (+ mcp SDK) | Outlives the client, shared across clients; does **not** self-restart | The app / installer launch; the stdio shim spawns it detached |
| **Kernel** (child Jupyter) | `biopb-mcp` bootstrap/jobs **+ `biopb`** (TensorFlightClient) + napari/dask | Lazily started on `start_kernel`; torn to idle on window close; `restart_kernel` respawns | `KernelHost` in the daemon |
| **Data server** | **`biopb-tensor-server` + `biopb`** (Arrow Flight) | Independent; may be an app-autostarted child *or* external/remote | `biopb server start` via `start_local_server()` / autostart, or pre-existing |

These are not loosely coupled. The wheels are **version-paired on purpose** —
the integration group pins `biopb` and `biopb-tensor-server` to the *same
monorepo SHA* — because the kernel's `biopb` client and the data server's
`biopb-tensor-server` speak a **matched Arrow Flight protocol**, and the kernel's
`biopb-mcp` bootstrap and the daemon's `biopb-mcp` `KernelHost` are two halves of
one IPC contract. Update the wheels on disk and the **running** processes keep
their old imported code until each restarts. So a partial restart produces
exactly the skew the pairing exists to prevent:

- **Kernel only** (`restart_kernel`) → new `biopb` client talks to an **old
  `biopb-tensor-server`** (Flight-protocol mismatch) *and* runs new bootstrap
  against an **old daemon** `KernelHost` (IPC mismatch).
- **Kernel + daemon, external data server** → still a client/server Flight skew
  against the unchanged remote server.
- **Daemon only** → the kernel it respawns picks up new `biopb-mcp` but the data
  server is still old.

**Conclusion:** the only consistent apply is to **tear down and relaunch all
three on the new wheels together** — practically, a single full app restart: the
launcher exits and is relaunched, which on a clean boot (a) installs any staged
update, (b) starts the daemon on new `biopb-mcp`, (c) lazily starts the kernel on
new `biopb`, and (d) autostarts the data server on new `biopb-tensor-server`. One
coordinated boot = one version everywhere. The MCP client (e.g. Claude Code)
reconnects to the fresh daemon (the shim replays `initialize`).

**Opt-out is allowed, but warned.** The user may decline the update entirely
(keep running the current version — perfectly fine), or decline the *full
restart* and keep working after a partial/kernel-only restart. The latter is
where the warning is mandatory: surface plainly that **running mixed versions
across the kernel, daemon, and data server may break compatibility** (Arrow
Flight protocol, daemon↔kernel IPC) and that the staged update will only become
consistent after a full restart. Never silently apply a partial update as if it
were complete.

## Applying the update: stage in the kernel, install on a clean restart

The popup lives **in the kernel**, but the kernel must not be the thing that
`pip install`s into the venv all three processes are importing from:

- On **Windows**, loaded `.pyd`/`.dll` files are locked — overwriting a package
  while three processes have it imported fails. (POSIX swaps inodes fine, but the
  consistency argument above means we don't want a live partial install anyway.)
- The kernel is a *child* of the daemon; having the child mutate the parent's
  runtime and then signal the parent to die is fragile.

So split **stage** from **apply**:

1. **Stage (kernel, on "Update now").** Download the three wheels from the new
   release into a staging dir and drop a pending-update marker (target version +
   wheel paths). No install yet. Verify wheels before trusting them (see Trust).
2. **Apply (launcher / installer, on clean start).** The app's launch wrapper
   checks the marker *before importing the stack* and runs the offline, atomic
   install: `pip install --upgrade --no-index --find-links <stage> biopb-mcp
   biopb biopb-tensor-server` — one version-consistent triple, no PyPI/server-
   release skew, then clears the marker and proceeds to boot all three.
3. **Trigger the restart.** After staging, the popup offers **Restart now** — the
   kernel signals the daemon to shut down the whole stack (the messenger dies
   with it); the user/app relaunches, the apply step runs, everything comes up
   paired. Or **Restart later** — the staged update simply applies on the next
   clean launch.

**Frozen PyInstaller bundle (`sys.frozen`, already checked in `_shim.py`).** A
frozen app cannot `pip`-upgrade itself; a real update is swapping the whole
platform bundle (heavy, OS-specific, its own project). For v1 the popup here is
**notify + link** — "vX.Y.Z is available", button opens the release page — not
self-install.

**External / remote data server.** If the data server is remote
(`BIOPB_TENSOR_URL` points off-box), the updater **cannot** restart it. The
paired client/server contract still applies, so the popup must warn that
compatibility depends on the server operator updating too; the local apply only
covers the daemon + kernel.

## What it checks

- **Local version:** `biopb_mcp.__version__` (setuptools_scm via `_version.py`,
  e.g. `0.6.5.dev4+gf5055bf63`). Parse with `packaging.version.Version`.
- **Remote version:** `GET
  https://api.github.com/repos/biopb/biopb-mcp/releases/latest` →
  `tag_name` (`v0.7.0`) + `assets` (the three wheels + bundles). Unauthenticated
  is fine (60 req/hr/IP; one check per kernel start). `/releases/latest` excludes
  prereleases; the `prerelease` channel lists `/releases` and takes the newest.
- **Prompt iff `remote > local`**, suppressed when:
  - the local version is a **dev/editable checkout** (`.dev` or a `+g<sha>` local
    segment) — those update through git, not the release;
  - the user previously **skipped** this exact version
    (`mcp.update.skipped_version`), until something newer appears.

The check runs **off the bootstrap path**: a background daemon thread fires it so
the window paints without waiting on the network, and the result is marshaled to
the Qt main thread via the existing `run_on_main(fn)`. **Any** failure (offline,
DNS/TLS, HTTP error, rate-limit, parse, unexpected schema) is swallowed (debug
log) and shows no popup — the updater fails open and never blocks, delays, or
crashes the kernel (consistent with bootstrap's graceful-degradation model;
errors there print `BOOTSTRAP_ERROR`, they don't propagate).

## The popup

A small, clearly actionable modal (`QMessageBox` or a minimal `QWidget`) — the
user asked for a "popup" and an updater needs an affirmative click, which a
napari toast affords weakly. Shown once, after the viewer exists. Actions:

- **Update & restart now** — stage, then signal a full coordinated restart.
- **Update, restart later** — stage; applies on next clean launch (and, if the
  user keeps working past a partial restart, show the **mixed-version
  compatibility warning** above).
- **Later** — do nothing; re-prompts next kernel start.
- **Skip vX.Y.Z** — persist `mcp.update.skipped_version`; re-prompts only when a
  newer release appears.

Modal-at-startup is acceptable (the user is looking at a just-opened window) and
never interrupts a running job — nothing is running yet.

## Config — new `mcp.update` section

Deep-merged like the rest of `DEFAULT_CONFIG`:

- `enabled` (default `true`) — opt-out switch.
- `repo` (default `"biopb/biopb-mcp"`) — the **only** repo checked; config, not
  agent/page-controllable.
- `channel` (`"stable"` | `"prerelease"`).
- `skipped_version` (default `""`) — set by **Skip**.
- `timeout` (default `5.0` s) — check network timeout; short and off-thread.

## Trust / security

The updater installs code that runs with the user's privileges — same trust as
the app — so the threat to close is *substitution*:

- Check and download **only** from the hardcoded canonical `repo` over **HTTPS**.
- GitHub release assets are **not signed by default**. v1 relies on HTTPS +
  repo trust; a hardening follow-up — worth doing before any non-loopback or
  multi-user deployment — is to publish a `SHA256SUMS` (ideally a
  sigstore/cosign signature) with the release and **verify the staged wheels
  against it before the apply step installs them**.

## Failure model (summary)

- Network / parse / rate-limit / schema error → **no popup**, debug log,
  bootstrap unaffected.
- Check is off-thread → never blocks the window paint; never raises into
  `exec_lines`.
- Exactly one check per kernel start; restart re-checks; no polling thread.
- Headless → the whole feature is skipped at the `if not headless` gate.
- A staged-but-unapplied update is inert until a clean launch applies it; a
  declined full restart shows the mixed-version warning rather than pretending
  the update took effect.

## Testing

- **Pure functions, unit-tested:** version compare (`remote > local`, dev/editable
  skip, `skipped_version` suppression), GitHub-API response parsing (mocked JSON:
  tag extraction, asset matching, prerelease selection), and the
  pending-update marker round-trip (stage writes / launcher reads & clears).
- **GUI / install / restart paths** are side-effectful and live behind the
  headless gate; keep them thin and exercise the pure core. The macOS-CI viewer
  skip (`make_napari_viewer` segfaults headless on macOS CI) applies to any test
  that instantiates a viewer.

## Open decisions

1. **Self-install vs notify-only for v1.** Recommendation: self-install for the
   installer-managed venv (model above), notify-only for frozen bundles,
   detected via `sys.frozen`. If the installer's launch-wrapper apply step isn't
   ready, ship notify-only first (popup + release link, every deployment) — it
   satisfies all three constraints and grows the self-install path later.
2. **Who owns the apply step.** The clean-restart install belongs in the
   *installer's launch wrapper*, which lives in the separate installer repo. This
   note specifies the contract (staging dir + marker format); implementing the
   apply side is a coordinated change there.
