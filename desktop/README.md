# HalalifyVid desktop app

A window for people who do not use a terminal: drop a video or paste a link,
choose who to blur and what to do with the sound, press one button.

The app does none of the work itself. It starts the `halalify` program with
`--progress json` and draws the events the program sends; see "The desktop
app" in [docs/DESIGN.md](../docs/DESIGN.md).

```text
desktop/
  ui/          the screens: plain HTML, CSS and JavaScript, no build step
  src-tauri/   the window (Tauri 2): starts halalify, passes its events on
```

## Build

Build the program first, then the app:

```text
cargo build --release
cargo build --release --manifest-path desktop/src-tauri/Cargo.toml
desktop\src-tauri\target\release\HalalifyVid.exe
```

A development build finds `halalify` in the repository's `target/release`. A
release has both programs in one folder; `scripts/package.sh` puts them there.
`HALALIFY_EXE` names the program explicitly, and `HALALIFY_HOME` works as it
does for the program.

Windows needs the WebView2 runtime (part of Windows 11, installed with
Microsoft Edge on Windows 10).

## Working on the screens

The screens also open in an ordinary browser, where `ui/preview.js` stands in
for the program with made-up progress:

```text
python -m http.server 5178 --directory desktop/ui
```

- `http://localhost:5178/` shows the home screen;
- `?state=fresh` starts at the one-time setup;
- `?fail=1` makes the next job end in an error.

All words on screen are in `ui/i18n.js`, in English and Arabic.

## Tests

```text
cargo test --release --manifest-path desktop/src-tauri/Cargo.toml
```
