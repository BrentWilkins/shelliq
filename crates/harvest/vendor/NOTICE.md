# Vendored content: tldr-pages

`tldr-pages.en.zip` in this directory is a filtered, English-only subset of the
[tldr-pages](https://github.com/tldr-pages/tldr) project, containing only the `common/`,
`linux/`, and `osx/` page directories — the platforms shelliq targets.

- Source: https://github.com/tldr-pages/tldr
- Pinned release: v2.3 (asset `tldr-pages.en.zip`)
- License: [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://github.com/tldr-pages/tldr/blob/main/LICENSE.md)
- Copyright: the tldr-pages contributors

No page content has been modified; the archive is re-zipped with the `android/`,
`cisco-ios/`, `dos/`, `freebsd/`, `netbsd/`, `openbsd/`, `sunos/`, and `windows/`
directories, and non-English translations, removed to keep the vendored footprint small.

shelliq reads examples out of this archive at harvest time to build the task-language
vocabulary bridge described in `PLAN.md`; every flag mentioned in an example is validated
against the flags actually harvested for the local machine's target before it is trusted,
so a generic tldr example never becomes a machine-specific fact on its own.
