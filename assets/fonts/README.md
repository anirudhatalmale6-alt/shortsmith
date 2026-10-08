# Caption fonts

Bundled so a fresh clone burns captions in the intended typeface instead of
silently falling back to whatever `fontconfig` happens to offer. ffmpeg is
pointed here with the `ass` filter's `fontsdir` option, so these do not need to
be installed system-wide.

| File | Family | Licence |
|---|---|---|
| `Anton-Regular.ttf` | Anton | SIL Open Font License 1.1 |
| `Poppins-ExtraBold.ttf` | Poppins ExtraBold | SIL Open Font License 1.1 |
| `Montserrat.ttf` | Montserrat (variable) | SIL Open Font License 1.1 |

All three are from the Google Fonts repository and are free for commercial use,
including in monetised video. To add your own, drop the TTF here and name the
family in `shortsmith/providers/captions.py`.
