# Music beds

Drop `.mp3`, `.wav`, `.m4a`, `.ogg` or `.flac` files here and they appear in the
music dropdown on the New Short page. Nothing is bundled, because music licences
do not travel.

The track is ducked automatically under the narration (`sidechaincompress` keyed
off the voice), so a bed at normal level will not fight the words.

Choosing "Built-in ambient pad" instead synthesises a plain minor-triad pad with
ffmpeg at render time. Nothing is sampled, so it carries no licence obligation.
