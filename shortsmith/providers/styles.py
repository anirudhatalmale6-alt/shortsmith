"""Visual style presets and the look lock.

Two different problems, solved together.

**Style** is what the video should look like: cinematic film still, illustrated,
anime, photoreal, noir. A preset is a prompt suffix plus a negative prompt, so
switching the whole channel's aesthetic is one dropdown.

**The look lock** is the harder one: making six separately generated images feel
like six frames of the same film rather than six unrelated pictures. Diffusion
models have no memory between calls, so consistency has to be forced from the
outside. Three levers, all applied here:

1. One seed base per video, offset by beat index. Same base, same model, similar
   prompt means similar composition, palette and lighting.
2. The same style suffix on every beat, word for word. Changing even the lens
   description between beats visibly changes the grade.
3. A character sheet: one short description of the protagonist, repeated verbatim
   in every prompt that features them. This is what stops the man in beat 2
   becoming a different man in beat 5.

None of it gives you true character identity (that needs an IP-Adapter or a
LoRA, which is a local-GPU feature). It gives you a consistent film, which is
what a story Short actually needs.
"""

from __future__ import annotations

import hashlib
import re

# key -> (label, prompt suffix, negative additions)
STYLE_PRESETS: dict[str, tuple[str, str, str]] = {
    "cinematic": (
        "Cinematic film still",
        "cinematic film still, 35mm anamorphic, shallow depth of field, "
        "volumetric light, filmic colour grade, subtle film grain, "
        "highly detailed, no text",
        "cartoon, illustration, 3d render, cgi",
    ),
    "photoreal": (
        "Photographic",
        "photograph, natural light, realistic skin texture, 50mm lens, "
        "documentary photography, high dynamic range, no text",
        "illustration, painting, cartoon, anime, 3d render, oversaturated",
    ),
    "illustrated": (
        "Painterly illustration",
        "digital painting, painterly brushwork, muted palette, soft edges, "
        "storybook illustration, atmospheric, no text",
        "photograph, photorealistic, 3d render",
    ),
    "anime": (
        "Anime",
        "anime key visual, cel shaded, clean linework, dramatic lighting, "
        "makoto shinkai inspired skies, no text",
        "photograph, photorealistic, western cartoon, 3d render",
    ),
    "noir": (
        "High contrast noir",
        "high contrast black and white photography, hard directional light, "
        "deep shadows, heavy grain, film noir, no text",
        "colour, saturated, cartoon, 3d render",
    ),
    "dark_fantasy": (
        "Dark fantasy",
        "dark fantasy concept art, moody atmosphere, dramatic rim light, "
        "intricate detail, matte painting, no text",
        "photograph, flat lighting, cartoon",
    ),
}

DEFAULT_STYLE = "cinematic"

BASE_NEGATIVE = (
    "text, words, letters, caption, subtitle, watermark, signature, logo, "
    "username, lowres, blurry, jpeg artifacts, extra fingers, deformed hands, "
    "disfigured, bad anatomy, cropped head, frame, border"
)

# Words a beat prompt uses to describe the protagonist. Matching on these is
# crude, but it only has to be good enough to decide whether the character sheet
# belongs in this particular prompt.
_PERSON = re.compile(
    r"\b(a|an|the)\s+("
    r"man|woman|boy|girl|father|mother|son|daughter|sister|brother|widow|"
    r"nurse|teacher|schoolteacher|driver|fisherman|postman|technician|pilot|"
    r"engineer|archivist|diver|medic|janitor|sitter|tenant|counsellor|"
    r"locksmith|operator|detective|librarian|clerk|instructor|journalist|"
    r"ranger|picker|dropout|cook|kid|barista|paramedic|volunteer|owner|"
    r"accountant|graduate|neighbour|waitress|person"
    r")\b",
    re.I,
)


def style_suffix(style: str) -> str:
    return STYLE_PRESETS.get(style, STYLE_PRESETS[DEFAULT_STYLE])[1]


def style_negative(style: str) -> str:
    extra = STYLE_PRESETS.get(style, STYLE_PRESETS[DEFAULT_STYLE])[2]
    return f"{BASE_NEGATIVE}, {extra}"


def seed_base(key: str) -> int:
    """One stable base seed per video, derived from its title."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def character_sheet(prompts: list[str]) -> str:
    """Build one short character description from the beats that mention a person.

    Taken from the first beat that names one, because that is the establishing
    shot and the rest of the story should match it rather than the other way
    round.
    """
    for prompt in prompts:
        match = _PERSON.search(prompt)
        if not match:
            continue
        # Keep the noun plus a few following words: "a father in a dark coat".
        tail = prompt[match.start():]
        words = tail.split(",")[0].split()
        return " ".join(words[:8]).strip(" ,.")
    return ""


def apply_look(
    prompt: str,
    style: str,
    sheet: str = "",
    tone_hint: str = "",
) -> str:
    """Compose the final prompt: beat, character sheet, tone, style suffix.

    Order matters for every diffusion model worth using: the subject goes first
    and the style modifiers last, because earlier tokens carry more weight.
    """
    # The beat prompt often carries style words of its own ("cinematic film
    # still, 35mm, ..."). Left in, they fight the preset and the model gets the
    # same modifier twice at different weights, so strip the overlap first.
    suffix_terms = {t.strip().lower() for t in style_suffix(style).split(",") if t.strip()}
    kept = [
        fragment.strip()
        for fragment in prompt.split(",")
        if fragment.strip() and fragment.strip().lower() not in suffix_terms
    ]
    prompt = ", ".join(kept) or prompt.strip()

    parts = [prompt.strip().rstrip(".")]
    if sheet and _PERSON.search(prompt):
        # Only inject the sheet where a person is actually in shot; forcing a
        # character into a close-up of a letter produces a stray face.
        if sheet.lower() not in prompt.lower():
            parts.append(f"the same character throughout: {sheet}")
    if tone_hint:
        parts.append(tone_hint)
    parts.append(style_suffix(style))
    seen: list[str] = []
    for part in parts:
        if part and part.lower() not in {p.lower() for p in seen}:
            seen.append(part)
    return ", ".join(seen)
