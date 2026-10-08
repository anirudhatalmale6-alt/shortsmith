"""Offline story writer.

Runs with no model, no API key and no internet.  It exists so a scheduled slot
is never missed because Ollama was restarting, and so a fresh clone produces a
real video on the first run.

It is not a Markov chain or a bag of stock sentences.  Each genre has a three
act beat plan; each beat is a sentence template with typed slots, and the slot
pools are large enough that two runs on the same topic read as different
stories.  A per-render seed makes any individual result reproducible.

Two things it gets right that a naive template engine gets wrong: the character
and the pronouns agree in gender for the whole script, and every verb phrase
carries its base, past and -ing forms so "would {ritual}" and "{name} {ritual}"
are both grammatical.
"""

from __future__ import annotations

import random
import re

from .script_types import Scene, ScriptResult

# ---------------------------------------------------------------------------
# genre detection
# ---------------------------------------------------------------------------

GENRE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "sad": (
        "sad", "heartbreak", "grief", "loss", "lonely", "loneliness", "goodbye",
        "funeral", "tears", "tragic", "tragedy", "miss you", "regret", "orphan",
        "breakup", "dying", "cancer", "widow", "emotional",
    ),
    "scifi": (
        "sci-fi", "scifi", "science fiction", "space", "alien", "robot", "ai ",
        "android", "mars", "galaxy", "future", "time travel", "cyberpunk",
        "spaceship", "colony", "dystopia", "simulation", "quantum", "starship",
    ),
    "horror": (
        "horror", "scary", "creepy", "haunted", "ghost", "nightmare", "demon",
        "ritual", "cursed", "monster", "basement", "3am", "3 am", "paranormal",
    ),
    "mystery": (
        "mystery", "unsolved", "disappeared", "vanished", "missing", "detective",
        "cold case", "conspiracy", "secret", "hidden", "strange", "true crime",
    ),
    "motivational": (
        "motivation", "motivational", "success", "discipline", "mindset",
        "hustle", "grind", "inspire", "inspiring", "never give up", "comeback",
        "self improvement", "stoic", "gym",
    ),
    "wholesome": (
        "wholesome", "kindness", "heartwarming", "faith in humanity", "rescue",
        "adopted", "reunion", "stranger", "good deed", "dog", "cat",
    ),
    "revenge": (
        "revenge", "karma", "got what", "payback", "justice", "entitled",
        "satisfying", "petty", "malicious compliance",
    ),
    "facts": (
        "facts", "did you know", "history", "science", "explained", "top 5",
        "top 10", "things you", "psychology", "ocean", "universe", "brain",
    ),
}

DEFAULT_GENRE = "sad"


def detect_genre(topic: str) -> str:
    low = f" {topic.lower().strip()} "
    best, best_hits = DEFAULT_GENRE, 0
    for genre, words in GENRE_KEYWORDS.items():
        hits = sum(1 for word in words if word in low)
        if hits > best_hits:
            best, best_hits = genre, hits
    return best if best_hits else DEFAULT_GENRE


# ---------------------------------------------------------------------------
# slot pools
# ---------------------------------------------------------------------------
# who / name are (text, gender).  Gender is matched across the whole script so
# the pronouns never contradict the character.
# ritual is (base, past, -ing) so every template slot is grammatical.

M, F, N = "m", "f", "n"

POOLS: dict[str, dict[str, tuple]] = {
    "sad": {
        "who": (("a father", M), ("an old woman", F), ("a nurse", F), ("a son", M),
                ("a schoolteacher", F), ("a bus driver", M), ("a young widow", F),
                ("a fisherman", M), ("a retired postman", M), ("a sister", F)),
        "name": (("Mara", F), ("Elena", F), ("Ruth", F), ("Noor", F), ("Petra", F),
                 ("Daniel", M), ("Tomas", M), ("Arjun", M), ("Owen", M), ("Henrik", M)),
        "place": ("a hospital corridor", "an empty kitchen", "a bus stop in the rain",
                  "a half packed apartment", "a winter beach", "a parked car",
                  "a hallway full of boxes", "a church car park"),
        "object": ("an unopened letter", "a voicemail nobody deleted", "a second cup",
                   "a hospital bracelet", "a folded coat", "a single train ticket",
                   "a photograph with one torn corner", "a phone still on the charger"),
        "ritual": (("set two places at the table", "set two places at the table",
                    "setting two places at the table"),
                   ("leave the porch light on", "left the porch light on",
                    "leaving the porch light on"),
                   ("drive the same route home", "drove the same route home",
                    "driving the same route home"),
                   ("answer a phone that never rang", "answered a phone that never rang",
                    "answering a phone that never rang"),
                   ("buy flowers every Sunday", "bought flowers every Sunday",
                    "buying flowers every Sunday")),
        "truth": ("{he} had been saying goodbye the whole time",
                  "the letter had been addressed to {him} all along",
                  "the number had been disconnected for over a year",
                  "the seat had been empty since the accident",
                  "it had been written the night before the accident"),
    },
    "scifi": {
        "who": (("the last technician", M), ("a cargo pilot", F), ("a terraform engineer", M),
                ("a station archivist", F), ("a salvage diver", F), ("a colony medic", M)),
        "name": (("Iris", F), ("Sable", F), ("Ori", F), ("Kade", M), ("Halden", M), ("Vex", M)),
        "place": ("a dead orbital station", "the Martian relay field", "a sleeper ship",
                  "the edge of the exclusion zone", "a flooded data vault",
                  "a colony under a frozen sea"),
        "object": ("a signal that repeats every nine minutes", "a sealed escape pod",
                   "a log entry nobody wrote", "a door with no record of being built",
                   "a crew manifest one name too long", "an answer sent before the question"),
        "ritual": (("run the diagnostic every cycle", "ran the diagnostic every cycle",
                    "running the diagnostic every cycle"),
                   ("log the silence as normal", "logged the silence as normal",
                    "logging the silence as normal"),
                   ("repair the same relay twice a day", "repaired the same relay twice a day",
                    "repairing the same relay twice a day"),
                   ("count the pods each morning", "counted the pods each morning",
                    "counting the pods each morning")),
        "truth": ("the signal was {his} own voice, ageing",
                  "the ship had never left orbit",
                  "the colony had been empty for forty years",
                  "the system had been keeping the crew asleep to keep them alive",
                  "{he} was the backup, and the original was still aboard"),
    },
    "horror": {
        "who": (("a night janitor", M), ("a house sitter", F), ("a new tenant", F),
                ("a camp counsellor", M), ("a locksmith", M), ("a radio operator", F)),
        "name": (("Clara", F), ("Nina", F), ("Delphine", F), ("Jonah", M), ("Ezra", M), ("Rhys", M)),
        "place": ("a building with no fourteenth floor", "a farmhouse past the tree line",
                  "a motel off the interstate", "a flooded basement",
                  "a lighthouse in the off season", "an apartment with one extra room"),
        "object": ("a knock that answers itself", "a camera pointed at an empty hallway",
                   "a key that fits a door nobody opens", "footprints leading only inward",
                   "a note in {his} own handwriting", "a voice on the baby monitor"),
        "ritual": (("check the locks three times", "checked the locks three times",
                    "checking the locks three times"),
                   ("stay upstairs after dark", "stayed upstairs after dark",
                    "staying upstairs after dark"),
                   ("leave the radio on all night", "left the radio on all night",
                    "leaving the radio on all night"),
                   ("count the doors on the way out", "counted the doors on the way out",
                    "counting the doors on the way out")),
        "truth": ("the footage was from last night",
                  "the door had been locked from the inside",
                  "the voice had been recorded in the room {he} was standing in",
                  "nobody had lived there for nine years",
                  "the handwriting was {his}, but the date was tomorrow"),
    },
    "mystery": {
        "who": (("a retired detective", M), ("a librarian", F), ("a postal clerk", M),
                ("a dive instructor", F), ("a journalist", F), ("a park ranger", M)),
        "name": (("Marion", F), ("Cass", F), ("Anya", F), ("Hollis", M), ("Jude", M), ("Byrne", M)),
        "place": ("a town of four hundred people", "a village on the coast road",
                  "a county with one road in", "an old mill town",
                  "a ferry port in the off season", "a valley above the reservoir"),
        "object": ("a bank box opened once a year", "a map with one road erased",
                   "a cassette labelled do not copy", "a storage unit paid for in cash",
                   "a letter postmarked after the funeral", "a photo nobody can place"),
        "ritual": (("file the same report every spring", "filed the same report every spring",
                    "filing the same report every spring"),
                   ("drive past the turning every week", "drove past the turning every week",
                    "driving past the turning every week"),
                   ("pay the rent on a unit {he} never opened",
                    "paid the rent on a unit {he} never opened",
                    "paying the rent on a unit {he} never opened")),
        "truth": ("the handwriting matched the investigating officer",
                  "the car had been reported stolen two days after it was found",
                  "all four witnesses had given the same address",
                  "the photograph had been taken from inside the house",
                  "the file had been requested nineteen times, always by the same name"),
    },
    "motivational": {
        "who": (("a warehouse picker", M), ("a single mother", F), ("a dropout", M),
                ("a line cook", M), ("a night shift nurse", F), ("a kid from a council estate", M)),
        "name": (("Toni", F), ("Joy", F), ("Reza", F), ("Sam", M), ("Malik", M), ("Dmitri", M)),
        "place": ("a gym at five in the morning", "a car park before sunrise",
                  "a kitchen table at midnight", "a library that closes at ten",
                  "a cold stairwell", "a rented room"),
        "object": ("a rejection email", "a notebook with one line per day",
                   "a bank balance in red", "a timer set for twenty five minutes",
                   "a chair nobody saved"),
        "ritual": (("show up anyway", "showed up anyway", "showing up anyway"),
                   ("do one more rep than yesterday", "did one more rep than yesterday",
                    "doing one more rep than yesterday"),
                   ("write the number down every single night",
                    "wrote the number down every single night",
                    "writing the number down every single night"),
                   ("put the phone in the other room", "put the phone in the other room",
                    "putting the phone in the other room")),
        "truth": ("the gap was never talent, it was attendance",
                  "nobody was coming, and that was the good news",
                  "the hard part was never the work, it was the waiting",
                  "the version of {him} that quit would have been right for exactly one more week"),
    },
    "wholesome": {
        "who": (("a bus driver", M), ("a barista", F), ("a retired teacher", F),
                ("a paramedic", M), ("a shelter volunteer", F), ("a corner shop owner", M)),
        "name": (("Grace", F), ("Lena", F), ("Priya", F), ("Hugo", M), ("Carl", M), ("Mo", M)),
        "place": ("a corner shop", "a bus on the last route of the night",
                  "a food bank queue", "a hospital waiting room", "a railway platform"),
        "object": ("a tab that never got paid", "a coat left on a chair",
                   "an envelope with no name", "a birthday cake for a stranger",
                   "a dog with no collar"),
        "ritual": (("keep the kettle on", "kept the kettle on", "keeping the kettle on"),
                   ("pay for the person behind {him}", "paid for the person behind {him}",
                    "paying for the person behind {him}"),
                   ("leave the lights on past closing", "left the lights on past closing",
                    "leaving the lights on past closing"),
                   ("save the last seat", "saved the last seat", "saving the last seat")),
        "truth": ("{he} had been that person once",
                  "{he} had been doing it every week for nine years",
                  "the envelope had been left by the man {he} helped in March",
                  "the dog had found its way back to the same door"),
    },
    "revenge": {
        "who": (("a junior accountant", F), ("a tenant", M), ("a delivery driver", M),
                ("a waitress", F), ("a graduate on probation", F), ("a neighbour", M)),
        "name": (("Ivy", F), ("Dana", F), ("Rae", F), ("Marcus", M), ("Theo", M), ("Paulo", M)),
        "place": ("an open plan office", "a block of flats with a shared drive",
                  "a chain restaurant", "a car park with numbered bays"),
        "object": ("a parking space taken every morning", "an email chain nobody read",
                   "a signed policy nobody expected to be used", "a receipt kept for a year"),
        "ritual": (("document every single incident", "documented every single incident",
                    "documenting every single incident"),
                   ("reply all, politely", "replied all, politely", "replying all, politely"),
                   ("follow the rule exactly as written", "followed the rule exactly as written",
                    "following the rule exactly as written")),
        "truth": ("the policy he wrote applied to him first",
                  "the audit had already been scheduled",
                  "{he} had been copying the regional director in since March",
                  "the bay he kept taking had never been the company's to give"),
    },
    "facts": {
        "who": (("a survey team", N), ("a group of divers", N), ("a research crew", N)),
        "name": (("the team", N), ("the crew", N), ("the lab", N)),
        "place": ("the deep ocean", "a glacier core", "a dig site", "low earth orbit",
                  "a cave system", "a sealed archive"),
        "object": ("a reading nobody could explain", "a sample dated wrong",
                   "a signal on a frequency nothing uses", "a layer that should not exist"),
        "ritual": (("run the test again", "ran the test again", "running the test again"),
                   ("check the instruments twice", "checked the instruments twice",
                    "checking the instruments twice"),
                   ("assume it was equipment error", "assumed it was equipment error",
                    "assuming it was equipment error")),
        "truth": ("the instruments were fine",
                  "it had been recorded four times before and filed as noise",
                  "the date put it two thousand years too early",
                  "the same pattern turns up on three continents"),
    },
}

PRONOUNS: dict[str, dict[str, str]] = {
    M: {"he": "he", "him": "him", "his": "his", "hes": "he's", "himself": "himself"},
    F: {"he": "she", "him": "her", "his": "her", "hes": "she's", "himself": "herself"},
    N: {"he": "they", "him": "them", "his": "their", "hes": "they're", "himself": "themselves"},
}

HOOKS: dict[str, tuple[str, ...]] = {
    "sad": (
        "{who_cap} kept {ritual_ing} for three years. Nobody ever asked why.",
        "{he_cap} found {object} on the table. {he_cap} did not open it for eleven months.",
        "Everyone assumed {he} was fine. {he_cap} had just gotten good at {ritual_ing}.",
        "The hardest part was never the funeral. It was {place}, the morning after.",
    ),
    "scifi": (
        "{who_cap} found {object}. Then {he} checked the date on it.",
        "The station had been empty for forty years. Something was still answering.",
        "They woke {him} up early. That was the first thing that was wrong.",
        "Every colony sent the same message home. Word for word.",
    ),
    "horror": (
        "{who_cap} watched the footage back. There was someone else in the hallway.",
        "The knocking stopped when {he} opened the door. It started again when {he} closed it.",
        "{he_cap} found {object}, and {he} had not written it.",
        "There were nine doors on the way in. There were ten on the way out.",
    ),
    "mystery": (
        "They found the car with the engine still warm. Nobody has explained that yet.",
        "{who_cap} opened the storage unit after nineteen years of paying for it.",
        "Four people saw it. All four gave the same address.",
        "The map is correct except for one road. That road is where it happened.",
    ),
    "motivational": (
        "Nobody clapped. {he_cap} kept going anyway. That is the whole story.",
        "Everyone wants the result. Almost nobody wants {place}, every single morning.",
        "{he_cap} was two years behind. {he_cap} finished four years ahead.",
        "The talent gap is mostly a myth. The attendance gap is not.",
    ),
    "wholesome": (
        "{who_cap} kept {ritual_ing} for nine years and never once mentioned it.",
        "She paid for the man behind her. Nine years later he found out who she was.",
        "Nobody asked {him} to do it. {he_cap} did it every week for a decade.",
        "It cost {him} four pounds. It changed how she saw the entire year.",
    ),
    "revenge": (
        "He took her parking space every morning. So she read the policy.",
        "They told {him} to follow the rules exactly. So {he} did.",
        "He wrote the policy himself. He forgot that it applied to him too.",
        "{he_cap} kept every single receipt. For one year. Quietly.",
    ),
    "facts": (
        "There is a sound in the deep ocean that nothing on earth should be able to make.",
        "They ran the test four times. The instruments were not broken.",
        "This shows up on three continents. Nobody agrees on why.",
        "The sample came back two thousand years too early.",
    ),
}

CTAS: dict[str, tuple[str, ...]] = {
    "sad": ("Some people never stop waiting.", "Tell them while you still can.",
            "Follow for more stories like this one."),
    "scifi": ("Follow if you want the rest of the transmission.",
              "Part two is already on the channel.", "Sleep well."),
    "horror": ("Check your hallway tonight.", "Follow if you can handle part two.",
               "Count the doors."),
    "mystery": ("The file is still open.", "Follow for part two.",
                "Tell me what you think happened."),
    "motivational": ("Start today. Badly. Just start.", "Save this and read it again tomorrow.",
                     "Follow for the next one."),
    "wholesome": ("Be the person in the story.", "Follow for more of these.",
                  "Send this to someone who needs it."),
    "revenge": ("Always read the policy.", "Follow for part two.", "Keep the receipts."),
    "facts": ("Follow for more things nobody can explain.", "Part two is on the channel.",
              "Save this one."),
}

# (narration template, image prompt template)
BEATS: dict[str, tuple[tuple[str, str], ...]] = {
    "sad": (
        ("For three years, {name} would {ritual_base} without thinking about it.",
         "{who} alone in {place}, soft grey window light, quiet, shallow depth of field, 35mm"),
        ("Friends stopped asking after a while. It was easier for everyone that way.",
         "empty chairs in {place}, cold morning light, muted colours, cinematic"),
        ("Then one morning there was {object}, exactly where it had always been.",
         "close up of {object} on a worn wooden surface, single shaft of light, melancholy"),
        ("{he_cap} had walked past it a thousand times. That morning {he} picked it up.",
         "hands lifting {object}, macro, warm lamp light against a cold blue room"),
        ("And that was when {he} understood that {truth}.",
         "{who} sitting very still in {place}, face turned away, rain on glass, heavy mood"),
        ("{he_cap} put it back exactly where it was. Then {he} made two cups anyway.",
         "two cups of tea on a table, one untouched, steam rising, dim kitchen, soft focus"),
    ),
    "scifi": (
        ("{name_cap} was the only one awake on {place}.",
         "{who} alone on {place}, cold cyan emergency lighting, vast empty corridor, sci-fi, volumetric light"),
        ("The routine was simple. Check the hull, log the readings, {ritual_base}.",
         "glowing control panel readouts in a dark room, reflections on a visor, sci-fi"),
        ("On the ninth cycle, the log had an entry {he} had not written.",
         "a single glowing line of text on a dark screen, dust in the air, sci-fi"),
        ("It described {object}, down to the timestamp.",
         "{object}, backlit, sterile metal environment, blue and amber rim light, sci-fi"),
        ("{he_cap} checked it twice, then once more. Then {he} understood that {truth}.",
         "{who} with a slowly dawning expression, lit only by a screen, sci-fi, close up"),
        ("{he_cap} logged it as instrument error. Then {he} turned the lights back off.",
         "a hand switching off a panel, a corridor falling into darkness, sci-fi, cinematic"),
    ),
    "horror": (
        ("{name_cap} took the job because it paid well and nobody else wanted it.",
         "{who} walking into {place} at night, torchlight, deep shadow, horror, 35mm"),
        ("The first week was quiet. In the second week, the sounds started.",
         "a long dark hallway in {place}, one door ajar, cold light spill, horror atmosphere"),
        ("Always at the same hour. Always from the same direction.",
         "an old clock reading three, dust, dim red light, unsettling, horror"),
        ("On the fourteenth night {he} found {object}.",
         "close up of {object}, harsh flashlight beam, heavy grain, horror still"),
        ("{he_cap} did not sleep that night, because {truth}.",
         "{who} frozen in a doorway, lit from behind, face in shadow, horror, high contrast"),
        ("{he_cap} handed in the keys the next morning. The agency sent someone else.",
         "a set of keys left on an empty counter, early grey light, cold and still"),
    ),
    "mystery": (
        ("In {place}, everyone still remembers the week it happened.",
         "wide establishing shot of {place}, overcast, documentary photography, muted palette"),
        ("{name_cap} has been {ritual_ing} ever since, without once explaining why.",
         "{who} looking out of a rain streaked window, documentary style, natural light"),
        ("The official report runs to forty pages. It does not mention {object}.",
         "close up of {object} on a desk beside a thick paper file, desk lamp, noir"),
        ("That detail only surfaced because a clerk filed the wrong copy.",
         "stacks of paper files in a dim archive, one folder pulled out, moody"),
        ("And when somebody finally read it properly, {truth}.",
         "a finger on a line of typed text, magnifier, harsh desk light, noir, close up"),
        ("The case was quietly reclassified. It has not been reopened since.",
         "a filing cabinet drawer closing, an empty office, late afternoon light, melancholy"),
    ),
    "motivational": (
        ("{name_cap} was not the most talented person in the room. Not close.",
         "{who} alone in {place}, pre-dawn blue light, visible breath, cinematic, gritty"),
        ("What {name} had instead was {object}, and one very boring habit.",
         "close up of {object}, harsh overhead light, documentary realism"),
        ("Every day, no exceptions, {name} would {ritual_base}.",
         "a repetitive training moment in {place}, sweat, hard light, cinematic"),
        ("Nobody noticed for the first eleven months. That part is normal.",
         "empty seats in a dark room, a single spotlight, lonely, cinematic"),
        ("By the second year the gap had closed, because {truth}.",
         "{who} standing taller in {place}, golden hour light breaking in, hopeful, cinematic"),
        ("You are not behind. You are early, and nobody has told you yet.",
         "sunrise over a city skyline from a rooftop, warm light, wide shot, hopeful"),
    ),
    "wholesome": (
        ("Every Thursday, {name} worked the late shift in {place}.",
         "{who} in {place}, warm practical lighting, cosy, documentary photography"),
        ("There was always someone short at the till. There always is.",
         "a small counter, coins on a worn surface, warm tungsten light, intimate"),
        ("So {name} started to {ritual_base}. Quietly. Never made a thing of it.",
         "hands passing a small note across a counter, warm light, shallow focus"),
        ("One night a man came back in holding {object}.",
         "close up of {object} held in two hands, warm light, hopeful, soft focus"),
        ("He had been looking for nine years, because {truth}.",
         "two people talking across a counter, warm backlight, emotional, cinematic"),
        ("They still have the same arrangement. Neither has ever mentioned the money.",
         "two mugs on a counter at closing time, warm lamp, empty shop, gentle mood"),
    ),
    "revenge": (
        ("{name_cap} worked in {place}, three desks from the window.",
         "wide shot of {place}, flat fluorescent light, corporate, slightly oppressive"),
        ("Every single morning, the same problem: {object}.",
         "close up of {object}, cold office light, flat corporate photography"),
        ("{name_cap} asked nicely twice. Then {he} stopped asking.",
         "{who} at a desk, calm expression, monitor glow, corporate thriller lighting"),
        ("Instead, {name} started to {ritual_base}. For eleven weeks.",
         "a neat stack of printed documents on a desk, a paperclip, cold light, deliberate"),
        ("Then the review came round, and it turned out {truth}.",
         "a meeting room seen through glass, people seated, one empty chair, cold light, tense"),
        ("The space was reassigned on the Monday. Nobody said a word about it.",
         "an empty numbered parking bay, wet asphalt, grey morning, quiet triumph"),
    ),
    "facts": (
        ("In {place}, a survey picked up {object}.",
         "wide shot of {place}, scientific documentary photography, natural light"),
        ("The team assumed it was equipment error, so they {ritual_past}.",
         "scientific instruments and readouts, close up, clinical lighting, documentary"),
        ("It came back the same. Then it came back the same again.",
         "a printout graph with an unexplained spike, close up, clinical light"),
        ("They checked every other explanation they could think of.",
         "researchers at a bench reviewing data, over the shoulder, documentary realism"),
        ("Every one of them failed, because {truth}.",
         "a single highlighted anomaly on a large screen in a dark control room, dramatic"),
        ("It is still listed as unexplained. The recordings are public.",
         "an archive shelf of labelled data tapes, cool light, documentary photography"),
    ),
}

STYLE_SUFFIX: dict[str, str] = {
    "sad": "cinematic film still, muted desaturated palette, soft natural light, 35mm, "
           "shallow depth of field, melancholic, no text, no watermark",
    "scifi": "cinematic sci-fi film still, volumetric light, cyan and amber palette, "
             "highly detailed, anamorphic, no text, no watermark",
    "horror": "cinematic horror film still, deep shadows, cold desaturated palette, "
              "35mm grain, unsettling, no text, no watermark",
    "mystery": "moody documentary photograph, overcast natural light, muted palette, "
               "film grain, no text, no watermark",
    "motivational": "cinematic film still, high contrast, warm rim light, gritty realism, "
                    "shallow depth of field, no text, no watermark",
    "wholesome": "warm cinematic film still, golden practical lighting, soft focus, "
                 "intimate, no text, no watermark",
    "revenge": "cinematic corporate thriller still, cool fluorescent light, clean lines, "
               "high detail, no text, no watermark",
    "facts": "scientific documentary photograph, natural light, high detail, realistic, "
             "no text, no watermark",
}

TONE_HINT: dict[str, str] = {
    "emotional": "emotional, intimate",
    "dark": "dark, ominous",
    "uplifting": "uplifting, hopeful",
    "suspenseful": "suspenseful, tense",
    "calm": "calm, contemplative",
    "energetic": "energetic, punchy",
}

HASHTAGS: dict[str, list[str]] = {
    "sad": ["#sadstory", "#emotional", "#storytime"],
    "scifi": ["#scifi", "#sciencefiction", "#storytime"],
    "horror": ["#horrorstories", "#scary", "#creepy"],
    "mystery": ["#mystery", "#unsolved", "#truecrime"],
    "motivational": ["#motivation", "#discipline", "#mindset"],
    "wholesome": ["#wholesome", "#kindness", "#faithinhumanity"],
    "revenge": ["#karma", "#satisfying", "#redditstories"],
    "facts": ["#facts", "#didyouknow", "#science"],
}

TITLES: dict[str, list[str]] = {
    "sad": ["He Never Knew Until It Was Too Late", "Nobody Noticed For Three Years",
            "The Letter He Never Opened"],
    "scifi": ["The Signal Was Coming From Inside", "Forty Years Of Silence",
              "They Woke Her Up Early"],
    "horror": ["There Were Ten Doors", "The Footage Was From Last Night",
               "She Should Not Have Watched It Back"],
    "mystery": ["Still Unexplained", "All Four Gave The Same Address",
                "The Road That Was Erased"],
    "motivational": ["You Are Not Behind", "The Boring Habit That Wins",
                     "Nobody Clapped. He Kept Going."],
    "wholesome": ["He Never Told Anyone", "Nine Years Later He Found Out",
                  "It Cost Four Pounds"],
    "revenge": ["So She Read The Policy", "He Wrote The Rule Himself",
                "She Kept Every Receipt"],
    "facts": ["They Checked It Four Times", "Still Listed As Unexplained",
              "The Sample Came Back Wrong"],
}


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def offline_script(
    topic: str,
    duration: int = 45,
    tone: str = "emotional",
    seed: int | None = None,
) -> ScriptResult:
    rng = random.Random(seed if seed is not None else random.randrange(1 << 30))
    genre = detect_genre(topic)
    pool = POOLS[genre]

    who, gender = rng.choice(pool["who"])
    names = [n for n, g in pool["name"] if g == gender] or [n for n, _ in pool["name"]]
    name = rng.choice(names)
    ritual_base, ritual_past, ritual_ing = rng.choice(pool["ritual"])

    slots = {
        **PRONOUNS[gender],
        "who": who,
        "name": name,
        "place": rng.choice(pool["place"]),
        "object": rng.choice(pool["object"]),
        "ritual_base": ritual_base,
        "ritual_past": ritual_past,
        "ritual_ing": ritual_ing,
        "truth": rng.choice(pool["truth"]),
    }

    def fill(template: str, depth: int = 2) -> str:
        out = template
        # Two passes: pools contain pronoun placeholders of their own, so a
        # single pass would leave "{he}" sitting inside a filled truth clause.
        for _ in range(depth):
            for key, value in slots.items():
                out = out.replace("{" + key + "}", value)
            out = re.sub(
                r"\{(\w+)_cap\}",
                lambda m: _cap(slots.get(m.group(1), m.group(0))),
                out,
            )
        out = re.sub(r"\s+", " ", out).strip()
        # Capitalise after sentence ends, in case a pronoun landed first.
        return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), out)

    hook = fill(rng.choice(HOOKS[genre]))

    # Piper speaks at roughly 2.6 words per second; a beat is about 7 seconds.
    beats = list(BEATS[genre])
    wanted = max(4, min(len(beats), round(duration / 7)))
    if wanted < len(beats):
        # Always keep the setup, the turn and the resolution; drop from the middle.
        middle = sorted(rng.sample(range(1, len(beats) - 2), wanted - 3))
        beats = [beats[0]] + [beats[i] for i in middle] + [beats[-2], beats[-1]]

    style = STYLE_SUFFIX[genre]
    tone_hint = TONE_HINT.get(tone.lower(), TONE_HINT["emotional"])
    scenes = [
        Scene(narration=fill(narration), visual_prompt=f"{fill(visual)}, {tone_hint}, {style}")
        for narration, visual in beats
    ]

    topic_clean = re.sub(r"\s+", " ", topic.strip()).strip(" .!?")
    title_pool = TITLES[genre] + [_cap(topic_clean), f"{_cap(topic_clean)} | A Short Story"]
    title = rng.choice(title_pool)[:95]

    return ScriptResult(
        title=title,
        hook=hook,
        scenes=scenes,
        cta=rng.choice(CTAS[genre]),
        description=f"{title}. {hook} Watch to the end.",
        hashtags=["#shorts", *HASHTAGS[genre]],
        provider="offline",
    )
