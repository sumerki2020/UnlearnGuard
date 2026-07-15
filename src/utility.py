"""General-capability proxy: perplexity on a fixed battery of natural text.

Deliberately disjoint from the training filler: the filler is combinatorial
"<subject> <place> <verb> <time>" sentences, and Stage 2 showed held-out
sentences from that same pool collapse to ppl ~2 after fine-tuning — that
measures filler fit, not capability. This battery mixes registers (facts,
instructions, narrative, code-adjacent) so damage to general ability shows up.
"""

from src import metrics

NATURAL_TEXTS = [
    "The capital of France is Paris, a city on the river Seine.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "In 1969, astronauts first walked on the surface of the Moon.",
    "To make tea, boil water and let the leaves steep for three minutes.",
    "Press and hold the power button for ten seconds to restart the device.",
    "Whisk the eggs with a pinch of salt before adding them to the pan.",
    "Remember to save your work before closing the application.",
    "She packed her bag quickly and caught the last train home.",
    "The old dog slept on the porch while rain drummed on the roof.",
    "After the meeting, they agreed to postpone the launch by a week.",
    "He read the letter twice, then folded it and said nothing.",
    "A function that calls itself is known as a recursive function.",
    "The database query returned more rows than the report expected.",
    "Interest rates influence how much it costs to borrow money.",
    "The museum's new exhibit features paintings from the early Renaissance.",
    "Regular exercise and adequate sleep both improve concentration.",
    "The recipe serves four people and takes about forty minutes.",
    "Lightning is an electrical discharge between clouds and the ground.",
    "The committee published its findings in a two-hundred-page report.",
    "Migration patterns of arctic birds shift as winters become milder.",
    "The bridge was closed for repairs, so traffic crossed at the ferry.",
    "Simple interest is calculated only on the original principal.",
    "The choir rehearsed the final movement until midnight.",
]


def natural_ppl(model, tokenizer):
    return metrics.perplexity(model, tokenizer, NATURAL_TEXTS)
