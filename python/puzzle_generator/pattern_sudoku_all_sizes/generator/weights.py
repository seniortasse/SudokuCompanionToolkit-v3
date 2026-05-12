
from generator.algo_human import TECHNIQUES


WEIGHTS = {
    "singles-1": 1,
    "singles-2": 10,
    "singles-naked-2": 20,
    "singles-3": 40,
    "singles-naked-3": 50,
    "doubles-naked": 100,
    "leftovers-1": 200,
    "triplets-naked": 300,
    "quads-naked": 400,
    "leftovers-2": 500,
    "singles-pointing": 500,
    "leftovers-3": 700,
    "singles-boxed": 800,
    "doubles": 900,
    "triplets": 1000,
    "quads": 1200,
    "x-wings": 1500,
    "leftovers-4": 1800,
    "y-wings": 2000,
    "leftovers-5": 2500,
    "remote-pairs": 3000,
    "leftovers-6": 3300,
    "boxed-doubles": 3500,
    "leftovers-7": 3800,
    "boxed-triplets": 4000,
    "leftovers-8": 4200,
    "boxed-wings": 4500,
    "boxed-rays": 5000,
    "ab-rings": 6000,
    "ab-chains": 7000,
    "x-wings-3": 8000,
    "boxed-quads": 9000,
    "x-wings-4": 10_000,
    "leftovers-9": 20_000,
}

assert not set(TECHNIQUES).symmetric_difference(WEIGHTS.keys())  # A weight should be defined for all techniques

assert list(WEIGHTS.values()) == sorted(WEIGHTS.values())  # List is ordered

# The order in which the techniques are applied is defined in TECHNIQUES, this should correspond with the weights!
assert [item[0] for item in sorted(WEIGHTS.items(), key=lambda item: item[1])] == TECHNIQUES


def determine_weight(counts_techniques):
    weight = sum(count * WEIGHTS[k] for k, count in counts_techniques.items())
    return weight
