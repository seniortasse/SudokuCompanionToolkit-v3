
from generator.algo_human import TECHNIQUES, BASE_TECHNIQUES
from generator.weights import WEIGHTS


# Default steering parameters for all grid sizes


DEFAULT_STEERING_PARAMETERS = {
    4: {
        1: {
            "non-empty-cells": (10, 12),
            "weight": (0, 6),
            "techniques-allowed": ["singles-1"],
        },
        2: {
            "non-empty-cells": (8, 10),
            "weight": (7, 10),
            "techniques-allowed": ["singles-1"],
        },
        3: {
            "non-empty-cells": (8, 12),
            "weight": (11, 50),
            "techniques-allowed": ["singles-1", "singles-2"],
        },
        4: {
            "non-empty-cells": (6, 10),
            "weight": (50, 1000),
            # All techniques with a weight lower than the max weight range
            "techniques-allowed": [name for name in TECHNIQUES if WEIGHTS.get(name, 0) < 1000],
        },
    },
    6: {
        1: {
            "non-empty-cells": (14, 20),
            "weight": (0, 75),
            "techniques-allowed": ["singles-1", "singles-2"],
        },
        2: {
            "non-empty-cells": (12, 18),
            "weight": (75, 125),
            # Only basic techniques
            "techniques-allowed": [name for name in TECHNIQUES if name in BASE_TECHNIQUES],
        },
        3: {
            "non-empty-cells": (12, 18),
            "weight": (125, 175),
            # Only basic techniques
            "techniques-allowed": [name for name in TECHNIQUES if name in BASE_TECHNIQUES],
        },
        4: {
            "non-empty-cells": (10, 16),
            "weight": (175, 10_000),
            # All techniques
            "techniques-allowed": TECHNIQUES,
        },
    },
    8: {
        1: {
            "non-empty-cells": (26, 32),
            "weight": (0, 250),
            # Only basic techniques
            "techniques-allowed": [name for name in TECHNIQUES if name in BASE_TECHNIQUES],
        },
        2: {
            "non-empty-cells": (22, 28),
            "weight": (250, 350),
            # Basic techniques plus a limited selection of advanced techniques
            "techniques-allowed": [name for name in TECHNIQUES if name in BASE_TECHNIQUES] + ["doubles", "triplets"],
        },
        3: {
            "non-empty-cells": (18, 24),
            "weight": (350, 500),
            # Basic techniques plus a limited selection of advanced techniques
            "techniques-allowed": [name for name in TECHNIQUES if name in BASE_TECHNIQUES] + ["doubles", "triplets", "quads"],
        },
        4: {
            "non-empty-cells": (14, 20),
            "weight": (500, 10_000),
            # All techniques
            "techniques-allowed": TECHNIQUES,
        },
    },
    9: {
        1: {
            "non-empty-cells": (38, 50),
            "weight": (0, 300),
            # Only basic techniques
            "techniques-allowed": ["singles-1", "singles-2"],
        },
        2: {
            "non-empty-cells": (30, 40),
            "weight": (300, 500),
            # Basic techniques plus a limited selection of advanced techniques
            "techniques-allowed": [name for name in TECHNIQUES if name in BASE_TECHNIQUES],
        },
        3: {
            "non-empty-cells": (24, 32),
            "weight": (500, 1000),
            # All techniques with a weight lower than the max weight range
            "techniques-allowed": [name for name in TECHNIQUES if name in BASE_TECHNIQUES] + ["doubles-naked", "triplets-naked", "quads-naked"]
        },
        4: {
            "non-empty-cells": (20, 26),
            "weight": (1000, float('inf')),
            # All techniques
            "techniques-allowed": TECHNIQUES,
        },
    },
    16: {
        1: {
            "non-empty-cells": (110, 125),
            "weight": (0, float('inf')),
            # Only basic techniques
            "techniques-allowed": [name for name in TECHNIQUES if name in BASE_TECHNIQUES],
        },
        2: {
            "non-empty-cells": (95, 109),
            "weight": (0, float('inf')),
            # Only basic techniques
            "techniques-allowed": [name for name in TECHNIQUES if name in BASE_TECHNIQUES],
        },
        3: {
            "non-empty-cells": (80, 94),
            "weight": (0, float('inf')),
            # All techniques with a difficulty lower than x-wings
            "techniques-allowed": [name for name in TECHNIQUES if WEIGHTS.get(name, 0) < 1000],
        },
        # Note: Rating 4 is not defined
    },
}

# Check that the technique names are properly defined
for _, params_for_size in DEFAULT_STEERING_PARAMETERS.items():
    for _, params in params_for_size.items():
        assert not set(params["techniques-allowed"]).difference(TECHNIQUES)
