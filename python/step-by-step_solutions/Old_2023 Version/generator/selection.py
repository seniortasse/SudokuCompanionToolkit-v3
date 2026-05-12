
from generator.model import EMPTY_CHAR, copy_instance
from generator.algo_human import solve_using_human_techniques, TECHNIQUES


WEIGHTS_STEERING = {
    "singles-1": 1.0,
    "singles-2": 1.1,
    "singles-naked-2": 1.2,
    "singles-3": 1.3,
    "singles-naked-3": 1.4,
    "doubles": 2.0,
    "triplets": 2.1,
    "quads": 2.2,
    "singles-pointing": 3.0,
    "x-wings": 4.0,
    "remote-pairs": 4.0,
    "magic": 100.0,
}


def weight_fnc(techniques):
    total_weight = 0
    for name, count in techniques.items():
        weight = count * WEIGHTS_STEERING[name]
        total_weight += weight
    # total_weight = sum(techniques.get(name, 0) * WEIGHTS_STEERING[name] for name in TECHNIQUES)
    return total_weight


def select_next_idx(instance, idxs, solution,
                    allowed_techniques, max_uses_techniques, only_accept_human_solvable_instances, show_logs):

    collection_techniques_counts = {}
    for idx in idxs:

        # TODO Speed up is possible by first ruling out that the idx is the only occurrence (which got removed before,
        #  before calling this function, but is now included in the can_remove_idx() function)

        _instance = copy_instance(instance)
        _instance[idx[0]][idx[1]] = EMPTY_CHAR
        _solved_instance, (_techniques, _) = solve_using_human_techniques(
            _instance,
            use_techniques=allowed_techniques, include_magic_technique=not only_accept_human_solvable_instances, magic_solution=solution,
            show_logs=show_logs,
        )

        is_humanly_solvable = _solved_instance == solution
        within_max_uses_techniques = all(
            name not in max_uses_techniques or _techniques.get(name, 0) <= max_uses_techniques[name]
            for name in TECHNIQUES
        )

        if is_humanly_solvable and within_max_uses_techniques:
            # print(f"Removing {idx} solvable with techniques and within allowed range -> {is_humanly_solvable} ({_techniques})")
            collection_techniques_counts[idx] = _techniques
        # else:
        #     # This cannot be done while looping over idxs
        #     idxs.remove(idx)

        if len(collection_techniques_counts) >= 3:
            break

    # print("Number un/solvable:", len(idxs) - len(collection_techniques_counts), '/', len(collection_techniques_counts))

    # TODO Efficiency improvement: Continue from a bit higher up in the branch (tree search), so that we do not
    #  have to start the next search from scratch
    if len(collection_techniques_counts) == 0:
        print("Aborted search as none of the next instances are humanly solvable")
        idx = None
    else:

        # TODO Make sure that a positive weight is given for all solvable instances, as otherwise non-solvable instances
        #  can be chosen here (which is what happened when all instances had only singles and were not valued)
        weights_for_idxs = {idx: weight_fnc(collection_techniques_counts.get(idx, {})) for idx in idxs}
        _idxs_sorted = sorted(idxs, key=lambda idx: weights_for_idxs[idx], reverse=True)
        idx = _idxs_sorted[0]

        # print(f"Continue with removing {idx} as weight is highest:", collection_techniques_counts.get(idx), weights_for_idxs[idx])

    # Remove idxs that dit not lead to a humanly solvable solution within the max number of uses for the techniques
    # (before this was done in the loop, but this cannot be done)
    # TODO This should not be modified from within the function
    idxs_to_be_removed = [idx for idx in idxs if idx not in collection_techniques_counts.keys()]
    for idx in idxs_to_be_removed:
        idxs.remove(idx)

    return idx
