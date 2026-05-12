
import itertools
import operator

from generator.techniques.options import determine_options_per_cell, copy_options
from generator.algo_human import apply_technique

from tool_logs.application import identify_relevant_applications, identify_earlier_applications


hits = 0


def identify_layers(step_instance_before, step_cleanup_steps, new_value, is_second_call=False):
    """
    Traces back which applications were required to find the final value (layer 1), which is the last applied technique
     but possible earlier applications as well, and the advanced techniques that were required for these applications
     to become available (layer 2)
    Each technique has its own logic for determining whether another technique was required earlier, and also which
     options had to be removed, which can be used to trace back earlier required applications
    """

    coords = new_value[0]
    char = new_value[1]
    dim_final_value = new_value[2].split(' & ')[0]
    details = new_value[3]

    # Identify layer 1
    # if dim_final_value == "cell":
    #     # Identify all applications which removed an option from the value cell
    #     pass

    # applications_layer_1 = []
    # for _, _, cleanup_step_details in enumerate(step_cleanup_steps):
    #     for name_application, application_details, removed_chars in cleanup_step_details:
    #         is_relevant = ...

    all_applications = list(itertools.chain.from_iterable(map(operator.itemgetter(-1), step_cleanup_steps)))

    # Pre-check: All applications should have a unique application name
    _names_applications = [application[0][1] for application in all_applications]
    assert len(_names_applications) == len(set(_names_applications)), "Some applications have a duplicate name!"

    # We can reuse the existing logic for determining relevant applications, by combining all potential applications
    applications_layer_1 = identify_relevant_applications(step_instance_before, coords, char, dim_final_value, all_applications)
    assert applications_layer_1, "Could not identify layer 1 applications!"

    print("Applications layer 1:")
    for idx, ((name_technique, name_application), _, removed_chars) in enumerate(applications_layer_1):
        _removed_chars = [(tuple(map(lambda x: x + 1, idx)), char) for idx, char in removed_chars]
        print("", idx + 1, name_application, "->", _removed_chars)

    # Check: The final application which found the new value should be included in the backtracked layer 1 applications
    #  -> This is invalid as during cleanup when the technique which found a new value does not need to be the latest one
    # assert step_technique_used in [name_technique for ((name_technique, _), _, _) in applications_layer_1]
    # Identify a cleanup issue
    # techniques_used = [name_technique for ((name_technique, _), _, _) in applications_layer_1]
    # if step_technique_used not in techniques_used:
    #     raise Exception(f"Identified cleanup issue! {step_technique_used} - {techniques_used}")

    # global hits
    # if len(applications_layer_1) > 1:
    #     hits += 1
    #     print("Number hits:", hits)
    #     import time
    #     time.sleep(2)
    #     if hits == 1:
    #         raise Exception("Investigate!")

    # Trace back what was the instance before filling in the new value
    # step_instance_before = copy_instance(step_instance)
    # step_instance_before[coords[0]][coords[1]] = EMPTY_CHAR
    options_natural = determine_options_per_cell(step_instance_before)

    # Identify layer 2 applications
    print("Applications layer 2:")
    applications_layer_2 = {}
    for application_layer_1 in applications_layer_1:
        name_application_layer_1 = application_layer_1[0][1]

        print(f"Identifying earlier applications needed for layer 1 application: {name_application_layer_1}")
        earlier_applications = identify_earlier_applications(
            application_layer_1, step_instance_before, options_natural, all_applications
        )
        # print(f"Number of options removed earlier to make it available: {len(earlier_applications)}")
        # Note: This is the number of applications as multiple removed options by the same application are grouped
        # print(f"Number of earlier applications needed: {len(earlier_applications)}")
        applications_layer_2[name_application_layer_1] = earlier_applications

        # TODO Rewrite to use recursion instead
        while len(earlier_applications) > 0:
            next_applications = []
            for application_layer_2 in earlier_applications:
                name_application_layer_2 = application_layer_2[0][1]

                print(f"Identifying earlier applications needed for layer 2 application: {name_application_layer_2}")
                _next_applications = identify_earlier_applications(
                    application_layer_2, step_instance_before, options_natural, all_applications
                )
                applications_layer_2[name_application_layer_2] = _next_applications
                next_applications.extend(_next_applications)
            earlier_applications = next_applications

    def print_recursive(application, level):
        ((_, name_application), _, removed_chars) = application
        _removed_chars = [(tuple(map(lambda x: x + 1, idx)), char) for idx, char in removed_chars]
        print('  ' * level, level, name_application, "->", _removed_chars)
        for application in applications_layer_2[name_application]:
            print_recursive(application, level + 1)

    print("Tree structure:")
    for application in applications_layer_1:
        print_recursive(application, 1)

    # if len(applications_layer_1) > 1 and any(len(v) > 1 for v in applications_layer_2.values()):
    #     print("Investigate!")
    #     quit()

    # TODO Post-process to a usable format both for formatting & messages

    # An extra postprocessing step:
    #  After identifying relevant applications, apply the algorithm once more and only accept an application if it has
    #  been identified as relevant -- This prevents removing irrelevant options, which are otherwise not explained in
    #  the formatting/messages and can lead to confusion

    # Steps:
    #  - Identify irrelevant techniques (all which are not identified as relevant)
    #  - Give this list to the algorithm, and do not remove options when an application is in this list

    applications_layer_2_flat = list(itertools.chain(*applications_layer_2.values()))
    unused_applications = [
        application
        for application in all_applications
        # CAREFUL! When identifying layer 1 & layer 2 applications the list of removed chars is updated according to
        #  which ones are relevant, so we cannot compare the full application
        if application[0] not in [_application[0] for _application in applications_layer_1 + applications_layer_2_flat]
    ]

    # To keep all logic related to re-applying in one place, we call the function another time with a flag - this is
    #  beautiful as this logic does not have to be added and copy/pasted to all places where this function is called,
    #  and all validation checks can be done here immediately; We can also disable this functionality easily
    # TODO This should be set to "True" when client wants to enable removing options only for relevant applications
    SHOULD_UPDATE_DETAILS = False
    if not is_second_call and SHOULD_UPDATE_DETAILS:

        relevant_applications = [
            application
            for application in all_applications
            # CAREFUL! When identifying layer 1 & layer 2 applications the list of removed chars is updated according to
            #  which ones are relevant, so we cannot compare the full application
            if application[0] in [_application[0] for _application in applications_layer_1 + applications_layer_2_flat]
        ]
        assert sum(map(len, [unused_applications, relevant_applications])) == len(all_applications)

        # Post-processing step: Reapply only relevant applications, preserving options originally removed by irrelevant
        #  applications
        updated_cleanup_steps = reapply_relevant_applications(
            step_instance_before, new_value, all_applications, relevant_applications
        )

        # Note: The instance before hasn't been updated, and when reapplying relevant applications we check that the
        #  final value found is the same as the original one
        updated_applications_layer_1, updated_applications_layer_2, updated_unused_applications = identify_layers(
            step_instance_before, updated_cleanup_steps, new_value, is_second_call=True
        )

        # Checks
        #  - The layer 1 & layer 2 applications are exactly the same
        #  - In the second call there are no unused applications anymore (as they have been filtered out from the
        #    details)

        # Check 1
        assert [application[0] for application in applications_layer_1] == \
               [application[0] for application in updated_applications_layer_1], f"{applications_layer_1} \n\n\n {updated_applications_layer_1}"
        assert {name_application: [application[0] for application in earlier_applications]
                for name_application, earlier_applications in applications_layer_2.items()} == \
               {name_application: [application[0] for application in earlier_applications]
                for name_application, earlier_applications in updated_applications_layer_2.items()}

        # Check 2
        assert len(updated_unused_applications) == 0

        # Update the values to return
        applications_layer_1 = updated_applications_layer_1
        applications_layer_2 = updated_applications_layer_2
        unused_applications = updated_unused_applications

    return applications_layer_1, applications_layer_2, unused_applications


def reapply_relevant_applications(step_instance_before, new_value, all_applications, relevant_applications):
    """
    Function specs

     As a user,
      - I want the function to produce correct results, ie updated cleanup logs with options removed by irrelevant
        applications still present

     As a developer,
      - I want to have a single function containing all the logic
      - I want to have a way to verify the code
      - I want to have a way to compare output visually
      - I want to have insights in the dynamics

    """

    # TODONE Check what is needed from the logs to generate formatting/messages, this should be updated here
    #  -> Only the cleanup_steps which are used to identify the layer 1 & 2 applications, so this should be updated;
    #     In the function identifying layers only the applications are taken from this, being the last element in the
    #     tuple, so we can just create a new list with single-element tuples

    # TODONE Structure to use:
    #  - Add indicator to identify_layers function, which is False the first call and gets called recursively inside
    #    the function at the end, with only the details of the relevant applications; All checks can be done inside the
    #    function!

    options_natural = determine_options_per_cell(step_instance_before)
    options_postprocessing = copy_options(options_natural)

    # Structure of "step_cleanup_steps":
    #  a list of tuples (technique_latest, technique_cleanup, counter_subprocess, options_before, details)

    # Used for validation check 2
    irrelevant_removed_options = []

    # Used to verify that validation check 3 was done
    is_check_3_done = False

    # Reconstruct the cleanup steps to re-identify relevant applications with the updated application details
    updated_cleanup_steps = []

    # for idx_step, (_, _, _, _, step_cleanup_step_details) in enumerate(step_cleanup_steps):
    #     for idx_it, application in enumerate(step_cleanup_step_details):
    #         (name_technique, name_application), application_details, removed_options = application
    for idx_step, application in enumerate(all_applications):
        (name_technique, name_application), application_details, removed_options = application
        idx_it = -1

        if application[0] in [_application[0] for _application in relevant_applications]:
            print(f"Re-applying application '{name_application}' in step {idx_step + 1} and iteration {idx_it} "
                  f"as it was identified to be relevant")
            _number_removed_options, updated_new_values, _options_not_accurate, updated_details = apply_technique(
                name_technique, options_postprocessing, step_instance_before, show_logs=False
            )

            # Validation checks:
            #  - Same application details
            #  - At least the same or some extra options removed
            #  - Same new value found if it is the last relevant application

            # Check 1
            assert len(updated_details) >= 1
            # Note: The application details do not need to be the same, in fact we are explicitly allowing for them
            #  not to be the same as we are possibly looking at different options
            assert name_application in [_name_application for (_, _name_application), _, _ in updated_details]

            # Check 2
            applications_reapplied = [_application for _application in updated_details if _application[0][1] == name_application]
            assert len(applications_reapplied) == 1
            application_reapplied = applications_reapplied[0]
            _removed_options = application_reapplied[-1]
            assert not set(_removed_options).difference(removed_options)
            additional_removed_options = set(removed_options).difference(_removed_options)
            assert not additional_removed_options.difference(irrelevant_removed_options)

            # Check 3
            # TODO This does not work - When the final value is found with some technique which has multiple
            #  applications in the same call, the last application is not the one finding the new value - Instead, we
            #  have to identify up front which application led to finding the final value
            # if idx_step == len(all_applications) - 1:
            if len(updated_new_values) > 0:
                # assert len(updated_new_values) >= 1
                # TODO This is dangerous as it depends on how we select the new value, which currently is just the first
                updated_new_value = updated_new_values[0]
                assert updated_new_value[:2] == new_value[:2], f"{updated_new_value[:2]} - {new_value[:2]}"
                # We should only find a new value once
                assert not is_check_3_done, "Found a new value more than once, which cannot be the case!"
                is_check_3_done = True

            print("Orginal removed options:", removed_options)
            print("Updated removed options:", _removed_options)

            # Only remove options removed by the relevant application
            #  (when taking the options returned by apply_technique, too many might have been removed as multiple
            #   applications can be found with a single call, not all of which need to be relevant)
            for (_i1, _i2), _char in _removed_options:
                options_postprocessing[_i1][_i2].remove(_char)

            # Re-generate details so that layer 1 & 2 applications can be re-identified
            # TODO How to do this neatly?
            # Note: From this list only the last tuple elements are taken, so we can just add a single-element tuple
            # Only keep the details of the relevant application reapplied - more applications might be contained in the
            #  updated details which are not necessarily relevant
            updated_cleanup_steps.append(([application_reapplied], ))

            # TODONE Validate the results after re-identifying the layer 1 & 2 applications, which should be exactly
            #  the same

        else:
            print(f"Do not re-apply application '{name_application}' in step {idx_step + 1} and iteration {idx_it} "
                  f"as it was identified to be irrelevant")
            irrelevant_removed_options.extend(removed_options)

    assert is_check_3_done, "Validation check 3 could not be done!"

    # TODONE One issue with the current implementation is that
    #  multiple applications can be found when applying a technique, and not all of them need to be relevant, so we
    #  might still be removing too many options
    #  -> only remove the options this time removed by re-applying the relevant application

    return updated_cleanup_steps
