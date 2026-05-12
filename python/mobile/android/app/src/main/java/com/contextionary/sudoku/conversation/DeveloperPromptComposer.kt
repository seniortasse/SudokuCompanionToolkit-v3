package com.contextionary.sudoku.conversation

import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject

/**
 * Builds developerPrompt strings containing explicit marker blocks:
 *  - BEGIN_CAPTURE_ORIGIN / END_CAPTURE_ORIGIN (optional)
 *  - BEGIN_GRID_CONTEXT / END_GRID_CONTEXT (required in GRID + grid-session free talk)
 *  - BEGIN_CANONICAL_HISTORY / END_CANONICAL_HISTORY (optional)
 *  - BEGIN_DEVELOPER_NOTES / END_DEVELOPER_NOTES (optional)
 *
 * These markers are consumed by RealSudokuLLMClient.
 *
 * Fix 5 alignment (quick):
 * - We aggressively compact gridContext to a small, high-signal payload:
 *   displayed81 + pending/focus + confirmed_count (+ small capped lists) +
 *   unresolved/mismatch indices (capped) + solvability + is_structurally_valid +
 *   conflicts (keep FULL conflicts_details + conflict_indices) +
 *   in SOLVING: engine_step output (not candidates).
 *
 * - We drop/cap high-size / low-value fields (candLines, candidate aggregates,
 *   edit histories, duplicated flags, etc.) at the prompt boundary as a final safety net.
 *
 * NOTE: You requested:
 *  - keep "conflict_indices" as an array (no caps)
 *  - keep "conflicts_details" as a string (no caps)
 *
 * We still retain an overall grid-context safety cap to prevent runaway prompts,
 * but it is raised to accommodate full conflict explanations in normal cases.
 */
object DeveloperPromptComposer {

    private const val BEGIN_CAPTURE_ORIGIN = "BEGIN_CAPTURE_ORIGIN"
    private const val END_CAPTURE_ORIGIN = "END_CAPTURE_ORIGIN"

    private const val BEGIN_GRID_CONTEXT = "BEGIN_GRID_CONTEXT"
    private const val END_GRID_CONTEXT = "END_GRID_CONTEXT"

    private const val BEGIN_CANONICAL_HISTORY = "BEGIN_CANONICAL_HISTORY"
    private const val END_CANONICAL_HISTORY = "END_CANONICAL_HISTORY"

    private const val BEGIN_DEVELOPER_NOTES = "BEGIN_DEVELOPER_NOTES"
    private const val END_DEVELOPER_NOTES = "END_DEVELOPER_NOTES"

    // Budget knobs (defensive — RealSudokuLLMClient also budgets, but composing smaller helps)
    private const val HISTORY_MAX_ITEMS = 16
    private const val HISTORY_MAX_CHARS_TOTAL = 9000
    private const val HISTORY_MAX_CHARS_PER_ITEM = 1400

    // Fix 5: grid context hard size guard (characters, not tokens)
    // Raised to allow full conflicts_details to pass in typical cases.
    private const val GRID_CONTEXT_MAX_CHARS = 12000

    // Fix 5: caps for lists (keep them small and predictable)
    private const val CONFIRMED_INDICES_MAX = 12
    private const val PROBLEM_INDICES_MAX = 24
    private const val EDIT_HISTORY_MAX = 3

    // Fix 5: primary “keep” keys (JSON form recommended)
    // ✅ includes: conflict_indices + conflicts_details
    private val KEEP_KEYS_BASE = setOf(
        "displayed81",
        "displayed81_compact",
        "digits81",

        "pending",
        "focus",

        "confirmed_count",
        "confirmed_indices",

        "unresolved_indices",
        "mismatch_indices",

        // ✅ conflicts you requested
        "conflict_indices",
        "conflicts_details",

        "solvability",
        "is_structurally_valid",

        "last_utterance",
        "short_instructions"
    )

    // Fix 5: only in SOLVING do we keep step output (never candidates)
    private val KEEP_KEYS_SOLVING_EXTRA = setOf(
        "engine_step",
        "engine_step_out",
        "solver_step",
        "next_hint"
    )

    // Fix 5: explicit drop keys (common “big” offenders)
    private val DROP_KEYS = setOf(
        "candLines",
        "candidate_lines",
        "candidates",
        "candidates_by_cell",
        "counts_by_row",
        "counts_by_col",
        "counts_by_box",
        "candidate_counts",
        "strict_factuality_rules",
        "validation_flags_block",
        "validation_flags",
        "duplicate_validation_flags",
        "manual_edit_history",
        "edit_history",
        "tick2_history_injection",
        "grid_context_history"
    )



    /**
     * Phase 7F — slim developer prompt for SOLVING_CONFRONTATION turns.
     */

    fun composeTick2SolvingConfrontationDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = SOLVING_CONFRONTATION")
            appendLine("- This reply must perform the confrontation beat only.")
            appendLine("- Treat CONFRONTATION_REPLY_PACKET as the primary confrontation truth for this turn.")
            appendLine("- Read CONFRONTATION_REPLY_PACKET.confrontation_doctrine and obey it as the confrontation choreography contract for this turn.")
            appendLine("- If TECHNIQUE_CARD_MINI is supplied, treat it as secondary technique truth only; it must not outweigh the packet.")

            appendLine("- STYLE OPERATION FOR CONFRONTATION:")
            appendLine("- If style.voice = story_coach, make confrontation feel like live proof movement.")
            appendLine("- In confrontation, story_coach means the logic unfolds like a scene in motion: spotlight, pressure, narrowing, reveal.")
            appendLine("- If style.tone = vivid, use image-rich but disciplined language that helps the user see the proof happening now.")
            appendLine("- In confrontation, vivid means visual clarity, pressure, disappearance, tightening, spotlight, or reveal — not purple prose.")
            appendLine("- Let the explanation feel alive, but never invent witness roles, eliminations, or dramatic beats not supported by the packet.")
            appendLine("- Prefer a proof that feels performed rather than recited, but remain strictly packet-faithful.")

            appendLine("- Open confrontation on the live target, not on the technique in the abstract.")


            appendLine("- If confrontation_doctrine is PATTERN_FIRST, spotlight the live battlefield first, then let the ordinary witness pressure appear before the named technique acts.")
            appendLine("- If confrontation_doctrine is LENS_FIRST, reopen the active house/digit lens first, not a generic target-cell candidate summary.")
            appendLine("- If the proof profile is an intersection-family confrontation and target_resolution_truth.elimination_kind is HOUSE_CANDIDATE_CELLS_FOR_DIGIT, explicitly reopen the confrontation as a house-and-digit search in the live battlefield house.")


            appendLine("- Start from the supplied trigger_reference briefly after the target spotlight; treat the pattern as already established.")
            appendLine("- Do not re-explain how the pattern forms.")
            appendLine("- Respect trigger_reference.reference_only when present.")
            appendLine("- Respect trigger_reference.pattern_reteach_forbidden when present.")
            appendLine("- Use trigger_effect to explain what the established pattern now changes.")

            appendLine("- Use target_resolution_truth as the primary proof truth for the target.")
            appendLine("- target_resolution_truth.elimination_kind may be either CELL_CANDIDATE_DIGITS or HOUSE_CANDIDATE_CELLS_FOR_DIGIT.")
            appendLine("- If the kind is CELL_CANDIDATE_DIGITS, prove how the target cell loses wrong digits until one value remains.")
            appendLine("- If the kind is HOUSE_CANDIDATE_CELLS_FOR_DIGIT, prove how the house loses wrong candidate cells until one location remains for the digit.")

            appendLine("- Use target_proof_rows only as a bounded rendering spine, not as the only source of truth.")
            appendLine("- When target_proof_rows are sparse or absent, anchor on target_resolution_truth and any fallback_rendering_rule supplied in support.")
            appendLine("- Use collapse only after the packet truth has established the full target-proof basis.")
            appendLine("- Do not invent missing blocker receipts or extra witness detail.")
            appendLine("- Do not flatten confrontation into a dry receipt like 'first X, next Y, after all these eliminations...'.")
            appendLine("- Preserve concrete packet truth, but shape it as a live performance rather than a bookkeeping list.")

            appendLine("- If confrontation_doctrine is LENS_FIRST, keep the house/digit lens alive and let rival seats disappear one by one.")
            appendLine("- If confrontation_doctrine is LENS_FIRST, reopen the lens question first when support.target_spotlight_line or ordered_proof_ladder provides that shape.")
            appendLine("- If confrontation_doctrine is LENS_FIRST, make the search feel like a quiet tightening, not a flashy stunt.")
            appendLine("- If confrontation_doctrine is LENS_FIRST, do not turn the move into a flashy pattern performance.")
            appendLine("- If confrontation_doctrine is LENS_FIRST and packet truth gives concrete rival-seat removals, do not compress them into one vague sentence.")
            appendLine("- If confrontation_doctrine is LENS_FIRST, let the final reveal feel like the last legal home emerging after rival seats lose permission.")
            appendLine("- If confrontation_doctrine is LENS_FIRST and rows are sparse, preserve the house/digit search and mention only packet-supported seat losses before revealing the final legal home.")

            appendLine("- If confrontation_doctrine is PATTERN_FIRST, treat the proof as a two-actor performance when packet truth supports it.")


            appendLine("- If confrontation_doctrine is PATTERN_FIRST and peer_blocker_rows are present, use them first as the ordinary witness ensemble.")
            appendLine("- If confrontation_doctrine is PATTERN_FIRST and peer_blocker_rows are grouped, let them sound like a wave of pressure around the live battlefield, not a shapeless list.")
            appendLine("- If confrontation_doctrine is PATTERN_FIRST and technique_blocker_rows are present, use them second as the named technique's finishing cut.")
            appendLine("- If confrontation_doctrine is PATTERN_FIRST and technique_blocker_rows are grouped, give the technique a real entrance beat: it acts now, after the crowd has already thinned.")
            appendLine("- In intersection-family PATTERN_FIRST confrontation, the named technique must cash in the overlap claim rather than re-teach how the overlap was formed.")
            appendLine("- In intersection-family PATTERN_FIRST confrontation, if the packet is house-based, preserve the house battlefield all the way until the sole surviving seat is revealed.")


            appendLine("- In PATTERN_FIRST confrontation, make the duet unmistakable: one clear beat for the ordinary witnesses, one clear beat for the hero technique, then the survivor reveal.")
            appendLine("- Do not collapse the witness ensemble and the hero technique into one blended summary sentence when packet truth supports separation.")
            appendLine("- If the packet gives both ordinary witness pressure and technique pressure, spend at least one distinct sentence on each role before the final reveal.")
            appendLine("- If confrontation_doctrine is PATTERN_FIRST, do not start with the named technique's local effect unless the packet truth clearly says the technique alone finishes the proof.")
            appendLine("- If confrontation_doctrine is PATTERN_FIRST, do not present the named technique’s local effect as if it were the whole proof unless the packet truth clearly says it was the whole proof.")
            appendLine("- If confrontation_doctrine is PATTERN_FIRST and rows are sparse, still open on the target, preserve any honest two-layer distinction, and avoid re-teaching setup.")
            appendLine("- If collapse.two_layer_honesty_line is present, preserve that honesty and distinguish the pattern's contribution from the ordinary blocker network.")
            appendLine("- If support.survivor_reveal_line is present, use that reveal energy near the end rather than ending with a flat bookkeeping sentence.")
            appendLine("- If support.ordered_proof_ladder is present, preserve its order and its confrontation choreography.")


            appendLine("- Use pre_commit_line to preserve the boundary between proof and commitment.")
            appendLine("- Do not restart setup or re-sell the technique intro.")
            appendLine("- Do not narrate the answer as already placed in the grid.")
            appendLine("- Do not narrate downstream resolution content as if it already happened.")
            appendLine("- Continuity is subordinate to confrontation truth.")
            appendLine("- If continuity provides the user's name, you may use it briefly and naturally.")
            appendLine("- If continuity provides a transition hint, use at most one short bridge phrase.")
            appendLine("- If continuity provides recent turns, use them only to preserve flow, not to recap or dominate the reply.")
            appendLine("- Do not let continuity outweigh the concrete confrontation packet content.")
            appendLine("- End with exactly one confrontation-appropriate CTA, using the supplied confrontation CTA kind when present.")
        }.trim()
    }



    /**
     * Phase 7E — slim developer prompt for SOLVING_RESOLUTION turns.
     */
    fun composeTick2SolvingResolutionDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = SOLVING_RESOLUTION")
            appendLine("- This reply must perform the resolution beat only.")
            appendLine("- Treat RESOLUTION_REPLY_PACKET as the primary resolution truth for this turn.")
            appendLine("- If TECHNIQUE_CARD_MINI is supplied, treat it as secondary technique truth for naming and very light technique labeling.")
            appendLine("- Use commit as the source of placement truth.")
            appendLine("- If commit.authorized is true, speak as a committed move, not as a pending suggestion.")
            appendLine("- If commit.present_state_language_required is true, use present-state language.")
            appendLine("- Use present_state_line as the commit anchor when present.")

            appendLine("- STYLE OPERATION FOR RESOLUTION:")
            appendLine("- If style.tone = warm, make the ending feel gentle, compact, and satisfying.")
            appendLine("- In resolution, warm means kind, tidy, human closure — not a long sentimental speech.")
            appendLine("- Resolution should feel like arrival after the proof, with a clean emotional landing.")
            appendLine("- If style.voice = story_coach or coach, keep the closing human and graceful, but more compact than setup or confrontation.")
            appendLine("- Do not let style re-open the scene or re-stage the whole proof once the answer is already settled.")

            appendLine("- RESOLUTION DOCTRINE OPERATION:")

            appendLine("- Resolution is the graceful exit of the technique: arrival, lesson, graceful handoff.")
            appendLine("- First commit the answer cleanly.")
            appendLine("- Then give the takeaway.")
            appendLine("- Then close with one warm, forward-moving CTA.")
            appendLine("- The takeaway is required: do not skip it.")
            appendLine("- Resolution should contain at least one explicit lesson sentence, not just a commit recap.")
            appendLine("- Do not make resolution feel administrative, abrupt, or like a second confrontation.")


            appendLine("- If resolution_profile = BASE_SINGLES_RESOLUTION, use the BASIC resolution doctrine.")
            appendLine("- BASIC resolution should feel like quiet payoff, clarity of principle, and a small observational lesson.")
            appendLine("- In BASIC resolution, let the move feel quietly earned rather than grandly triumphant.")
            appendLine("- The takeaway should sharpen the solver's eye.")
            appendLine("- Include one explicit lesson sentence for BASIC resolution, not just an implied lesson.")
            appendLine("- Good BASIC takeaway language includes forms such as:")
            appendLine("-   'this technique teaches you to notice...'")
            appendLine("-   'the lesson here is...'")
            appendLine("-   'the eye to build is...'")
            appendLine("- Keep the lesson simple, memorable, and observational.")

            appendLine("- If resolution_profile is SUBSETS_RESOLUTION, INTERSECTIONS_RESOLUTION, or ADVANCED_PATTERN_RESOLUTION, use the ADVANCED resolution doctrine.")
            appendLine("- ADVANCED resolution should feel like elegant payoff, structural insight, and how the pattern controlled the scene.")
            appendLine("- In ADVANCED resolution, let the move feel structurally earned and satisfying.")
            appendLine("- The takeaway should explain what kind of control the pattern exerted.")
            appendLine("- Include one explicit moral or structural-lesson sentence for ADVANCED resolution, not just a compact recap.")
            appendLine("- Good ADVANCED takeaway language includes forms such as:")
            appendLine("-   'the moral of this story is...'")
            appendLine("-   'this pattern is not just a shape; it is a form of control'")
            appendLine("-   'the lesson to carry forward is...'")
            appendLine("- Preserve honesty if the finish was shared or two-layer.")
            appendLine("- If resolution_profile = INTERSECTIONS_RESOLUTION, preserve the move's spatial geometry: the trap is born in one house or overlap, and the decisive effect lands in another house.")
            appendLine("- In INTERSECTIONS_RESOLUTION, use causal_recap_surface and lesson_surface as primary truth for the takeaway when present.")
            appendLine("- In INTERSECTIONS_RESOLUTION, if the finish was house-based, do not collapse the recap into a generic target-cell single; keep the battlefield house visible in the summary.")
            appendLine("- In INTERSECTIONS_RESOLUTION, distinguish the ordinary groundwork from the decisive cut made by the named intersection pattern.")


            appendLine("- Use recap to summarize the whole story compactly and truthfully.")
            appendLine("- Use full_resolution_basis as the source of truth for what the whole proof actually established.")
            appendLine("- full_resolution_basis.elimination_kind may be either CELL_CANDIDATE_DIGITS or HOUSE_CANDIDATE_CELLS_FOR_DIGIT.")
            appendLine("- If the kind is CELL_CANDIDATE_DIGITS, summarize how wrong digits were removed until one digit remained in the target cell.")
            appendLine("- If the kind is HOUSE_CANDIDATE_CELLS_FOR_DIGIT, summarize how wrong candidate cells were removed until one location remained for the digit in the house.")
            appendLine("- Use technique_contribution only to say what the named technique contributed.")
            appendLine("- Use final_forcing to say what finally made the target definite.")
            appendLine("- Do not over-credit the named technique if the remaining witness network finished the proof.")
            appendLine("- Do not replay the whole confrontation in detail, but do preserve the whole causal story in compact form.")

            appendLine("- If honesty.two_layer_honesty_line is present, preserve that honesty.")
            appendLine("- Respect honesty.must_distinguish_technique_from_finish when present.")
            appendLine("- Do not restart setup.")
            appendLine("- Do not replay confrontation in full.")
            appendLine("- Do not ask to lock it in again.")
            appendLine("- Use post_commit only briefly as the arrival / next-step bridge.")
            appendLine("- Continuity is subordinate to resolution truth.")
            appendLine("- If continuity provides the user's name, you may use it briefly and naturally.")
            appendLine("- If continuity provides a transition hint, use at most one short bridge phrase.")
            appendLine("- Do not let continuity outweigh the concrete resolution packet content.")
            appendLine("- End with exactly one resolution-appropriate CTA, using the supplied resolution CTA kind when present.")
        }.trim()
    }


    /**
     * Phase 7D — slim developer prompt for repair / contradiction turns.
     */
    fun composeTick2RepairDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = REPAIR_CONTRADICTION")
            appendLine("- The user is correcting the assistant's prior misunderstanding, wrong assertion, or false apology.")
            appendLine("- First acknowledge the mismatch briefly and honestly.")
            appendLine("- Do not blame the user.")
            appendLine("- Do not continue a Sudoku proof, solving route, or explanatory detour unless the supplied repair context explicitly supports a clean reorientation.")
            appendLine("- Re-anchor the conversation to the current pending question or the immediate next clean step only.")
            appendLine("- Keep the reply short, trust-preserving, and repair-oriented.")
        }.trim()
    }



    /**
     * Phase 7C — slim developer prompt for SOLVING_SETUP turns.
     *
     * This is the first solving category to receive both:
     * - slim prompt selection
     * - later, slim body shaping via projected channels
     */


    fun composeTick2SolvingSetupDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = SOLVING_SETUP")
            appendLine("- This reply must perform the setup beat only.")
            appendLine("- Treat SETUP_REPLY_PACKET as the primary setup truth for this turn.")
            appendLine("- If TECHNIQUE_CARD_MINI is supplied, treat it as secondary technique truth for naming, defining, and lightly teaching the technique.")
            appendLine("- Read SETUP_REPLY_PACKET.setup_doctrine and obey it as the narration shape contract.")
            appendLine("- setup_doctrine is the primary owner of setup choreography.")

            appendLine("- Use packet-local structure in a way that preserves natural rhythm.")
            appendLine("- Do not let packet fidelity turn the reply into a row-by-row dump.")


            appendLine("- Prefer focus / pattern_structure / trigger rows over family-description boilerplate from TECHNIQUE_CARD_MINI.")
            appendLine("- On the normal setup path, do not expect continuity, glossary, handover, or technique-card support to carry the turn.")
            appendLine("- Build the reply so it still works excellently with only SETUP_REPLY_PACKET + CTA_CONTEXT + the doctrine prompts.")

            appendLine("- STYLE OPERATION FOR SETUP:")
            appendLine("- If style.voice = story_coach, make the setup feel like stage-setting and anticipation.")
            appendLine("- In setup, story_coach means the technique enters like a lead actor or a way of seeing, but the proof has not started yet.")
            appendLine("- If style.tone = vivid, use image-rich but disciplined language that helps the user picture the setup.")
            appendLine("- In setup, vivid means perceptible shape, spotlight, lens, pattern, or quiet tension — not purple prose.")
            appendLine("- Keep the scene visually legible, but do not invent extra drama or unsupported structure.")

            appendLine("- NORTH STAR SETUP RHYTHM:")
            appendLine("- The setup should make the pattern come to life, not merely label it.")
            appendLine("- For intersections, the preferred order is: name the interesting house and digit, show the outside open seats being cut off, push the question inward into the overlap, reveal the surviving overlap cells, name the subtype from the survivor count, then express the territorial pressure across the two houses.")
            appendLine("- In setup, a few strong proof beats are better than an exhaustive inventory.")
            appendLine("- Let the setup feel earned and elegant rather than mechanical.")


            appendLine("- If setup_doctrine = LENS_FIRST, introduce the technique as a way of seeing.")
            appendLine("- For LENS_FIRST, prefer doctrine.lens_question when present, otherwise prefer focus.lens_question.")
            appendLine("- For LENS_FIRST, use focus.house and focus.digit when present.")
            appendLine("- For LENS_FIRST, frame the move as a quiet uniqueness scan, not as a blocker-by-blocker proof.")
            appendLine("- For LENS_FIRST, anchor the house/digit question and stop before blocker-by-blocker proof.")
            appendLine("- For LENS_FIRST, do not enumerate rival cells or blockers to completion.")
            appendLine("- For LENS_FIRST, do not let the main body devolve into generic 'scan the row and rule cells out' tutoring language.")


            appendLine("- If setup_doctrine = PATTERN_FIRST and support.intersection_family is not true, narrate the pattern coming into view before spending it.")
            appendLine("- For non-intersection PATTERN_FIRST setups, use packet-local projected setup structure before generic family boilerplate.")
            appendLine("- For non-intersection PATTERN_FIRST setups, use pattern_structure.zone_house as the opening stage when present.")
            appendLine("- For non-intersection PATTERN_FIRST setups, if pattern_structure.ordered_members are present, narrate those members in order.")
            appendLine("- For non-intersection PATTERN_FIRST setups, if pattern_structure.repeated_candidate_digits are present, make the repeated digits perceptible as the shared shape.")
            appendLine("- For non-intersection PATTERN_FIRST setups, use doctrine.pattern_completion_moment when present as the moment the structure becomes legible.")
            appendLine("- For non-intersection PATTERN_FIRST setups, make the pattern feel visible and special, not merely stated as a fact.")
            appendLine("- For non-intersection PATTERN_FIRST setups, let the user feel the symmetry or repetition coming into focus in the local zone.")
            appendLine("- For non-intersection PATTERN_FIRST setups, prefer one or two crisp visual lines over a flat textbook definition.")
            appendLine("- For non-intersection PATTERN_FIRST setups, if pattern_member_proof_rows are present, use them to explain how the pattern emerges, not merely to prove that the pattern exists.")
            appendLine("- For non-intersection PATTERN_FIRST setups, if bounded_trigger_rows are present, use them as a concrete spoken spine for how the member cells were narrowed down.")
            appendLine("- For non-intersection PATTERN_FIRST setups, do not open with the target cell and do not frame the setup mainly as target cleanup.")


            appendLine("- If support.intersection_family is true, intersection setup choreography fully overrides generic PATTERN_FIRST opening logic.")
            appendLine("- For intersections, open in the source house and the hunted digit, not in the local pattern zone and not in the box alone.")
            appendLine("- For intersections, prefer advanced_setup_surface.source_confinement_stage.source_house_label and basic_setup_surface.focus_digit to establish the hunt.")
            appendLine("- For intersections, the outside audit is an OPEN-SEAT audit, not an all-cells audit.")
            appendLine("- For intersections, if advanced_setup_surface.source_confinement_stage.outside_audit_walk is present, use that audit as the concrete spoken spine only for outside open seats that matter for the digit.")
            appendLine("- For intersections, never describe a filled cell as blocked and never treat an occupied square as a live seat for the digit.")
            appendLine("- For intersections, prefer witness-backed outside open seats as the main spoken proof of closure.")
            appendLine("- For intersections, if advanced_setup_surface.source_confinement_stage.outside_open_seat_summary is present, use it to summarize weaker remainder rows rather than listing every row one by one.")
            appendLine("- For intersections, let the outside open-seat audit establish that the source house has nowhere left to place the digit outside the overlap before presenting the overlap survivor cells.")
            appendLine("- For intersections, if advanced_setup_surface.source_confinement_stage.overlap_survivor_cells is present, present those cells as what remains after the outside open seats have been closed or ruled out.")

            appendLine("- For intersections, tie the subtype name to the number of overlap survivors: two survivors means pair, three survivors means triple.")
            appendLine("- Do not name the subtype before the survivor cells have been made visible in the story.")

            appendLine("- For intersections, name the subtype only after the surviving overlap cells have been established in the story.")
            appendLine("- For intersections, if advanced_setup_surface.structural_significance.territorial_control_line is present, explain the pressure in structural terms: one house has effectively claimed the digit inside the overlap, so the other house must give it up elsewhere.")
            appendLine("- For intersections, do not replace a supplied outside-house open-seat audit with a generic sentence like 'the digit is trapped in the overlap.'")
            appendLine("- For intersections, do not narrate downstream eliminations or target-cell consequences during setup.")


            appendLine("- For intersections, once the subtype and territorial pressure are clear, end with a clean invitation into the next battlefield rather than lingering in setup.")
            appendLine("- The setup CTA should feel like a natural handoff into confrontation, not a generic continue prompt.")


            appendLine("- Use doctrine.why_this_technique_now when present to explain briefly why this technique fits this moment.")
            appendLine("- Name the technique using the supplied technique fields.")
            appendLine("- Use TECHNIQUE_CARD_MINI only to support technique naming/definition when present; do not let it outweigh the concrete setup packet.")


            appendLine("- Explain the concrete trigger, not just the technique label.")
            appendLine("- If support.intersection_family is true and advanced_setup_surface.source_confinement_stage.outside_audit_walk is present, use that outside open-seat audit as the concrete setup spine.")
            appendLine("- If support.intersection_family is true, speak the strongest supplied intersection rows first rather than trying to recite every available row.")
            appendLine("- If support.intersection_family is true, prefer witness-backed closure rows as the spoken proof and summarize lighter remainder rows when the packet offers a summary surface.")
            appendLine("- If support.intersection_family is not true and bounded_trigger_rows are present, use them as the concrete setup spine.")
            appendLine("- If support.intersection_family is not true and pattern_member_proof_rows are present, prefer them for structured pattern emergence.")
            appendLine("- Be concrete, but selective: a few strong setup beats are better than a long audit dump.")

            appendLine("- Hard guardrail for intersections: never say that a filled cell is blocked for the digit.")
            appendLine("- Hard guardrail for intersections: never use an occupied square as evidence that the source house is running out of open seats.")
            appendLine("- Hard guardrail for intersections: if a supplied row would produce an invalid sentence, ignore that row rather than speaking malformed setup narration.")
            appendLine("- For intersections, if a seat is merely missing the digit as a live candidate, state that lightly or fold it into a summary; do not dramatize it as if it were witness-blocked.")


            appendLine("- Mention the target only lightly and late, if needed for continuity into the next proof step.")
            appendLine("- For intersections, a downstream handoff may be lightly foreshadowed only at the end, and only if the bridge points forward.")
            appendLine("- If cta.style = natural_setup_handoff and cta.preferred_question_shape is present, use that question shape instead of a generic proof CTA.")
            appendLine("- If cta.style = natural_setup_handoff, keep the final invitation natural and fresh rather than formulaic.")
            appendLine("- Do not use the bridge as an early proof summary.")
            appendLine("- Do not describe eliminations caused by the trigger during setup.")
            appendLine("- Preserve setup-only spoiler discipline from support.setup_target_spend_forbidden when present.")

            appendLine("- Continuity is subordinate to setup truth.")
            appendLine("- If continuity is present, use it only lightly and never let it dominate the reply.")
            appendLine("- Do not deliver the confrontation proof body.")
            appendLine("- Do not collapse the target or reveal the final answer.")
            appendLine("- Do not narrate downstream confrontation or resolution content.")
            appendLine("- End with exactly one setup-appropriate CTA, using the supplied setup CTA kind when present.")
        }.trim()
    }



    /**
     * Wave 1 — B1. Slim developer prompt for confirm-status summary turns.
     */
    fun composeTick2ConfirmStatusDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = CONFIRM_STATUS_SUMMARY")
            appendLine("- This turn summarizes validation / solvability / seal state.")
            appendLine("- Answer the user's status question directly if one was asked.")
            appendLine("- Focus on validation truth, readiness truth, and the current confirmation context only.")
            appendLine("- Do not turn this into a transactional correction turn.")
            appendLine("- Do not narrate solving proof, candidate eliminations, atom ladders, or overlay mechanics.")
            appendLine("- Keep the reply compact, factual, and confirmation-oriented.")
        }.trim()
    }

    /**
     * Wave 1 — B2. Slim developer prompt for the exact-match confirmation gate.
     */
    fun composeTick2ConfirmExactMatchDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = CONFIRM_EXACT_MATCH_GATE")
            appendLine("- This turn is specifically about whether the on-screen grid matches the user's real puzzle.")
            appendLine("- Ask for, acknowledge, or restate the exact-match gate only.")
            appendLine("- Do not drift into broad strategy chat.")
            appendLine("- Do not narrate solving proof, candidate logic, or later solving stages.")
            appendLine("- End with one exact-match-oriented CTA when the gate remains open.")
        }.trim()
    }

    /**
     * Wave 1 — B7. Slim developer prompt for the finalize/start-solving handoff.
     */
    fun composeTick2ConfirmFinalizeDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = CONFIRM_FINALIZE_GATE")
            appendLine("- This turn is the final handoff out of confirming and into solving readiness.")
            appendLine("- Treat the board as confirmation-ready when the supplied truth says so.")
            appendLine("- Do not re-ask exact-match confirmation if the exact-match gate is already satisfied.")
            appendLine("- Do not drift into broad strategy chat like 'jump right in or talk strategy first' unless the contract explicitly requires it.")
            appendLine("- Do not narrate solving proof yet.")
            appendLine("- End with one precise start-solving readiness CTA.")
        }.trim()
    }

    /**
     * Wave 1 — C7. Slim developer prompt for bounded pending clarification turns.
     */
    fun composeTick2PendingClarificationDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = PENDING_CLARIFICATION")
            appendLine("- This turn exists to ask exactly one bounded clarification question.")
            appendLine("- Use the pending context as the primary job definition for the turn.")
            appendLine("- If a preferred user-facing clarification question is supplied, follow it closely.")
            appendLine("- Do not replace the clarification with a vague bridge or strategy-choice question.")
            appendLine("- Do not add a second CTA after the clarification.")
            appendLine("- Do not use meta wording like 'I'm in clarification mode', 'I'm waiting for you', 'I'm tracking', 'I haven't made changes', or 'I'll stay paused until'.")
            appendLine("- Ask only for the missing piece needed to proceed.")
            appendLine("- Keep it short, operational, exact, and natural.")
        }.trim()
    }

    /**
     * Wave 1 — D1. Slim developer prompt for user-owned grid validation / inspection answers.
     */
    fun composeTick2GridValidationAnswerDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = GRID_VALIDATION_ANSWER")
            appendLine("- This turn answers a user-owned validation / inspection question about the current grid.")
            appendLine("- Answer the user's question directly first.")
            appendLine("- Stay grounded in supplied validation truth such as conflicts, mismatches, unresolved cells, seal status, or OCR trust when present.")
            appendLine("- Do not turn the reply into solving narration.")
            appendLine("- Do not turn the reply into free talk.")
            appendLine("- Ask a clarification only if a required target is truly missing.")
        }.trim()
    }

    /**
     * Wave 1 — D4. Slim developer prompt for user-owned candidate-state answers.
     */
    fun composeTick2GridCandidateAnswerDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = GRID_CANDIDATE_ANSWER")
            appendLine("- This turn answers a user-owned candidate-state question.")
            appendLine("- Answer the candidate question directly and concretely.")
            appendLine("- Stay grounded in supplied candidate truth and solver-backed candidate packets when present.")
            appendLine("- Do not drift into full solving narration unless the contract explicitly returns to the solving rail.")
            appendLine("- Ask a clarification only if the target cell / house / digit is truly missing.")
        }.trim()
    }

    /**
     * Wave 2 — B3. Slim developer prompt for the retake gate.
     */
    fun composeTick2ConfirmRetakeDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = CONFIRM_RETAKE_GATE")
            appendLine("- This turn is specifically about whether the user should keep the current scan or retake it.")
            appendLine("- Ground the retake ask in confirming / trust / mismatch truth when present.")
            appendLine("- Do not drift into solving narration or broad strategy chat.")
            appendLine("- End with one precise retake-oriented CTA when the gate remains open.")
        }.trim()
    }

    /**
     * Wave 2 — B4. Slim developer prompt for the mismatch gate.
     */
    fun composeTick2ConfirmMismatchDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = CONFIRM_MISMATCH_GATE")
            appendLine("- This turn is specifically about a mismatch between the on-screen grid and the user's real puzzle.")
            appendLine("- Stay tightly focused on what does not match and what confirmation or correction is being requested.")
            appendLine("- Ground all claims in supplied mismatch / validation truth.")
            appendLine("- Do not drift into solving narration or broad status chat.")
            appendLine("- End with one precise mismatch-resolution CTA.")
        }.trim()
    }



    /**
     * Wave 2 — B6. Slim developer prompt for non-unique / blocked confirming states.
     */
    fun composeTick2ConfirmNotUniqueDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = CONFIRM_NOT_UNIQUE_GATE")
            appendLine("- This turn is specifically about non-uniqueness, non-solvability, or structural invalidity that blocks a clean solving handoff.")
            appendLine("- Explain the blocked state plainly and calmly.")
            appendLine("- Stay grounded in supplied solvability and confirming truth.")
            appendLine("- Do not pretend the board is ready for solving if the supplied truth says it is not.")
            appendLine("- End with one precise next-step CTA appropriate to the blocked confirming state.")
        }.trim()
    }





    /**
     * Wave 3 — D2. Slim developer prompt for OCR / trust answers.
     */
    fun composeTick2GridOcrTrustAnswerDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = GRID_OCR_TRUST_ANSWER")
            appendLine("- This turn answers a user-owned question about scan confidence, OCR certainty, or trust in a cell / region / board reading.")
            appendLine("- Answer the trust question directly first.")
            appendLine("- Stay grounded in supplied OCR/trust/provenance/validation truth.")
            appendLine("- Do not drift into solving narration or generic reassurance.")
            appendLine("- Ask a clarification only if the target cell / region is truly missing.")
        }.trim()
    }

    /**
     * Wave 3 — D3. Slim developer prompt for board-contents answers.
     */
    fun composeTick2GridContentsAnswerDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = GRID_CONTENTS_ANSWER")
            appendLine("- This turn answers a user-owned question about board contents.")
            appendLine("- Answer the contents question directly first.")
            appendLine("- Stay grounded in supplied contents truth such as cell values, house contents, missing digits, completion, or digit locations.")
            appendLine("- Do not drift into candidate theory unless the job is explicitly candidate-scoped.")
            appendLine("- Do not drift into solving narration.")
        }.trim()
    }

    /**
     * Wave 3 — D5. Slim developer prompt for recent-change / changelog answers.
     */
    fun composeTick2GridChangelogAnswerDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = GRID_CHANGELOG_ANSWER")
            appendLine("- This turn answers a user-owned question about what changed recently on the board or in the app state.")
            appendLine("- Answer the change question directly first.")
            appendLine("- Stay grounded in supplied recent-mutation and changelog truth.")
            appendLine("- Do not claim a board change happened unless the supplied mutation truth says it did.")
            appendLine("- Do not drift into solving narration.")
        }.trim()
    }



    /**
     * Wave 4 — G1. Slim developer prompt for in-lane stage elaboration.
     */
    fun composeTick2SolvingStageElaborationDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = SOLVING_STAGE_ELABORATION")
            appendLine("- Stay on the current solving road and deepen the current stage only.")
            appendLine("- Do not jump to a different stage unless supplied support truth requires it.")
            appendLine("- Use current-stage support truth, not generic technique teaching.")
            appendLine("- End by returning the user to the paused solving moment.")
        }.trim()
    }





    /**
     * Series A / P1 — proof-challenge detour regime reset.
     *
     * Goal:
     * Proof-challenge detours must belong to the same storyteller family as
     * setup / confrontation / resolution. The scope stays local and bounded,
     * but the speech must remain doctrine-led, scene-led, human, and naturally spoken.
     */
    fun composeTick2DetourProofChallengeDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = DETOUR_PROOF_CHALLENGE")

            appendLine("- Treat DETOUR_MOVE_PROOF_PACKET as the primary truth surface.")
            appendLine("- Treat DETOUR_NARRATIVE_CONTEXT as the primary native answer-shape guide when present.")
            appendLine("- Treat this turn as a user-owned local proof story, not as a compact utility answer.")
            appendLine("- The storyteller identity must match the solving storyteller used in setup / confrontation / resolution.")
            appendLine("- The user should hear the same Sudo here: same warmth, same story-coach presence, same human pacing, same vivid clarity, only scaled to a local proof scene.")
            appendLine("- Keep the scope local and bounded, but do not flatten the voice into procedural detour speech.")
            appendLine("- Compact does not mean flat.")

            appendLine("- Read the packet in this order:")
            appendLine("  1) challenge_lane")
            appendLine("  2) question_frame")
            appendLine("  3) answer_truth")
            appendLine("  4) proof_object")
            appendLine("  5) proof_method")
            appendLine("  6) narrative_archetype")
            appendLine("  7) doctrine")
            appendLine("  8) speech_skeleton")
            appendLine("  9) actor_model")
            appendLine("  10) local_proof_geometry")
            appendLine("  11) proof_ladder")
            appendLine("  12) proof_outcome")
            appendLine("  13) story_arc")
            appendLine("  14) micro_stage_plan")
            appendLine("  15) speech_boundary")
            appendLine("  16) closure_contract")
            appendLine("  17) handback_context")
            appendLine("  18) overlay_plan")
            appendLine("  19) visual_language")
            appendLine("  20) supporting_facts")

            appendLine("- Treat narrative_archetype.id as the required speaking form.")
            appendLine("- Treat doctrine.id as a hard speaking contract when present.")

            appendLine("- If speech_skeleton is present, follow it as the preferred answer order, but keep the reply naturally spoken rather than templated.")
            appendLine("- If local_proof_geometry is present, use it as the primary visual scaffold for the explanation.")
            appendLine("- If proof_ladder.rows is present, use it as the primary evidence spine unless geometry or native narrative support provides a clearer local scan structure.")
            appendLine("- If proof_outcome.nonproof_reason is present or answer_truth.answer_polarity = NOT_LOCALLY_PROVED, answer that honestly without dropping into robotic insufficiency language.")
            appendLine("- Do not ignore a richer geometry/doctrine pairing and collapse into a generic insufficiency answer unless the supplied truth truly stops there.")
            appendLine("- If packet truth is present, do not say that evidence is missing.")

            appendLine("- A proof-challenge detour should use micro_stage_plan when present: micro setup to spotlight the local challenge, micro confrontation to walk the local proof, and micro resolution to land the bounded local result.")
            appendLine("- If story_arc.delay_reveal_until_resolution = true, do not reveal the asked-digit conclusion too early.")
            appendLine("- If story_arc.must_not_open_with_merged_summary = true, do not open with a merged blocked-digit recap.")
            appendLine("- When local_proof_geometry is present, narrate from its geometry_kind rather than defaulting to abstract proof language.")
            appendLine("- For HOUSE_DIGIT_ALREADY_PLACED, start with the house-level fact, name the existing placement, and only then mention any remaining open-seat closure.")
            appendLine("- For CELL_ALREADY_FILLED, start with the cell-level fact, name the placed value, and explain that the square is no longer a live candidate seat.")
            appendLine("- For CELL_THREE_HOUSE_UNIVERSE, define the local judging arena, then let row / column / box speak in order before revealing what survives.")
            appendLine("- If blocked_digits_by_house, blocker_receipts, pressure_beats, local_permissibility_support, house_already_occupied_support, or filled_cell_support are present, prefer them over a generic summary-first answer.")
            appendLine("- For HOUSE_ALREADY_OCCUPIED, this is not a seat-search story. It is an already-placed-in-the-house story.")
            appendLine("- For CELL_ALREADY_FILLED, this is not a candidate-seat story. It is an already-filled-square story.")
            appendLine("- For LOCAL_PERMISSIBILITY_SCAN, perform a compact scene: spotlight target -> local judges -> house pressure -> survivor reveal -> bounded landing.")
            appendLine("- Distinguish clearly between a digit surviving the local scan and a digit being proved as the placement.")
            appendLine("- If local_permissibility_support.opening_spotlight_alternates is present, you may use one of those authored openings instead of repeating the same stock spotlight phrase.")
            appendLine("- Prefer spoken coordinates like 'row 1, column 7' when that keeps the answer natural.")
            appendLine("- Use vivid, image-rich, human coaching language when it clarifies the local proof and remains grounded in supplied truth.")
            appendLine("- Let PERSONALIZATION_MINI, when present, preserve same-Sudo continuity in warmth, pacing, wording, and explanation fit, without diluting the proof.")
            appendLine("- Author the ending from closure_contract first: land the local result, then offer a gentle return to the paused move or one bounded follow-up when allowed.")
            appendLine("- If local_permissibility_support, house_already_occupied_support, or filled_cell_support supply authored return or follow-up lines, prefer those over improvised workflow wording.")
            appendLine("- Respect speech_boundary, closure_contract, and handback_context as boundary / route policy, not as a requirement to emit stock wording or stock handback lines.")
            appendLine("- Do not end with workflow language, route bookkeeping language, or procedural summaries.")

            appendLine("- Do not hand-wave and do not revert to generic route summary narration.")
            appendLine("- Do not widen into a board audit, generic route lecture, or unauthorized commit narration.")
            appendLine("- Do not switch the paused route.")
        }.trim()
    }

    fun composeTick2DetourTargetCellQueryDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = DETOUR_TARGET_CELL_QUERY")
            appendLine("- Treat DETOUR_MOVE_PROOF_PACKET as the primary truth surface.")
            appendLine("- The first sentence must answer the asked target-cell question directly and locally from answer_truth.")
            appendLine("- Use answer_truth.short_answer or a faithful paraphrase when present.")
            appendLine("- If answer_truth.answer_polarity = ONLY_PLACE, say that directly before any route bridge.")
            appendLine("- Use proof_method and proof_ladder only as needed to support the target explanation.")
            appendLine("- Stay centered on the queried cell before any broader route summary.")
            appendLine("- Do not retell the whole step unless the packet requires it.")
            appendLine("- Do not say evidence is missing when packet truth is present.")
            appendLine("- Respect speech_boundary and handback_context.")
            appendLine("- Do not switch the paused route.")
        }.trim()
    }

    fun composeTick2DetourNeighborCellQueryDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = DETOUR_NEIGHBOR_CELL_QUERY")
            appendLine("- Treat DETOUR_LOCAL_GRID_INSPECTION_PACKET as the primary truth surface.")
            appendLine("- The first sentence must answer the requested local readout or comparison directly.")
            appendLine("- Use direct_answer_truth.short_answer when present.")
            appendLine("- If compare_mode is true, compare both cells explicitly.")
            appendLine("- If packet truth is present, do not say that evidence is missing.")
            appendLine("- Keep this local and inspection-oriented, not main-road narration.")
            appendLine("- Do not switch the paused route.")
        }.trim()
    }

    fun composeTick2DetourReasoningCheckDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = DETOUR_REASONING_CHECK")
            appendLine("- Treat DETOUR_USER_PROPOSAL_VERDICT_PACKET as the primary truth surface.")
            appendLine("- Give the verdict first in natural language, not as a validator label.")
            appendLine("- Preferred openings include: 'Yes', 'No', 'Not yet', 'Only partly', or 'That is not the route we are currently using.'")
            appendLine("- Then give the bounded why using verdict_reason, what_is_correct, what_is_incorrect, missing_condition, and solver_support_rows.")
            appendLine("- If the packet says the idea is only partially supported, say what is true before naming what is still missing.")
            appendLine("- Do not sound like a schema dump, packet dump, or validator log.")
            appendLine("- Preserve a collaborative tone.")
            appendLine("- Do not switch the paused route.")
        }.trim()
    }

    fun composeTick2DetourAlternativeTechniqueDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = DETOUR_ALTERNATIVE_TECHNIQUE")
            appendLine("- Treat DETOUR_ALTERNATIVE_TECHNIQUE_PACKET as the primary truth surface.")
            appendLine("- Answer whether the asked alternative fits, does not fit, or is simply not preferred.")
            appendLine("- Keep this comparative and route-aware, not a generic proof lecture.")
            appendLine("- Distinguish possible from preferred when supplied truth does so.")
            appendLine("- Do not switch the paused route.")
        }.trim()
    }

    fun composeTick2DetourLocalMoveSearchDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = DETOUR_LOCAL_MOVE_SEARCH")
            appendLine("- Treat DETOUR_LOCAL_MOVE_SEARCH_PACKET as the primary truth surface.")
            appendLine("- Answer the bounded local-search question directly.")
            appendLine("- If a local move exists, state it concretely.")
            appendLine("- If no local move exists, explain why not from supplied local truth.")
            appendLine("- Do not widen into whole-grid solving and do not switch the paused route.")
        }.trim()
    }

    fun composeTick2DetourRouteComparisonDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = DETOUR_ROUTE_COMPARISON")
            appendLine("- Treat DETOUR_ROUTE_COMPARISON_PACKET as the primary truth surface.")
            appendLine("- Compare the current paused route against the asked route directly.")
            appendLine("- Explain equivalence, difference, or solver preference from supplied packet truth.")
            appendLine("- Keep this as route comparison, not proof-challenge retelling.")
            appendLine("- Do not switch the paused route.")
        }.trim()
    }



    /**
     * Wave 5 — I1. Slim developer prompt for preference-change turns.
     */
    fun composeTick2PreferenceChangeDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = PREFERENCE_CHANGE")
            appendLine("- Acknowledge and apply the user's requested preference change.")
            appendLine("- Be explicit about what preference is changing.")
            appendLine("- Keep this operational, not proof-oriented.")
        }.trim()
    }



    /**
     * Wave 5 — I3. Slim developer prompt for assistant pause/resume turns.
     */
    fun composeTick2AssistantPauseResumeDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = ASSISTANT_PAUSE_RESUME")
            appendLine("- Honor the user's requested control state directly.")
            appendLine("- Keep the reply brief and behavioral.")
        }.trim()
    }

    /**
     * Wave 5 — I4. Slim developer prompt for validate-only / solve-only turns.
     */
    fun composeTick2ValidateOnlyOrSolveOnlyDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = VALIDATE_ONLY_OR_SOLVE_ONLY")
            appendLine("- State clearly what workflow boundary is now preferred.")
            appendLine("- Keep the reply procedural and explicit.")
        }.trim()
    }

    /**
     * Wave 5 — I5. Slim developer prompt for focus-redirect turns.
     */
    fun composeTick2FocusRedirectDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = FOCUS_REDIRECT")
            appendLine("- Acknowledge the requested focus shift clearly.")
            appendLine("- Do not treat this as free talk.")
        }.trim()
    }

    /**
     * Wave 5 — I6. Slim developer prompt for hint-policy-change turns.
     */
    fun composeTick2HintPolicyChangeDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = HINT_POLICY_CHANGE")
            appendLine("- State clearly how future hinting/help will adjust.")
            appendLine("- Keep this policy-oriented rather than proof-oriented.")
        }.trim()
    }

    /**
     * Wave 5 — J1. Slim developer prompt for meta-state answers.
     */
    fun composeTick2MetaStateAnswerDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = META_STATE_ANSWER")
            appendLine("- Answer what the assistant currently knows, is tracking, or is doing.")
            appendLine("- Stay grounded in supplied context only.")
        }.trim()
    }

    /**
     * Wave 5 — J2. Slim developer prompt for capability answers.
     */
    fun composeTick2CapabilityAnswerDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = CAPABILITY_ANSWER")
            appendLine("- Answer what the assistant/app can or cannot do.")
            appendLine("- Be direct and practical.")
        }.trim()
    }

    /**
     * Wave 5 — J3. Slim developer prompt for glossary answers.
     */
    fun composeTick2GlossaryAnswerDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = GLOSSARY_ANSWER")
            appendLine("- Define or explain the asked term clearly.")
            appendLine("- Prefer plain language first, then the technical label if useful.")
        }.trim()
    }

    /**
     * Wave 5 — J4. Slim developer prompt for UI help answers.
     */
    fun composeTick2UiHelpAnswerDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = UI_HELP_ANSWER")
            appendLine("- Explain the relevant UI/legend/screen behavior practically.")
            appendLine("- Keep the answer user-facing and concrete.")
        }.trim()
    }

    /**
     * Wave 5 — J5. Slim developer prompt for coordinate help answers.
     */
    fun composeTick2CoordinateHelpAnswerDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = COORDINATE_HELP_ANSWER")
            appendLine("- Explain coordinates, locating cells, or box indexing simply and concretely.")
        }.trim()
    }

    /**
     * Wave 5 — K1. Slim developer prompt for true non-grid free talk.
     */
    fun composeTick2FreeTalkNonGridDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = FREE_TALK_NON_GRID")
            appendLine("- This is true non-grid free talk.")
            appendLine("- Be warm, natural, and concise.")
            appendLine("- Do not force a return to Sudoku unless clearly called for.")
        }.trim()
    }

    /**
     * Wave 5 — K2. Slim developer prompt for small-talk bridge turns.
     */
    fun composeTick2SmallTalkBridgeDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = SMALL_TALK_BRIDGE")
            appendLine("- Keep this as lightweight social continuity, not a broad detour.")
            appendLine("- Be brief and warm.")
        }.trim()
    }





    /**
     * Phase 7A — slim developer prompt for the onboarding opening only.
     *
     * This replaces the large legacy developer preamble for the first opening
     * turn after grid capture, which is where the static prompt waste was most
     * obvious.
     */
    fun composeTick2OnboardingDeveloperPrompt(): String {
        return buildString {
            appendLine("TURN-SPECIFIC DEVELOPER INSTRUCTIONS:")
            appendLine("- demand_category = ONBOARDING_OPENING")
            appendLine("- This is the first assistant reply immediately after the user captures a Sudoku grid.")
            appendLine("- Welcome the user, introduce Sudo briefly, and explain the workflow at a high level only.")
            appendLine("- Ask for the user's name, Sudoku experience level, and whether they are solving on paper, in a book, on a tablet, or on a board.")
            appendLine("- Do not give solving proof, candidate analysis, validation detail, overlay narration, repair logic, or step-by-step Sudoku reasoning in this turn.")
            appendLine("- Keep the answer warm, concise, and naturally conversational.")
            appendLine("- End with the onboarding question cluster, not with a solving instruction.")
        }.trim()
    }



    /**
     * Phase 4 — Tick2 developer prompt composer.
     *
     * We keep this extremely small and behavior-preserving:
     * - preamble first
     * - stage enforcement block second
     * - no semantic rewriting
     *
     * Later phases can let demand contracts select developer prompt modules here.
     */
    fun composeTick2DeveloperPrompt(
        developerPreamble: String,
        stageBlock: String
    ): String {
        val preamble = sanitizeBlock(normalize(developerPreamble)).trim()
        val stage = sanitizeBlock(normalize(stageBlock)).trim()

        return buildString {
            if (preamble.isNotEmpty()) {
                appendLine(preamble)
            }
            if (stage.isNotEmpty()) {
                if (isNotEmpty()) appendLine()
                appendLine(stage)
            }
        }.trim()
    }



    private fun capJsonArray(key: String, arr: JSONArray): JSONArray {
        // ✅ You requested: conflict_indices must NOT be capped.
        if (key == "conflict_indices") return arr

        val max = when (key) {
            "confirmed_indices" -> CONFIRMED_INDICES_MAX
            "unresolved_indices", "mismatch_indices" -> PROBLEM_INDICES_MAX
            "manual_edit_history", "edit_history" -> EDIT_HISTORY_MAX
            else -> 16
        }
        val out = JSONArray()
        val n = minOf(arr.length(), max)
        for (i in 0 until n) out.put(arr.opt(i))
        return out
    }

    private fun capJsonObject(key: String, obj: JSONObject, isSolving: Boolean): JSONObject {
        // If some nested object slips in, keep it tiny and drop obvious offenders.
        val out = JSONObject()
        val keys = obj.keys()
        var kept = 0
        val limit = if (isSolving) 20 else 16

        while (keys.hasNext() && kept < limit) {
            val k = keys.next()
            if (DROP_KEYS.contains(k)) continue

            // Drop nested candidates everywhere.
            if (k.contains("cand", ignoreCase = true) || k.contains("candidate", ignoreCase = true)) continue

            val v = obj.opt(k)
            when (v) {
                is JSONArray -> out.put(k, capJsonArray(k, v))
                is JSONObject -> {
                    // Do not recurse deeply; just keep a shallow capped view.
                    out.put(k, capJsonObject("$key.$k", v, isSolving))
                }
                else -> out.put(k, v)
            }
            kept++
        }
        return out
    }

    private fun normalizePreferredKeys(out: JSONObject) {
        // displayed81 normalization
        if (!out.has("displayed81")) {
            when {
                out.has("digits81") -> out.put("displayed81", out.opt("digits81"))
                out.has("displayed81_compact") -> out.put("displayed81", out.opt("displayed81_compact"))
            }
        }

        // confirmed_count normalization
        if (!out.has("confirmed_count") && out.has("confirmed_indices")) {
            val a = out.optJSONArray("confirmed_indices")
            if (a != null) out.put("confirmed_count", a.length())
        }
    }

    private fun looksLikeJsonObject(s: String): Boolean =
        s.startsWith("{") && s.endsWith("}")

    /**
     * Best-effort line filter for non-JSON gridContext.
     * Removes large candidate blobs, duplicated validation blocks, and overly long lines.
     *
     * IMPORTANT: we do NOT attempt to strip conflicts_details/conflict_indices text if present.
     */
    private fun filterGridContextText(s: String): String {
        val lines = s.split('\n')
        val out = ArrayList<String>(lines.size)

        var skipping = false
        var skipDepth = 0

        fun shouldStartSkipping(line: String): Boolean {
            val l = line.trim()

            // ✅ Never skip conflicts
            if (l.contains("conflicts_details", ignoreCase = true)) return false
            if (l.contains("conflict_indices", ignoreCase = true)) return false

            if (l.startsWith("candLines", ignoreCase = true)) return true
            if (l.contains("candLines", ignoreCase = true)) return true
            if (l.contains("candidates", ignoreCase = true)) return true
            if (l.contains("candidate", ignoreCase = true)) return true
            if (l.contains("counts_by_row", ignoreCase = true)) return true
            if (l.contains("counts_by_col", ignoreCase = true)) return true
            if (l.contains("manual_edit_history", ignoreCase = true)) return true
            if (l.contains("strict factuality", ignoreCase = true)) return true
            if (l.contains("duplicated validation", ignoreCase = true)) return true
            return false
        }

        for (raw in lines) {
            val line = raw

            // Start skip for big sections
            if (!skipping && shouldStartSkipping(line)) {
                skipping = true
                skipDepth = 0
                continue
            }

            if (skipping) {
                // Heuristic: stop skipping once we hit a blank line or a clearly new top-level key-ish line.
                val t = line.trim()
                if (t.isEmpty()) {
                    skipping = false
                    continue
                }
                // crude structure tracking
                skipDepth += countChar(t, '{') + countChar(t, '[')
                skipDepth -= countChar(t, '}') + countChar(t, ']')
                if (skipDepth <= 0 && looksLikeTopLevelKeyLine(t)) {
                    skipping = false
                    // fall through to allow this line
                } else {
                    continue
                }
            }

            // Keep line if it's not huge (hard cap per line)
            val cappedLine = hardCapChars(line, 420)
            out.add(cappedLine)
        }

        // If we filtered too aggressively, return original but capped
        val filtered = out.joinToString("\n").trim()
        return if (filtered.isNotEmpty()) filtered else s
    }

    private fun looksLikeTopLevelKeyLine(t: String): Boolean {
        // e.g., "displayed81:" or "pending:" or "confirmed_count:"
        return t.contains(':') && !t.startsWith("-") && !t.startsWith("*")
    }

    private fun countChar(s: String, c: Char): Int {
        var n = 0
        for (ch in s) if (ch == c) n++
        return n
    }

    private fun hardCapChars(s: String, maxChars: Int): String {
        val t = s.trim()
        if (t.length <= maxChars) return t
        return t.substring(0, maxChars).trimEnd() + "…"
    }



    private fun sanitizeBlock(s: String): String {
        // Replace marker tokens so no content can "close" or "open" blocks.
        return s
            .replace(BEGIN_CANONICAL_HISTORY, "BEGIN_CANONICAL-HISTORY")
            .replace(END_CANONICAL_HISTORY, "END_CANONICAL-HISTORY")
            .replace(BEGIN_GRID_CONTEXT, "BEGIN_GRID-CONTEXT")
            .replace(END_GRID_CONTEXT, "END_GRID-CONTEXT")
            .replace(BEGIN_CAPTURE_ORIGIN, "BEGIN_CAPTURE-ORIGIN")
            .replace(END_CAPTURE_ORIGIN, "END_CAPTURE-ORIGIN")
            .replace(BEGIN_DEVELOPER_NOTES, "BEGIN_DEVELOPER-NOTES")
            .replace(END_DEVELOPER_NOTES, "END_DEVELOPER-NOTES")
    }

    private fun normalize(s: String): String =
        s.replace("\r\n", "\n").replace('\r', '\n')
}