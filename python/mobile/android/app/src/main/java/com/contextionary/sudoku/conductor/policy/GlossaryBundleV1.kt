package com.contextionary.sudoku.conductor.policy

import org.json.JSONArray
import org.json.JSONObject

/**
 * GlossaryBundleV1 — Player-language bridge.
 *
 * Purpose:
 * - Teach Tick2 how to translate app vocabulary + UI legend into natural Sudoku-player speech.
 * - Provide do_say / dont_say, agenda anchors, and short response templates.
 *
 * This is intentionally compact to avoid prompt bloat.
 */
object GlossaryBundleV1 {

    private const val VERSION = "glossary_v1"

    fun toFactBundle(): FactBundleV1 =
        FactBundleV1(
            type = FactBundleV1.Type.GLOSSARY_BUNDLE,
            payload = toJson()
        )

    fun toJson(): JSONObject = JSONObject().apply {
        put("version", VERSION)

        // -------------------------
        // Speech policy
        // -------------------------
        put("speech_policy", JSONObject().apply {
            put("coordinate_speech", JSONObject().apply {
                put(
                    "do",
                    "Speak as 'row X column Y'. Use rXcY only in on-screen text, logs, or when user explicitly asks for coordinates."
                )
                put(
                    "dont",
                    "Don’t rattle off r7c2,r9c6 as primary spoken language."
                )
            })
            put("box_naming", JSONObject().apply {
                put(
                    "do",
                    "Use human box names: top-left, top-middle, top-right, middle-left, center, middle-right, bottom-left, bottom-middle, bottom-right."
                )
                put(
                    "dont",
                    "Don’t say 'box 7' unless you also include the human position name."
                )
                put(
                    "optional_numbering_rule",
                    "If numbering boxes, define once: left-to-right, top-to-bottom."
                )
            })
            put("avoid_terms", JSONArray().apply {
                put("unresolved")
                put("low confidence")
                put("severity")
                put("conflict scoop")
                put("hiding")
                put("atom 0")
                put("beat_kind")
                put("overlay_intent")
                put("overlay_role")
                put("witness_by_peer")
                put("toolplan")
                put("pending_before")
                put("expected_answer_kind")
            })
            put("preferred_terms", JSONObject().apply {
                put("low_confidence", "scan isn’t sure")
                put("unresolved", "needs your confirmation / question-mark cell")
                put("conflict", "breaks Sudoku rules (duplicate in a row/column/box)")
                put("mismatch", "one of your filled-in answers is wrong")
                put("auto_correct", "I made a safe scan correction")

                put("atom", "story beat")
                put("beat", "step in the explanation")
                put("spotlight", "set the scene")
                put("commit", "the final placement")
                put("witness", "the proof you can point at")
                put("overlay_frame", "what I’m highlighting on the grid")
            })
            put("phase_rule", "Only reference the current phase if the user explicitly asks (CONFIRMING/SEALING/SOLVING).")
            put("brevity_rule", "Answer the user’s exact question first, with the minimal necessary info. Don’t list all blanks unless asked.")
            put(
                "proof_discipline_rule",
                "Never invent digits, candidates, or patterns. If you state a Sudoku claim, tie it to witnesses (cells, houses, candidate eliminations, or explicit pattern structure) provided by the app."
            )
            put(
                "overlay_language_rule",
                "If you mention highlighting, describe it in player terms (row/column/box/cell focus). Don’t mention internal overlay roles or ids."
            )
        })

        // -------------------------
        // Narration contract summary
        // -------------------------
        put("narration_contract", JSONObject().apply {
            put(
                "what_is_a_step_story",
                "For each solving step, the app brain provides a ladder of story beats (atoms). The assistant performs them in order: setup (Spotlight) → proof chain → final placement (Commit)."
            )
            put(
                "beats",
                JSONArray().apply {
                    put(JSONObject().apply {
                        put("beat_kind", "SPOTLIGHT")
                        put("meaning", "Setup: introduce the next technique and point the user’s attention to the focal opportunity.")
                        put("say", "Let’s zoom in on this area first…")
                        put("dont_say", "Atom 0 / spotlight frame_id …")
                    })
                    put(JSONObject().apply {
                        put("beat_kind", "WITNESS_ELIMINATION")
                        put("meaning", "A proof beat that removes candidates using explicit witnesses such as cells or houses.")
                        put("say", "Because of what’s already in this row, column, or box, that digit can’t go here.")
                        put("dont_say", "witness_kind=BLOCKS_CANDIDATE")
                    })
                    put(JSONObject().apply {
                        put("beat_kind", "LOCK_IN")
                        put("meaning", "A proof beat that locks a digit or location, such as only place left in a house.")
                        put("say", "That leaves only one possible place for this digit in the house.")
                        put("dont_say", "lock_digit role / lock-in object")
                    })
                    put(JSONObject().apply {
                        put("beat_kind", "COMMIT")
                        put("meaning", "Resolution: the final placement for the step.")
                        put("say", "So the move is: row X column Y is …")
                        put("dont_say", "commit beat / result_place overlay role")
                    })
                }
            )
            put(
                "overlays",
                "The app may provide overlay frames that match beats. The assistant can reference them as 'what I’m highlighting' without mentioning internal ids or roles."
            )
            put(
                "witnessing",
                "A witness is the concrete Sudoku evidence the user can verify: cells, houses, candidate eliminations, restricted positions, or pattern structure that justifies a claim."
            )
        })

        // -------------------------
        // UI legend
        // -------------------------
        put("ui_legend", JSONArray().apply {
            put(JSONObject().apply {
                put("cue", "printed_font")
                put("means", "Given clue from the original puzzle")
                put("say", "This is a printed clue (a given).")
            })
            put(JSONObject().apply {
                put("cue", "handwritten_font")
                put("means", "Player-filled entry (solution layer)")
                put("say", "This looks like one of your filled-in answers.")
            })
            put(JSONObject().apply {
                put("cue", "cyan_border")
                put("means", "Auto-correct changed this cell")
                put("say", "Cyan outline means I auto-corrected this digit from the scan.")
            })
            put(JSONObject().apply {
                put("cue", "red_border")
                put("means", "Still uncertain; needs user confirmation")
                put("say", "Red outline means I’m not confident—please confirm what’s written there.")
            })
            put(JSONObject().apply {
                put("cue", "yellow_pulse_border")
                put("means", "Focus cell currently being asked about")
                put("say", "Yellow pulse means that’s the cell I’m asking you about right now.")
            })
            put(JSONObject().apply {
                put("cue", "dot_marker")
                put("means", "Medium confidence OCR")
                put("say", "The dot means the scan is somewhat unsure—worth a quick double-check.")
            })
            put(JSONObject().apply {
                put("cue", "underline_marker")
                put("means", "Low confidence OCR")
                put("say", "Underline means the scan is very unsure—please confirm that digit.")
            })
            put(JSONObject().apply {
                put("cue", "mini_digits_candidates")
                put("means", "Candidates / pencil marks")
                put("say", "Tiny digits are candidate notes—possible values, not confirmed.")
            })
        })

        // -------------------------
        // Terms
        // -------------------------
        put("terms", JSONArray().apply {

            // --- Existing core terms ---
            term("confidence", "How sure the scan is (0–1).", "I’m sure / I’m not fully sure.", "confidence=0.63")
            term("low_confidence_cell", "A cell where the scan might have guessed wrong.", "The scan isn’t sure about this digit.", "low confidence")
            term("conflict", "Current grid breaks Sudoku rules (duplicate in row/column/box).", "This breaks Sudoku rules right now (two 9s in the same row).", "conflict")
            term("mismatch", "Your filled-in answer differs from the correct value implied by the givens.", "One of your filled-in answers is wrong here.", "mismatch")
            term("given", "Original printed clue digit.", "This is a printed clue (given).", "truth_givens")
            term("player_entry", "Digit you filled in while solving.", "This looks like one of your filled-in answers.", "truth_solution")
            term("auto_correct", "App applied a safe fix to likely scan errors.", "I made a safe scan correction here.", "autocorrect")
            term("changed_cell", "Cell changed by auto-correct.", "Cyan outline = I corrected this from the scan.", "changedIndices")
            term("unresolved_cell", "Still uncertain after auto-correct; needs confirmation.", "This cell needs your confirmation.", "unresolvedIndices")
            term("unique_solution", "Exactly one solution exists from givens.", "This puzzle has one definite solution.", "deduced_solution_count_capped==1")
            term("multiple_solutions", "More than one solution exists from givens (scan likely wrong).", "The clues don’t pin down a single solution—likely a scan issue.", ">=2")
            term("no_solution", "No solution exists as currently read (scan likely wrong).", "As read, this can’t be solved—usually a misread clue.", "==0")

            // --- Narration / atoms ---
            term(
                "narrative_atom",
                "One authored beat in the step story: a small chunk of explanation that has matching evidence and often a matching overlay highlight.",
                "Let me walk you through this in small steps.",
                "Atom 3 says… / beat_kind=WITNESS_ELIMINATION"
            )
            term(
                "atom_0",
                "The first beat of a solving step: the setup / intro anchor. For advanced techniques it should align to the final target, introduce the technique lens, expose the trigger pattern, explain why that trigger is valid, and bridge it to the target.",
                "Let’s zoom in on the target and see why this technique fits here.",
                "Atom 0 / intro anchor / setup packet"
            )
            term(
                "narrative_atoms_v1",
                "The ordered list of story beats for a single solving step.",
                "I’ll explain this in a short sequence, then we’ll place the digit.",
                "narrative_atoms_v1"
            )
            term(
                "beat_kind",
                "The type of story beat, such as Spotlight, proof beats, or Commit.",
                "First I’ll set the scene, then show the proof, then we place it.",
                "beat_kind"
            )
            term(
                "spotlight",
                "Setup beat: introduce the technique and point attention to the focal opportunity.",
                "Let’s focus your eyes on this area first.",
                "SPOTLIGHT beat / Atom 0"
            )
            term(
                "proof_beat",
                "Any beat that advances the logic before the final placement.",
                "Here’s the reason it can’t be those digits…",
                "proof beat #2"
            )
            term(
                "commit",
                "The final beat where the assistant states the placement once the user is ready.",
                "Alright—ready for the move?",
                "COMMIT beat / result_place role"
            )
            term(
                "archetype",
                "A high-level technique family bucket used by the narration system, such as Hidden Singles, Naked Singles, Subsets, Intersections, Fish, Wings, Chains, Rings, or other advanced families.",
                "This is a Subsets-style step.",
                "archetype=SINGLES"
            )
            term(
                "spoiler_level",
                "What is allowed to be revealed in this beat, such as none, candidates, or digit.",
                "I can keep it as a hint, or I can reveal the digit—your call.",
                "spoiler_level=DIGIT"
            )

            // --- Witnesses / evidence ---
            term(
                "witness",
                "A concrete piece of Sudoku evidence the user can verify: a cell, house, elimination, restriction, or structured pattern fact that supports a claim.",
                "Because of what’s already in this row, column, or box, that option is blocked.",
                "witness_kind / witness_by_peer"
            )
            term(
                "witness_cells",
                "The exact cells you can point to as evidence for a claim.",
                "Look at these cells—they force the conclusion.",
                "witness_cells=[…]"
            )
            term(
                "witness_by_peer",
                "Evidence grouped by peer relationship, such as same row, same column, or same box.",
                "The key blockers are in the same row, column, or box as the target cell.",
                "witness_by_peer"
            )
            term(
                "reason_code",
                "A short internal label for why a witness supports a claim.",
                "The row or box blocks that candidate.",
                "reason_code=witness_blocks_candidate"
            )

            // --- Overlay lane ---
            term(
                "overlay_frame",
                "A visual highlight frame on the grid that matches a story beat.",
                "See what I’m highlighting on the grid.",
                "overlay_frame_id / overlay_role"
            )
            term(
                "overlay_intent",
                "The high-level purpose of an overlay frame, such as spotlight, witness, sweep, or commit.",
                "I’m highlighting the key evidence first.",
                "SHOW_WITNESS"
            )
            term(
                "overlay_role",
                "The internal label for each highlighted element, such as focus, witness, peer, subset cell, or pattern cell.",
                "I’m highlighting the target cell and the important cells around it.",
                "role=peer / role=primary_house"
            )

            // --- Sudoku structural vocabulary ---
            term(
                "target_cell",
                "The main cell the step is about.",
                "Our target is row X column Y.",
                "target_cell"
            )
            term(
                "target_digit",
                "The digit being placed, only when the current reveal level allows it.",
                "I can reveal the digit when you’re ready.",
                "target_digit"
            )
            term(
                "house",
                "A row, column, or box.",
                "In this row, column, or box…",
                "houseRefV2"
            )
            term(
                "primary_house",
                "The main row, column, or box where the logic happens for this step.",
                "Let’s work inside this row, column, or box first.",
                "primary_house"
            )
            term(
                "secondary_house",
                "A supporting row, column, or box used as extra evidence for the step.",
                "And this also lines up with this other row, column, or box.",
                "secondary_house"
            )
            term(
                "peer",
                "A cell that shares a row, column, or box with another cell.",
                "Cells in the same row, column, or box affect each other.",
                "peer cell"
            )
            term(
                "candidate",
                "A possible digit for a cell, not confirmed yet.",
                "That digit is still a candidate here.",
                "mini_digits_candidates"
            )
            term(
                "elimination",
                "Removing a candidate because Sudoku rules make it impossible.",
                "So we can cross that digit out here.",
                "eliminate_digit"
            )
            term(
                "lock_in",
                "A constraint that forces a digit or location.",
                "That leaves only one place it can go.",
                "LOCK_IN"
            )

            // --- Technique / pattern vocabulary ---
            term(
                "trigger_pattern",
                "The structured pattern that makes an advanced technique relevant in this step.",
                "Here is the pattern we want to notice first.",
                "trigger_pattern"
            )
            term(
                "trigger_explanation",
                "The explanation of why the trigger pattern is real. It usually shows why certain cells or houses qualify as the pattern’s defining structure.",
                "Here’s why these cells or houses really count as the pattern.",
                "trigger_explanation"
            )
            term(
                "trigger_bridge",
                "The bridge from the advanced pattern to the target cell or target house: why the pattern matters for the downstream resolution.",
                "This is why the pattern matters for our target.",
                "trigger_bridge"
            )
            term(
                "pattern_identity",
                "The pattern’s type label, such as subset naked, subset hidden, or box-line interaction.",
                "This is the kind of pattern we’ve found.",
                "pattern_identity"
            )
            term(
                "pattern_structure",
                "The structural pieces of the pattern: key houses, defining cells, locked or support digits, sweep cells, and related houses.",
                "These are the cells and houses that make up the pattern.",
                "pattern_structure"
            )
            term(
                "pattern_explanation",
                "The proof-level explanation for why the pattern’s defining cells or houses have the required structure.",
                "These witnesses explain why the pattern is valid.",
                "pattern_explanation"
            )
            term(
                "pattern_to_target_bridge",
                "The part of the step story that connects the trigger pattern to target cleanup or target resolution.",
                "Now let’s connect that pattern back to the target.",
                "pattern_to_target_bridge"
            )
            term(
                "pattern_type",
                "The canonical structural family assigned during step normalization, such as subset, hidden_subset, box_line_interaction, fish, wing, chain, ring, leftovers, or fallback.",
                "This tells us the shape of the technique pattern.",
                "pattern_type"
            )
            term(
                "pattern_subtype",
                "The more specific subtype of the canonical pattern, such as naked_pair, hidden_triplet, pointing, claiming, xwing, or ywing.",
                "This tells us the specific flavor of the pattern.",
                "pattern_subtype"
            )

            // --- SUBSETS vocabulary (broad, not technique-specific) ---
            term(
                "subset_member",
                "A cell that belongs to the subset’s defining support set. For a naked subset, these are the cells that contain the subset digits. For a hidden subset, these are the support cells inside the house that carry the hidden digits.",
                "These are the cells that make up the subset itself.",
                "subset_member"
            )
            term(
                "supporting_cell",
                "A support cell used by a hidden subset or other restriction-style subset logic. In practice it is the subset-side cell set that carries the restricted digits.",
                "These are the support cells for the subset.",
                "supporting_cell"
            )
            term(
                "subset_cells",
                "The defining cell set of a subset pattern: the cells that together express the subset structure.",
                "These are the subset cells.",
                "subset_cells"
            )
            term(
                "subset_candidates",
                "The candidate digits shown on subset-member cells to visualize the subset-defining digit structure. For naked subsets, these are usually the locked subset digits remaining in those cells. For hidden subsets, these are the emphasized support digits inside the support cells.",
                "These are the key candidate digits that define the subset.",
                "subset_candidates"
            )
            term(
                "locked_digits",
                "The digits reserved or emphasized by the subset structure. In naked subsets they are locked into the subset-member cells; in hidden subsets they are the support digits confined to the support set.",
                "These are the digits spoken for by the subset structure.",
                "locked_digits"
            )
            term(
                "restricted_digits",
                "Digits removed or tightened inside support cells by a restriction-style subset, especially hidden subsets.",
                "These are the digits the subset helps remove from the support cells.",
                "restricted_digits"
            )
            term(
                "sweep_cells",
                "The non-member cells in the affected house that feel the consequence of the subset pattern.",
                "These are the other cells affected by the subset.",
                "sweep_cells"
            )
            term(
                "cleanup_digits",
                "The digits removed from affected cells because of the advanced pattern.",
                "These are the digits the pattern lets us remove.",
                "cleanup_digits"
            )

            // --- INTERSECTIONS vocabulary (broad, not technique-specific) ---
            term(
                "overlap",
                "The shared cells where one box and one line intersect. Intersection techniques are narrated from this shared territory outward.",
                "This is the crossroads where the two houses overlap.",
                "overlap / overlap_cells"
            )
            term(
                "source_house",
                "The house that forces the digit into the overlap. In claiming, this is usually the row or column. In pointing, this is usually the box.",
                "This is the house that runs out of room elsewhere and traps the digit in the overlap.",
                "source_house"
            )
            term(
                "cross_house",
                "The other house in the box-line relationship: once the digit is trapped in the overlap, this house loses permission to keep that digit elsewhere.",
                "This is the house whose outside territory gets restricted.",
                "cross_house"
            )
            term(
                "overlap_cells",
                "The cells shared by the source house and the cross house. These are the cells that end up carrying the trapped digit.",
                "These are the overlap cells.",
                "overlap_cells"
            )
            term(
                "source_outside_overlap_cells",
                "The cells in the source house that lie outside the overlap. Setup must explicitly audit these cells to prove the source house has run out of room for the digit elsewhere.",
                "These are the other source-house cells we must check one by one.",
                "source_outside_overlap_cells"
            )
            term(
                "cross_outside_overlap_cells",
                "The cells in the cross house that lie outside the overlap. Once the digit is trapped in the overlap, these cells are no longer allowed to keep that digit.",
                "These are the cells the pattern squeezes in the crossing house.",
                "cross_outside_overlap_cells"
            )
            term(
                "territorial_control",
                "The structural idea that an intersection pattern changes what one house is allowed to do because another house has trapped the digit in the overlap.",
                "The pattern takes control of the shared territory and redraws the legal map around it.",
                "territorial_control"
            )
            term(
                "permission_change",
                "The decisive logical consequence of an intersection: because the digit is confined to the overlap, the crossing house must give it up elsewhere.",
                "That changes which other cells are still allowed to keep the digit.",
                "permission_change"
            )

            // --- Generic pattern-cell vocabulary ---
            term(
                "pattern_cell",
                "A family-neutral pattern-forming cell. Use this when the pattern is not subset-specific, or when the code wants a generic label for cells that belong to the pattern shape.",
                "This cell is part of the pattern.",
                "pattern_cell"
            )
            term(
                "pattern_cells",
                "The set of cells that define the visible shape or structure of a technique pattern.",
                "These are the cells that draw the pattern on the grid.",
                "pattern_cells"
            )
            term(
                "pattern",
                "A broad generic label for pattern-related visual or logical content when a more specific family term is not being used.",
                "This highlighted structure belongs to the pattern.",
                "pattern"
            )
            term(
                "focus_cells",
                "Cells the app treats as the local focal area for the technique application.",
                "These are the cells to keep your eye on.",
                "focus_cells"
            )
            term(
                "target_cells",
                "Cells directly affected by the technique’s resolution or cleanup.",
                "These are the cells the pattern acts on.",
                "target_cells"
            )
            term(
                "peer_cells",
                "Related cells in the same relevant houses that help define or feel the effect of the pattern.",
                "These surrounding cells help shape the logic.",
                "peer_cells"
            )
            term(
                "anchors",
                "Important reference cells that stabilize the pattern description, often the defining cells of the technique.",
                "These are the anchor cells of the pattern.",
                "anchors"
            )

            // --- Bridge / target-relation vocabulary ---
            term(
                "target_relation",
                "The specific way the pattern affects the target, such as cleaning up a cell’s candidates or restricting a house.",
                "This is how the pattern acts on the target.",
                "target_relation"
            )
            term(
                "sweep_relation",
                "The description of which house and which cells are affected by the pattern cleanup or restriction.",
                "This tells us where the cleanup happens.",
                "sweep_relation"
            )
            term(
                "why_this_matters",
                "A plain-language sentence explaining why the trigger pattern matters for the target.",
                "This is why the pattern matters here.",
                "why_this_matters"
            )
            term(
                "cell_outcome",
                "The final candidate-state or value-state outcome for a target cell.",
                "This is what the target cell is left with.",
                "cell_outcome"
            )
            term(
                "house_claim",
                "A final house-based outcome, such as the remaining cells or positions for a digit in a house.",
                "This is the house-level conclusion of the step.",
                "house_claim"
            )

            // --- Atom-0 / intro contract vocabulary ---
            term(
                "advanced_setup_payload",
                "The intro/setup packet for an advanced step. It carries atom-0 truth: target alignment, trigger pattern, trigger explanation, trigger bridge, summary fields, and intro contracts.",
                "This is the setup information for the advanced step.",
                "advanced_setup_payload"
            )
            term(
                "atom0_invariant_contract",
                "The contract saying what atom 0 must contain so the intro stays aligned with the real target and the real trigger.",
                "The intro must stay faithful to the target and the pattern.",
                "atom0_invariant_contract"
            )
            term(
                "target_alignment",
                "The atom-0 requirement that the intro spotlight the same target cell, and the same primary house when the final resolution is house-based.",
                "The intro and final target must match.",
                "target_alignment"
            )
            term(
                "advanced_trigger",
                "The advanced-technique trigger packet inside atom 0: the pattern, the explanation of the pattern, and the bridge from that pattern to the target.",
                "The intro must include the pattern and why it matters.",
                "advanced_trigger"
            )
            term(
                "intro_alignment",
                "The atom-0 rules for how the intro must behave, including route order and not resolving the final answer too early.",
                "The intro has a required structure.",
                "intro_alignment"
            )
            term(
                "required_narrative_route",
                "The ordered route the intro should follow, such as trigger explanation, trigger, bridge, then final-resolution setup.",
                "First explain the pattern, then name it, then connect it to the target.",
                "required_narrative_route"
            )
            term(
                "requires_focus_cell_match",
                "Atom-0 contract flag saying the intro must spotlight the same focus cell as the final resolution.",
                "The intro target cell must match the real target.",
                "requires_focus_cell_match"
            )
            term(
                "requires_primary_house_match_when_house_based",
                "Atom-0 contract flag saying that when the final resolution is house-based, the intro must also keep the same primary house.",
                "The intro house must match the real house-based resolution.",
                "requires_primary_house_match_when_house_based"
            )
            term(
                "advanced_trigger_required",
                "Atom-0 contract flag saying an advanced technique must expose its trigger packet in the intro.",
                "Advanced steps must surface their trigger pattern in the setup.",
                "advanced_trigger_required"
            )
            term(
                "must_not_resolve_final_answer_in_setup",
                "A rule saying the intro may prepare the target but must not place the final answer yet.",
                "This setup clears the way, but it doesn’t place the digit yet.",
                "must_not_resolve_final_answer_in_setup"
            )
            term(
                "requires_single_walkthrough_cta",
                "A rule saying the intro should end with one clean call-to-action inviting the user into the proof.",
                "Would you like me to walk through it?",
                "requires_single_walkthrough_cta"
            )
            term(
                "setup_role",
                "The semantic role of the intro/setup beat. For advanced atom 0, this is the advanced-trigger setup lane.",
                "This beat is the setup for the advanced trigger.",
                "setup_role"
            )
            term(
                "advanced_trigger_setup",
                "The setup role used when atom 0 is introducing an advanced trigger pattern before the detailed proof.",
                "This intro is about noticing the pattern first.",
                "advanced_trigger_setup"
            )

            // --- Intro narration summaries ---
            term(
                "intro_overlay_contract",
                "The atom-0 visual contract saying what should be drawn on screen during the intro.",
                "This controls what gets highlighted in the setup view.",
                "intro_overlay_contract"
            )
            term(
                "intro_narration_contract",
                "The atom-0 speech contract saying what the intro must include and in what order.",
                "This controls the spoken structure of the intro.",
                "intro_narration_contract"
            )
            term(
                "intro_route",
                "The preferred ordered list of intro beats used for setup narration.",
                "This is the intro’s storytelling route.",
                "intro_route"
            )
            term(
                "intro_derived",
                "The prewritten intro summaries derived from the step truth, such as target orientation, technique lens, trigger summary, and bridge summary.",
                "These are the setup summaries the assistant should use.",
                "intro_derived"
            )
            term(
                "target_orientation_summary",
                "The opening line that spotlights the final target for the step.",
                "Let’s zoom in on row X, column Y.",
                "target_orientation_summary"
            )
            term(
                "technique_lens_summary",
                "The early setup line that says why this is a good moment for the technique.",
                "This is a good moment for this technique.",
                "technique_lens_summary"
            )
            term(
                "trigger_explanation_summary",
                "A compact statement saying why the trigger members or trigger houses have the right shape.",
                "Here’s why this pattern is really present.",
                "trigger_explanation_summary"
            )
            term(
                "trigger_member_explanation_rows",
                "Bounded spoken lines that explain, member by member, why the trigger structure qualifies as the pattern.",
                "Here is the first part of the pattern… now look at the next one…",
                "trigger_member_explanation_rows"
            )
            term(
                "trigger_summary",
                "The line that names the pattern once its structure is understood.",
                "So now we have the pattern itself.",
                "trigger_summary"
            )
            term(
                "bridge_summary",
                "The line that says how the pattern affects the target or target house.",
                "This is how the pattern matters for the target.",
                "bridge_summary"
            )
            term(
                "final_resolution_setup_summary",
                "The line that says the intro is only preparing the target, not resolving it yet.",
                "This is a setup move, not the final placement yet.",
                "final_resolution_setup_summary"
            )
            term(
                "honesty_note",
                "The plain-language reminder that the advanced trigger sets up the answer rather than placing it directly.",
                "The pattern clears the way; it does not place the answer by itself.",
                "honesty_note"
            )
            term(
                "cta_kind",
                "The kind of call-to-action the beat should end with, such as asking whether the user wants the proof.",
                "Would you like me to walk through the proof?",
                "cta_kind"
            )

            // --- Intro overlay flags ---
            term(
                "setup_variant",
                "The visual flavor of the intro overlay, such as subset tableau, intersection tableau, or advanced pattern tableau.",
                "This controls what style of setup picture the intro uses.",
                "setup_variant"
            )
            term(
                "show_target_focus",
                "Overlay flag saying the target cell should be spotlighted during intro.",
                "Highlight the target cell.",
                "show_target_focus"
            )
            term(
                "show_subset_house",
                "Overlay flag saying the subset’s house should be highlighted.",
                "Highlight the row, column, or box where the subset lives.",
                "show_subset_house"
            )
            term(
                "show_subset_cells",
                "Overlay flag saying the subset’s defining cells should be highlighted.",
                "Highlight the subset cells.",
                "show_subset_cells"
            )
            term(
                "show_subset_candidates",
                "Overlay flag saying the key candidate digits that define the subset should be drawn on the subset cells.",
                "Show the defining candidate digits on the subset cells.",
                "show_subset_candidates"
            )
            term(
                "show_source_house",
                "Overlay flag saying the source house of an interaction-style pattern should be highlighted.",
                "Highlight the source house.",
                "show_source_house"
            )
            term(
                "show_target_house",
                "Overlay flag saying the target house of an interaction-style pattern should be highlighted.",
                "Highlight the target house.",
                "show_target_house"
            )
            term(
                "show_pattern_cells",
                "Overlay flag saying generic pattern-forming cells should be highlighted.",
                "Highlight the cells that draw the pattern.",
                "show_pattern_cells"
            )
            term(
                "show_sweep_cells",
                "Overlay flag saying the affected cleanup cells should also be highlighted.",
                "Highlight the cells that must let go of those digits.",
                "show_sweep_cells"
            )
            term(
                "show_blocker_network",
                "Overlay flag saying the witness cells and witness houses explaining the pattern should be drawn.",
                "Show the cells and houses that prove the pattern.",
                "show_blocker_network"
            )
            term(
                "show_resolution_collapse",
                "Overlay flag saying the target’s final collapse should be shown visually.",
                "Show the target narrowing to its final answer.",
                "show_resolution_collapse"
            )
            term(
                "show_commit",
                "Overlay flag saying the final committed answer should be shown.",
                "Show the placed digit.",
                "show_commit"
            )

            // --- App flow / telemetry vocabulary ---
            term(
                "tick1_tick2",
                "Tick1 = intent understanding; Tick2 = natural-language reply generation from app-provided facts.",
                "—",
                "Tick1 / Tick2"
            )
            term(
                "grid_hash",
                "Short id for the current grid state used to ensure replies match the right board.",
                "—",
                "gridHash12"
            )
            term(
                "pending",
                "Internal state saying what the system expects next, such as confirm a cell or ask for proof.",
                "—",
                "pending_before / expected_answer_kind"
            )
        })

        // -------------------------
        // FAQ
        // -------------------------
        put("faq", JSONArray().apply {
            faq(
                "What do you mean by low confidence?",
                "It means the scan isn’t fully sure what digit it saw in that cell. I’ll ask you to confirm the uncertain ones."
            )
            faq(
                "What does a red border mean?",
                "Red outline means that cell is still a question mark—I need you to confirm what’s written there."
            )
            faq(
                "What does a cyan border mean?",
                "Cyan outline means I changed that cell during scan correction to restore a clean grid."
            )
            faq(
                "What’s the difference between conflicts and mismatches?",
                "Conflicts break Sudoku rules right now (duplicates). Mismatches mean one of your filled-in answers is wrong compared to the correct solution implied by the clues."
            )
            faq(
                "Can you check if my filled-in answers are correct?",
                "Yes—I’ll compare your filled-in cells against what the puzzle logically requires and list any wrong ones with the correct digits."
            )
            faq(
                "Why are you saying box 7?",
                "I’ll use human names like 'top-left box'. If I ever number boxes, I’ll define the numbering once."
            )
            faq(
                "What are you highlighting on the grid?",
                "I’m showing the exact cells that justify the next move—first the focus area, then the key evidence, then the final placement."
            )
            faq(
                "What do you mean by proof?",
                "It’s the Sudoku reason you can verify: specific cells or houses that block candidates, restrict positions, or define a valid pattern."
            )
            faq(
                "What is a subset member?",
                "It’s one of the cells that belongs to the subset’s defining support set."
            )
            faq(
                "What are subset candidates?",
                "They’re the key candidate digits shown on the subset cells to make the subset structure visible."
            )
            faq(
                "What are sweep cells?",
                "They’re the other affected cells in the relevant house that feel the subset’s cleanup or restriction effect."
            )
        })

        // -------------------------
        // Response templates
        // -------------------------
        put("response_templates", JSONArray().apply {
            template(
                "explain_conflict",
                "A conflict means the grid breaks Sudoku rules right now—like two {d}’s in the same {house}. I can point to exactly where that happens."
            )
            template(
                "explain_mismatch",
                "A mismatch is about your filled-in answers: the clues imply a different digit there. I’ll list the mismatched cells and what the correct digits should be."
            )
            template(
                "row_contents_short",
                "Row {r} currently has {digits_summary}. Want the missing digits too?"
            )
            template(
                "story_intro_spotlight",
                "Alright—let’s set the scene. I’m going to highlight the key area first, then I’ll show the reason step-by-step. Want to try it yourself, or should I guide you?"
            )
            template(
                "story_proof_chain_bridge",
                "Here’s the proof in order. I’ll keep it tight and point to the exact cells or houses that force each conclusion."
            )
            template(
                "story_commit_bridge",
                "Clear so far—ready for the answer?"
            )
            template(
                "overlay_explain_generic",
                "What you see highlighted are the exact Sudoku witnesses—cells or houses that block options, restrict positions, or define the pattern."
            )
        })
    }

    private fun term(id: String, meaning: String, doSay: String, dontSay: String): JSONObject =
        JSONObject().apply {
            put("id", id)
            put("meaning", meaning)
            put("do_say", doSay)
            put("dont_say", dontSay)
        }

    private fun faq(q: String, a: String): JSONObject =
        JSONObject().apply {
            put("q", q)
            put("a", a)
        }

    private fun template(id: String, text: String): JSONObject =
        JSONObject().apply {
            put("id", id)
            put("text", text)
        }
}