package com.contextionary.sudoku.conductor.solving

import com.contextionary.sudoku.conductor.PendingChoiceId
import com.contextionary.sudoku.conductor.StoryStage

/**
 * SOLVING CTAs only (story-level).
 *
 * Invariant: Google Maps must NOT author spoken scripts.
 * This object may define CTA options and (optionally) UI labels, but must not
 * produce deterministic narration intended for the assistant to read.
 *
 * North Star rule:
 * - each normal solving stage exports exactly one normal CTA family
 * - detours/repair are handled elsewhere, not as co-equal default siblings
 */
object SolvingPromptParts {

    data class StoryCta(
        // Intentionally blank: do NOT inject deterministic scripts into the LLM.
        val prompt: String = "",
        val options: List<String>
    )

    /**
     * Stage-native CTA contract for the 3-beat solving story.
     *
     * SETUP         -> guided proof / walkthrough
     * CONFRONTATION -> lock it in / commit
     * RESOLUTION    -> next step
     *
     * This is the canonical normal-route contract.
     * Clarifications, try-self, and technique detours are not default sibling CTAs here.
     */
    fun forStoryStage(stage: StoryStage): StoryCta {
        val options = when (stage) {
            StoryStage.SETUP ->
                listOf(PendingChoiceId.SHOW_PROOF)

            StoryStage.CONFRONTATION ->
                listOf(PendingChoiceId.LOCK_IT_IN)

            StoryStage.RESOLUTION ->
                listOf(PendingChoiceId.NEXT_STEP)
        }

        return StoryCta(prompt = "", options = options)
    }

    /**
     * Atom-native CTA contract.
     *
     * Transitional rule:
     * atom prompt codes may still help map older step/atom contracts into the
     * canonical North Star CTA family, but they must no longer introduce
     * try-vs-guide branching or any other multi-option setup menu in the normal solve rail.
     *
     * Atom prompt codes (wire):
     * - ASK_NEXT_HINT        -> walkthrough/proof
     * - ASK_USER_TRY         -> legacy alias; still normalize to walkthrough/proof only
     * - ASK_NEXT_STEP        -> proceed to next step
     * - READY_FOR_ANSWER     -> commit/apply if actually available
     */
    fun forAtomPromptCode(
        promptCode: String?,
        allowRevealNow: Boolean = false
    ): StoryCta {

        val code = (promptCode ?: "").trim().uppercase()

        val options = when (code) {
            "ASK_NEXT_HINT" -> listOf(PendingChoiceId.SHOW_PROOF)

            "ASK_USER_TRY" -> listOf(PendingChoiceId.SHOW_PROOF)

            "ASK_NEXT_STEP" -> listOf(PendingChoiceId.NEXT_STEP)

            "READY_FOR_ANSWER", "ASK_READY_FOR_ANSWER" ->
                if (allowRevealNow) listOf(PendingChoiceId.LOCK_IT_IN)
                else listOf(PendingChoiceId.SHOW_PROOF)

            else -> listOf(PendingChoiceId.SHOW_PROOF)
        }

        return StoryCta(prompt = "", options = options)
    }

    fun afterResolution(): StoryCta {
        return forStoryStage(StoryStage.RESOLUTION)
    }

    fun optionsLabel(options: List<String>): String {
        return options.joinToString(" / ") {
            when (it) {
                PendingChoiceId.SHOW_PROOF -> "Show me"
                PendingChoiceId.LOCK_IT_IN -> "Lock it in"
                PendingChoiceId.NEXT_STEP -> "Next step"

                // Explicit detours / repair labels (not normal default rail)
                PendingChoiceId.TRY_SELF -> "I’ll look"
                PendingChoiceId.EXPLAIN_MORE -> "Clarify that"
                PendingChoiceId.ASK_TECHNIQUE -> "How this works"
                PendingChoiceId.RETURN_TO_ROUTE -> "Back to Sudoku"
                else -> it
            }
        }
    }

    fun ctaQuestionShapeForStage(stage: StoryStage?): String =
        when (stage) {
            StoryStage.SETUP -> "ask_for_discovery"
            StoryStage.CONFRONTATION -> "ask_for_next_inference"
            StoryStage.RESOLUTION -> "ask_for_commit_or_apply"
            null -> "ask_for_next_user_move"
        }

    fun ctaSurfaceRulesForStage(stage: StoryStage?): List<String> =
        when (stage) {
            StoryStage.SETUP -> listOf(
                "End with a discovery CTA, not a commit CTA.",
                "Ask for one concrete next observation only.",
                "Good setup CTA examples: identify a digit, identify a cell, inspect the row/column/box.",
                "Do not ask the user to place the answer yet.",
                "Avoid vague endings like 'Ready?' when a more specific ask is possible."
            )

            StoryStage.CONFRONTATION -> listOf(
                "End with a proof-step CTA, not a setup CTA and not a next-step CTA.",
                "Ask for the next local inference only.",
                "Do not ask the user to jump ahead as if the proof is already complete unless the facts say it is.",
                "Good confrontation CTA examples: what does that rule out, which cell is left, what remains possible now."
            )

            StoryStage.RESOLUTION -> listOf(
                "End with a commit/apply CTA or next-step handoff, depending on commit truth.",
                "Do not reopen setup or replay the whole proof.",
                "Good resolution CTA examples: Shall we place it now? Would you like to fill it in? Ready for the next step?"
            )

            null -> listOf(
                "End with one clear user-facing CTA.",
                "Avoid internal route jargon."
            )
        }

    fun detourReturnSurfaceRulesV1(): List<String> = listOf(
        "If the detour answer is complete, close the answer before offering return.",
        "Use a smooth bridge back to the paused move.",
        "Offer a clear choice: return to the move or ask one more question.",
        "Do not say setup, confrontation, resolution, agenda, detour, checkpoint, owner, or packet.",
        "Preferred return wording uses: this move, this step, where we left off, the idea we were following."
    )


    /**
     * Phase 4 — shared stage enforcement block for Tick2 developer prompt.
     *
     * Ownership lives here because these instructions are solving-stage-specific,
     * not generic conversation prompt logic.
     *
     * This is a direct lift of the prior CompanionConversation local block, moved
     * into the solving prompt layer without changing wording.
     */
    fun buildStageEnforcementBlock(stageScope: String): String {
        return buildString {
            appendLine("TURN-SPECIFIC STAGE ENFORCEMENT:")
            appendLine("- stage_scope = $stageScope")
            appendLine("- SETUP_ONLY => sell the solving lens for this step. Direct attention to the target area, name the technique, and explain why this is the right lens now. No answer reveal. No proof sweep.")
            appendLine("- CONFRONTATION_ONLY => use the already-established pattern briefly, state its effect, combine that effect with the target blocker receipts, and reach the logical answer. No pattern reteach. No final placement commit yet.")
            appendLine("- RESOLUTION_ONLY => commit the already-proved answer, recap the whole story compactly, name the technique contribution honestly, and move to the next step. Do not restart setup. Do not replay confrontation in full.")
            appendLine("- Never mix setup, confrontation, and resolution in the same answer.")
            appendLine("- In CONFRONTATION_ONLY, do not stop midway through the proof and do not ask for the next blocker / next hint / more guiding.")
            appendLine("- In CONFRONTATION_ONLY, the correct rhythm is: established trigger reference -> trigger effect -> target blocker receipts -> collapse -> one lock-in CTA.")
            appendLine("- In CONFRONTATION_ONLY, do not re-explain how the pattern forms.")
            appendLine("- In RESOLUTION_ONLY, use present-state language only when commit truth supports it, but when commit truth is active do not fall back to pre-commit language.")
            appendLine("- In RESOLUTION_ONLY, do not pretend the named technique alone placed the digit if the facts show a two-layer finish.")
            appendLine("- In RESOLUTION_ONLY, keep the recap compact and do not reconstruct the full proof body.")
            appendLine("- Avoid generic filler like 'by Sudoku rules' when the provided facts/atoms contain a specific rule, witness, or house/cell proof.")
            appendLine("- If structure is thin, stay brief rather than inventing missing proof steps.")
            appendLine("- CTA must match stage_scope and must stay user-facing:")
            appendLine("  - setup => discovery CTA only; ask for one concrete observation or forced digit/cell, not commit")
            appendLine("  - confrontation => proof-step CTA only; ask for the next local inference, not next step")
            appendLine("  - resolution => commit/apply CTA or next-step handoff, depending on commit truth")
            appendLine("- Do not use internal route words like setup, confrontation, or resolution in the spoken CTA.")
            appendLine("- Prefer specific asks over vague endings like 'Ready?' when a more precise question is available.")
            appendLine("- For SUBSETS: in confrontation, refer back to the established subset briefly, say what it removes or reserves, then explain how the remaining blocker network forces the target. If the finish is two-layer, say that honestly.")
            appendLine("- For INTERSECTIONS: preserve the two-layer story when available: established interaction first, downstream final-answer proof second.")
            appendLine("- For RESOLUTION across all archetypes: the normal rail is commit -> compact recap -> next-step handoff.")
        }.trim()
    }



}