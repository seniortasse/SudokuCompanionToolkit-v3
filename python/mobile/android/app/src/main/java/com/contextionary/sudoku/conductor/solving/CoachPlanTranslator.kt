package com.contextionary.sudoku.conductor.solving

/**
 * Deterministic translation layer:
 *   SolveStepV2 (engine truth) -> CoachPlan (app UX plan)
 *
 * IMPORTANT:
 * - Deterministic only (no interpretation / no LLM).
 * - V2 is the single source of truth.
 */
object CoachPlanTranslator {

    data class TranslateResult(
        val plan: CoachPlan? = null,
        val step: SolveStepV2? = null,
        val warnings: List<String> = emptyList()
    )

    /**
     * Preferred entrypoint from reducer:
     * - Parses envelope
     * - Enforces V2 contract
     * - Returns CoachPlan
     */
    fun translate(
        solveStepJson: String,
        gridHash12Fallback: String? = null
    ): TranslateResult {
        val warnings = mutableListOf<String>()

        val env = SolveStepV2Parser.parseEnvelope(solveStepJson)
            ?: return TranslateResult(
                plan = null,
                step = null,
                warnings = listOf("solve_step_v2: envelope_parse_failed")
            )

        if (!env.ok) {
            val code = env.error?.code ?: "unknown"
            warnings += "solve_step_v2: envelope_not_ok:$code"
            return TranslateResult(plan = null, step = null, warnings = warnings)
        }

        if (env.status != "ok") {
            warnings += "solve_step_v2: status_not_ok:${env.status}"
            return TranslateResult(plan = null, step = null, warnings = warnings)
        }

        val step = env.step
            ?: return TranslateResult(
                plan = null,
                step = null,
                warnings = listOf("solve_step_v2: missing_step_object")
            )

        val plan = fromSolveStepV2(step = step, gridHash12Fallback = gridHash12Fallback, warningsOut = warnings)
        return TranslateResult(plan = plan, step = step, warnings = warnings)
    }

    /**
     * Pure transform: SolveStepV2 -> CoachPlan.
     */
    fun fromSolveStepV2(
        step: SolveStepV2,
        gridHash12Fallback: String? = null,
        warningsOut: MutableList<String>? = null
    ): CoachPlan {
        val warnings = warningsOut ?: mutableListOf()

        val gridHash12 = step.ids.gridHash12Before.takeIf { it.isNotBlank() }
            ?: gridHash12Fallback
            ?: run {
                warnings += "coach_plan: missing_grid_hash12_before_using_fallback=unknown"
                "unknown"
            }

        val stepId = step.ids.stepId.takeIf { it.isNotBlank() } ?: run {
            warnings += "coach_plan: missing_step_id"
            "unknown"
        }

        val techniqueTitle = step.technique.techniqueName.ifBlank { step.technique.techniqueId }
        val techniqueId = step.technique.techniqueId

        // ---- Target / reveal
        val targetCellIndex = step.target.cell?.cellIndex
        val targetDigit = step.target.digit

// Normalize engine kinds (bridge-safe)
        val kind = step.target.kind.trim().lowercase()
        val isPlacement =
            kind == "placement" || kind == "place_digit" || kind == "place-digit" || kind == "placedigit"
        val isElimination =
            kind == "elimination" || kind == "eliminate_digit" || kind == "eliminate-digit" || kind == "eliminatedigit"

        val reveal: RevealPlan? =
            if (isPlacement && targetCellIndex != null && targetDigit != null) {
                RevealPlan(
                    placementCellIndex = targetCellIndex,
                    placementDigit = targetDigit,
                    explain = "Place $targetDigit at cell $targetCellIndex.",
                    rationale = null,
                    rationaleSummary = null
                )
            } else {
                null
            }

// ---- Summary line (deterministic)
        val summary = when {
            isPlacement && step.target.cell != null && step.target.digit != null -> {
                val r = step.target.cell.r
                val c = step.target.cell.c
                "Next move: r${r}c${c} = ${step.target.digit}"
            }
            isElimination && step.target.cell != null && step.target.digit != null -> {
                val r = step.target.cell.r
                val c = step.target.cell.c
                "Eliminate ${step.target.digit} from r${r}c${c}"
            }
            else -> "Apply $techniqueTitle"
        }


        // ---- Hint ladder (V2 identity preserved; must be >= 2)
        // Phase 3: Evidence-first hint ladder.
// If Python provided proof.hint_ladder_v1, use it. Otherwise fall back to minimal stable frames.
        val ladder = mutableListOf<Pair<String, String>>()

        val evidenceHints = runCatching {
            step.proof.hintLadderV1
        }.getOrNull().orEmpty()

        if (evidenceHints.isNotEmpty()) {
            evidenceHints.forEach { h ->
                val fid = h.overlayFrameId?.trim().orEmpty().ifBlank { "h${h.index}" }
                // No scripts: keep text empty; the Driver will speak from SOLVING_STEP_PACKET_V1 evidence.
                ladder += fid to ""
            }
        } else {
            // Minimal deterministic overlay ids; still no scripts.
            ladder += "FRAME_TARGET_CELL_GREEN" to ""
            ladder += "FRAME_HOUSE_GREY_DIGITS" to ""
        }

// Build CoachPlan hints preserving overlay linkage
        val hints: List<Hint> = ladder.mapIndexed { idx, (frameId, text) ->
            Hint(
                index = idx,
                title = if (idx == 0) "Spot the pattern" else "Apply the rule",
                text = text,
                frameId = frameId,
                overlayFrameId = frameId, // convention: ladder frame_id matches overlay frame_id
                evidence = HintEvidence()
            )
        }

        // ---- Overlay frames (aligned to hint indices + reveal)
        //val overlays = buildOverlayFrames(step = step, hintCount = hints.size)
        val overlays = buildOverlayFrames(step = step, hints = hints)

        return CoachPlan(
            version = "v1",
            gridHash12 = gridHash12,
            stepId = stepId,
            techniqueId = techniqueId,
            title = techniqueTitle,
            summary = summary,
            targetCellIndex = targetCellIndex,
            targetDigit = targetDigit,
            hints = hints,
            reveal = reveal,
            overlayFrames = overlays,
            source = SourceInfo(
                engineSchemaVersion = step.schemaVersion,
                engineStepSha12 = step.ids.stepId.take(12),
                warnings = warnings.toList()
            ),
            focusCellIndex = targetCellIndex
        )
    }

    private fun buildOverlayFrames(step: SolveStepV2, hints: List<Hint>): List<OverlayFrame> {
        val engineFrames = step.overlayFrames
        val byId = engineFrames.associateBy { it.frameId }

        // If engine frames are missing, fall back to minimal deterministic frames
        if (engineFrames.isEmpty()) {
            val frames = mutableListOf<OverlayFrame>()
            val cell = step.target.cell

            if (cell != null) {
                frames += OverlayFrame(
                    frameId = "h0",
                    hintIndex = 0,
                    kind = OverlayKind.HIGHLIGHT_CELLS,
                    payload = mapOf("cells" to listOf(cell.cellIndex), "role" to "focus")
                )
            }

            frames += OverlayFrame(
                frameId = "reveal",
                hintIndex = -1,
                kind = OverlayKind.HIGHLIGHT_CELLS,
                payload = mapOf(
                    "placements" to step.proof.placements.map { p -> mapOf("cellIndex" to p.cellIndex, "digit" to p.digit) },
                    "eliminations" to step.proof.eliminations.map { e -> mapOf("cellIndex" to e.cellIndex, "digit" to e.digit) },
                    "role" to "reveal"
                )
            )

            return frames
        }

        val out = mutableListOf<OverlayFrame>()

        // 1) Hint frames: use each hint’s overlayFrameId (or fallback to "h{idx}")
        for (h in hints) {
            val wantId = (h.overlayFrameId ?: h.frameId ?: "h${h.index}").trim()
            val f = byId[wantId] ?: byId["h${h.index}"] ?: continue

            out += OverlayFrame(
                frameId = f.frameId,
                hintIndex = h.index,
                kind = OverlayKind.HIGHLIGHT_CELLS,
                payload = mapOf(
                    "frame_id" to f.frameId,
                    "style" to f.style,
                    "highlights" to f.highlights.map { hi ->
                        mapOf(
                            "kind" to hi.kind,
                            "cellIndex" to hi.cellIndex,
                            "role" to hi.role,
                            "digit" to hi.digit,
                            "house" to hi.house?.let { house -> mapOf("type" to house.type, "index1to9" to house.index1to9) }
                        )
                    }
                )
            )
        }

        // 2) Reveal frame (if present)
        byId["reveal"]?.let { reveal ->
            out += OverlayFrame(
                frameId = reveal.frameId,
                hintIndex = -1,
                kind = OverlayKind.HIGHLIGHT_CELLS,
                payload = mapOf(
                    "frame_id" to reveal.frameId,
                    "style" to reveal.style,
                    "highlights" to reveal.highlights.map { hi ->
                        mapOf(
                            "kind" to hi.kind,
                            "cellIndex" to hi.cellIndex,
                            "role" to hi.role,
                            "digit" to hi.digit,
                            "house" to hi.house?.let { house -> mapOf("type" to house.type, "index1to9" to house.index1to9) }
                        )
                    }
                )
            )
        }

        return out
    }
}