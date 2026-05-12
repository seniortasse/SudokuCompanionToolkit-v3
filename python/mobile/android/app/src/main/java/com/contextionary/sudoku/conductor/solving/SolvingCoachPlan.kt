package com.contextionary.sudoku.conductor.solving

/**
 * Canonical, typed, deterministic CoachPlan.
 *
 * ✅ Goals:
 * - stable across runs (same engine step => same CoachPlan shape)
 * - purely derived from engine step JSON (no LLM invention)
 * - supports Phase 3: hint ladder + last-hint awareness
 * - supports Phase 5+: overlay frames tied to hint index
 */
data class CoachPlan(
    val version: String = "v1",
    val gridHash12: String,
    val stepId: String,

    // Human-facing metadata (LLM can verbalize, but not invent)
    val techniqueId: String? = null,
    val title: String,              // e.g., "Naked Single", "Hidden Single", "Next logical move"
    val summary: String,            // short one-liner for the step

    // Target (optional if engine step doesn't provide it)
    val targetCellIndex: Int? = null,   // 0..80
    val targetDigit: Int? = null,       // 1..9

    // Hint ladder
    val hints: List<Hint>,

    // Reveal info (Phase 4/5)
    val reveal: RevealPlan? = null,

    // ✅ Phase 5+: overlay frames aligned to hintIndex (and addressable by frameId)
    val overlayFrames: List<OverlayFrame> = emptyList(),

    // Debug / traceability
    val source: SourceInfo = SourceInfo(),

    // Optional UI focus
    val focusCellIndex: Int? = null
) {
    val hintCount: Int get() = hints.size
    val lastHintIndex: Int get() = (hints.size - 1).coerceAtLeast(0)

    init {
        require(gridHash12.length >= 8) { "CoachPlan.gridHash12 must be >= 8 chars" }
        require(stepId.isNotBlank()) { "CoachPlan.stepId must be non-blank" }
        require(hints.size >= 2) { "CoachPlan must have at least 2 hints (Phase 3 contract)" }

        focusCellIndex?.let { require(it in 0..80) { "CoachPlan.focusCellIndex must be 0..80" } }
        targetCellIndex?.let { require(it in 0..80) { "CoachPlan.targetCellIndex must be 0..80" } }
        targetDigit?.let { require(it in 1..9) { "CoachPlan.targetDigit must be 1..9" } }
    }

    /** Lookup helper for overlay frames by id (used by renderer/effects). */
    fun overlayByFrameId(frameId: String): OverlayFrame? {
        val key = frameId.trim()
        if (key.isEmpty()) return null
        return overlayFrames.firstOrNull { it.frameId == key }
    }
}

data class Hint(
    val index: Int,
    val title: String,
    // Phase 0: text may be empty; evidence is the canonical payload.
    val text: String = "",

    // ✅ V2 ladder identity: ties hint -> overlay frame
    val frameId: String? = null,         // e.g. "h0", "h1"
    val overlayFrameId: String? = null,  // usually same as frameId

    val evidence: HintEvidence = HintEvidence()
)

data class HintEvidence(
    val houses: List<HouseRef> = emptyList(),
    val focusCells: List<Int> = emptyList(),          // 0..80
    val candidateDigit: Int? = null,                  // 1..9
    val eliminateCells: List<Int> = emptyList(),      // 0..80
    val notes: List<String> = emptyList()
)

/** Reference to a Sudoku "house": row/col/box. */
data class HouseRef(
    val kind: HouseKind,
    val index1to9: Int
)

enum class HouseKind { ROW, COL, BOX }

/**
 * Reveal plan:
 * - placementCellIndex + placementDigit are the authoritative placement
 * - explain is optional text used by overlays / deep-dive hooks (deterministic)
 *
 * Back-compat:
 * - rationale kept because older code may still reference it.
 */
data class RevealPlan(
    val placementCellIndex: Int,   // 0..80
    val placementDigit: Int,       // 1..9

    // conductor expects rev.explain
    val explain: String? = null,

    // Back-compat (older code may still use this)
    val rationale: String? = null,

    // ✅ Step 9: deterministic summary suitable for post-reveal CTA / deep dive hooks
    val rationaleSummary: String? = null
)

/**
 * Phase 5+ overlay frames:
 * deterministic UI instruction set, aligned to hint index.
 *
 * frameId is the stable, tool-addressable id (LLM must reference existing ids).
 */
data class OverlayFrame(
    val frameId: String,           // e.g. "h0", "h1", "reveal"
    val hintIndex: Int,            // 0..lastHintIndex, or -1 for non-hint frames (e.g. "reveal")
    val kind: OverlayKind,
    val payload: Map<String, Any?> = emptyMap()
)

enum class OverlayKind {
    NONE,
    HIGHLIGHT_HOUSES,
    HIGHLIGHT_CELLS,
    CANDIDATE_ELIMINATION,
    TARGET_CELL
}

data class SourceInfo(
    val engineSchemaVersion: String? = null,
    val engineStepSha12: String? = null,
    val warnings: List<String> = emptyList()
)