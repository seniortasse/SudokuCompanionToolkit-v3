package com.contextionary.sudoku.conversation

/**
 * Builds developerPrompt strings containing explicit marker blocks:
 *  - BEGIN_CAPTURE_ORIGIN / END_CAPTURE_ORIGIN (optional)
 *  - BEGIN_GRID_CONTEXT / END_GRID_CONTEXT (required in GRID + grid-session free talk)
 *  - BEGIN_CANONICAL_HISTORY / END_CANONICAL_HISTORY (optional)
 *  - BEGIN_DEVELOPER_NOTES / END_DEVELOPER_NOTES (optional)
 *
 * These markers are consumed by RealSudokuLLMClient.
 *
 * Safety:
 * - We sanitize content so user/app text cannot accidentally contain marker strings
 *   that would "break out" of a block.
 * - We also indent multiline history content so parsers don’t interpret inner lines
 *   as new "role: ..." records.
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

    /**
     * GRID mode prompt: always includes grid grounding; optionally includes canonical history
     * (same format as FREE_TALK) and optional notes.
     *
     * canonicalHistory: role -> text (user/assistant)
     */
    fun composeForGridMode(
        gridContext: String,
        captureOrigin: String = "",
        canonicalHistory: List<Pair<String, String>> = emptyList(),
        extraDeveloperNotes: String? = null
    ): String {
        val grid = sanitizeBlock(normalize(gridContext)).trim()
        require(grid.isNotEmpty()) { "gridContext must be non-empty in GRID mode" }

        val origin = sanitizeBlock(normalize(captureOrigin)).trim()
        val notes = sanitizeBlock(normalize(extraDeveloperNotes ?: "")).trim()

        val historyBlock = buildCanonicalHistoryBlock(canonicalHistory)

        return buildString {
            if (origin.isNotEmpty()) {
                appendLine(BEGIN_CAPTURE_ORIGIN)
                appendLine(origin)
                appendLine(END_CAPTURE_ORIGIN)
                appendLine()
            }

            appendLine(BEGIN_GRID_CONTEXT)
            appendLine(grid)
            appendLine(END_GRID_CONTEXT)
            appendLine()

            if (historyBlock.isNotEmpty()) {
                appendLine(historyBlock)
                appendLine()
            }

            if (notes.isNotEmpty()) {
                appendLine(BEGIN_DEVELOPER_NOTES)
                appendLine(notes)
                appendLine(END_DEVELOPER_NOTES)
            }
        }.trim()
    }

    fun composeForFreeTalkInGridSession(
        gridContext: String,
        captureOrigin: String = "",
        canonicalHistory: List<Pair<String, String>> = emptyList(),
        extraDeveloperNotes: String? = null
    ): String {
        val grid = sanitizeBlock(normalize(gridContext)).trim()
        require(grid.isNotEmpty()) { "gridContext must be non-empty in grid-session free talk" }

        val origin = sanitizeBlock(normalize(captureOrigin)).trim()
        val notes = sanitizeBlock(normalize(extraDeveloperNotes ?: "")).trim()

        val historyBlock = buildCanonicalHistoryBlock(canonicalHistory)

        return buildString {
            if (origin.isNotEmpty()) {
                appendLine(BEGIN_CAPTURE_ORIGIN)
                appendLine(origin)
                appendLine(END_CAPTURE_ORIGIN)
                appendLine()
            }

            appendLine(BEGIN_GRID_CONTEXT)
            appendLine(grid)
            appendLine(END_GRID_CONTEXT)
            appendLine()

            if (historyBlock.isNotEmpty()) {
                appendLine(historyBlock)
                appendLine()
            }

            if (notes.isNotEmpty()) {
                appendLine(BEGIN_DEVELOPER_NOTES)
                appendLine(notes)
                appendLine(END_DEVELOPER_NOTES)
            }
        }.trim()
    }

    private fun buildCanonicalHistoryBlock(
        canonicalHistory: List<Pair<String, String>>
    ): String {
        if (canonicalHistory.isEmpty()) return ""

        // Keep tail (most recent), then enforce a total character budget.
        val tail = if (canonicalHistory.size > HISTORY_MAX_ITEMS)
            canonicalHistory.takeLast(HISTORY_MAX_ITEMS)
        else
            canonicalHistory

        fun capItem(s: String, maxChars: Int): String {
            val n = normalize(s).trim()
            if (n.length <= maxChars) return n
            return n.substring(0, maxChars).trimEnd() + "…"
        }

        fun safeLine(s: String): String {
            // Prevent accidental marker collisions inside history content.
            return sanitizeBlock(s).trimEnd()
        }

        val lines = ArrayList<String>(tail.size + 2)
        lines.add(BEGIN_CANONICAL_HISTORY)

        var usedChars = 0

        for ((roleRaw, textRaw) in tail) {
            val role = roleRaw.lowercase().trim()
            if (role != "user" && role != "assistant") continue

            val capped = capItem(textRaw, HISTORY_MAX_CHARS_PER_ITEM)
            if (capped.isBlank()) continue

            // Indent continuation lines so parsers do NOT treat them as new "role:" records.
            val safe = safeLine(capped).replace("\n", "\n  ")

            val line = "$role: $safe"
            usedChars += line.length
            if (usedChars > HISTORY_MAX_CHARS_TOTAL) break

            lines.add(line)
        }

        lines.add(END_CANONICAL_HISTORY)

        // If everything was filtered out, don't emit an empty history block.
        return if (lines.size <= 2) "" else lines.joinToString("\n")
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