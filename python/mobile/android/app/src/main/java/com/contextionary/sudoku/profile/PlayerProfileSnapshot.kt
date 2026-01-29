package com.contextionary.sudoku.profile

/**
 * Lightweight, read-only snapshot of player info the LLM needs.
 * Add/rename fields here if your LLM coordinator requires more.
 *
 * NOTE: This is safe to expand; defaults preserve backward compatibility.
 */
data class PlayerProfileSnapshot(
    val name: String? = null,
    val locale: String? = null,               // e.g., "en", "fr", "ar"
    val favoriteDifficulty: String? = null,   // "easy" | "medium" | "hard"
    val interests: List<String> = emptyList(),

    // ---- Conversation preferences (new) ----

    /**
     * If true: avoid canned salutations/intros like "Good evening" or "I'm Sudo—ready..."
     * and speak as if the companion already knows the player and is focused on the moment.
     */
    val preferNoScriptedGreetings: Boolean = false,

    /**
     * If true: the first priority after scan is verifying that on-screen grid matches the paper/book.
     * Sudo should optimize Turn 1 for scan verification (confirm one doubtful cell if needed).
     */
    val preferScanMatchFirst: Boolean = true,

    /**
     * If true: assume the source puzzle is intended to be uniquely solvable, unless evidence says otherwise.
     * Useful for guiding “educated guesses” about likely misreads.
     */
    val assumeUniqueSolutionByDefault: Boolean = true,

    /**
     * If true: keep the number of verification turns minimal (confirm exactly one cell at a time).
     */
    val preferMinimalVerificationTurns: Boolean = true,

    // Optional: allows the app to tell Sudo what kind of player this is, without guessing
    val sudokuSkillHint: String? = null, // "beginner" | "intermediate" | "expert" | null

    // ---- Optional extras (safe defaults) ----
    val id: String? = null,
    val avatarUrl: String? = null,
    val level: Int? = null,
    val xp: Long? = null,
    val streak: Int? = null
)