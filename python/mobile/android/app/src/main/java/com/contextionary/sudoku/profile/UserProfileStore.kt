package com.contextionary.sudoku.profile

import android.content.Context



private fun String.trimOrNullTop(): String? = this.trim().takeIf { it.isNotEmpty() }
private fun String.nullIfBlankTop(): String? = this.takeIf { it.isNotBlank() }?.trim()

/**
 * Lightweight, app-local user profile that Sudo can learn over time.
 *
 * - Safe to call from UI threads (SharedPreferences are fast enough for this size).
 * - Fields are optional; only persist what we know.
 * - Interests are de-duplicated via a Set.
 */
data class UserProfile(
    var name: String? = null,
    var preferredLanguage: String? = null,   // e.g., "en", "fr", "ar"
    var favoriteDifficulty: String? = null,  // "easy" | "medium" | "hard"
    var interests: MutableSet<String> = mutableSetOf(),

    // --- Conversation preferences (new) ---
    var preferNoScriptedGreetings: Boolean = false,
    var preferScanMatchFirst: Boolean = true,
    var assumeUniqueSolutionByDefault: Boolean = true,
    var preferMinimalVerificationTurns: Boolean = true,
    var sudokuSkillHint: String? = null // "beginner" | "intermediate" | "expert" | null
) {
    /** Non-empty, trimmed view of interests (for UI/logs). */
    fun interestsList(): List<String> = interests
        .mapNotNull { it.trim().takeIf { s -> s.isNotEmpty() } }
        .distinct()

    /** Map to the lightweight snapshot the LLM coordinator consumes. */
    fun toSnapshot(): PlayerProfileSnapshot = PlayerProfileSnapshot(
        name = name?.trimOrNullTop(),
        locale = preferredLanguage?.trimOrNullTop(),
        favoriteDifficulty = favoriteDifficulty?.trimOrNullTop(),
        interests = interestsList(),

        preferNoScriptedGreetings = preferNoScriptedGreetings,
        preferScanMatchFirst = preferScanMatchFirst,
        assumeUniqueSolutionByDefault = assumeUniqueSolutionByDefault,
        preferMinimalVerificationTurns = preferMinimalVerificationTurns,
        sudokuSkillHint = sudokuSkillHint?.trimOrNullTop()
    )
}

/**
 * Simple persistence layer for [UserProfile] backed by SharedPreferences.
 */
object UserProfileStore {
    private const val PREF = "sudo_profile"

    // Keys (existing)
    private const val K_NAME  = "name"
    private const val K_LANG  = "lang"
    private const val K_DIFF  = "diff"
    private const val K_INTR  = "interests_csv"

    // Keys (new prefs)
    private const val K_PREF_NO_SCRIPT = "pref_no_scripted_greetings"
    private const val K_PREF_SCAN_FIRST = "pref_scan_match_first"
    private const val K_PREF_ASSUME_UNIQUE = "pref_assume_unique_solution"
    private const val K_PREF_MIN_TURNS = "pref_min_verification_turns"
    private const val K_SKILL_HINT = "sudoku_skill_hint"

    // --- Public API ------------------------------------------------------------

    /** Load the current profile (returns an object even if nothing was saved yet). */
    fun load(ctx: Context): UserProfile {
        val sp = ctx.getSharedPreferences(PREF, Context.MODE_PRIVATE)
        val name = sp.getString(K_NAME, null)
        val lang = sp.getString(K_LANG, null)
        val diff = sp.getString(K_DIFF, null)
        val interestsCsv = sp.getString(K_INTR, "") ?: ""

        // New prefs (with safe defaults)
        val noScript = sp.getBoolean(K_PREF_NO_SCRIPT, false)
        val scanFirst = sp.getBoolean(K_PREF_SCAN_FIRST, true)
        val assumeUnique = sp.getBoolean(K_PREF_ASSUME_UNIQUE, true)
        val minTurns = sp.getBoolean(K_PREF_MIN_TURNS, true)
        val skillHint = sp.getString(K_SKILL_HINT, null)

        return UserProfile(
            name = name?.nullIfBlankTop(),
            preferredLanguage = lang?.nullIfBlankTop(),
            favoriteDifficulty = diff?.nullIfBlankTop(),
            interests = csvToSet(interestsCsv),

            preferNoScriptedGreetings = noScript,
            preferScanMatchFirst = scanFirst,
            assumeUniqueSolutionByDefault = assumeUnique,
            preferMinimalVerificationTurns = minTurns,
            sudokuSkillHint = skillHint?.nullIfBlankTop()
        )
    }

    /** Save (overwrites all fields). Use [mergeAndSave] if you want to update incrementally. */
    fun save(ctx: Context, p: UserProfile) {
        val sp = ctx.getSharedPreferences(PREF, Context.MODE_PRIVATE)
        sp.edit()
            .putString(K_NAME, p.name?.trimOrNullTop())
            .putString(K_LANG, p.preferredLanguage?.trimOrNullTop())
            .putString(K_DIFF, p.favoriteDifficulty?.trimOrNullTop())
            .putString(K_INTR, setToCsv(p.interests))

            // New prefs
            .putBoolean(K_PREF_NO_SCRIPT, p.preferNoScriptedGreetings)
            .putBoolean(K_PREF_SCAN_FIRST, p.preferScanMatchFirst)
            .putBoolean(K_PREF_ASSUME_UNIQUE, p.assumeUniqueSolutionByDefault)
            .putBoolean(K_PREF_MIN_TURNS, p.preferMinimalVerificationTurns)
            .putString(K_SKILL_HINT, p.sudokuSkillHint?.trimOrNullTop())

            .apply()
    }

    /**
     * Merge non-null / non-blank fields from [delta] into [into], then persist.
     * - Null/blank fields in [delta] are ignored (won’t clobber existing data).
     * - Interests are unioned and de-duplicated.
     * - Boolean prefs: delta values always override (explicit user preference).
     */
    fun mergeAndSave(ctx: Context, into: UserProfile, delta: UserProfile) {
        delta.name?.trimOrNullTop()?.let { into.name = it }
        delta.preferredLanguage?.trimOrNullTop()?.let { into.preferredLanguage = it }
        delta.favoriteDifficulty?.trimOrNullTop()?.let { into.favoriteDifficulty = it }
        if (delta.interests.isNotEmpty()) {
            into.interests.addAll(delta.interests.mapNotNull { it.trimOrNullTop() })
        }

        // New prefs: apply directly
        into.preferNoScriptedGreetings = delta.preferNoScriptedGreetings
        into.preferScanMatchFirst = delta.preferScanMatchFirst
        into.assumeUniqueSolutionByDefault = delta.assumeUniqueSolutionByDefault
        into.preferMinimalVerificationTurns = delta.preferMinimalVerificationTurns
        delta.sudokuSkillHint?.trimOrNullTop()?.let { into.sudokuSkillHint = it }

        save(ctx, into)
    }

    /** Clear all saved profile data. */
    fun clear(ctx: Context) {
        val sp = ctx.getSharedPreferences(PREF, Context.MODE_PRIVATE)
        sp.edit().clear().apply()
    }

    // --- Convenience helpers ---------------------------------------------------

    /** Add one interest (trimmed, ignored if blank), then save. */
    fun addInterest(ctx: Context, interest: String) {
        val p = load(ctx)
        interest.trimOrNullTop()?.let { p.interests.add(it) }
        save(ctx, p)
    }

    /** Set/override the player name (trimmed, blank → remove). */
    fun setName(ctx: Context, name: String?) {
        val p = load(ctx)
        p.name = name?.trimOrNullTop()
        save(ctx, p)
    }

    /** Set preferred language like "en", "fr", "ar" (trimmed, blank → remove). */
    fun setPreferredLanguage(ctx: Context, lang: String?) {
        val p = load(ctx)
        p.preferredLanguage = lang?.trimOrNullTop()
        save(ctx, p)
    }

    /** Set favorite difficulty ("easy"|"medium"|"hard", blank → remove). */
    fun setFavoriteDifficulty(ctx: Context, diff: String?) {
        val p = load(ctx)
        p.favoriteDifficulty = diff?.trimOrNullTop()
        save(ctx, p)
    }

    // New setters (optional but handy)
    fun setPreferNoScriptedGreetings(ctx: Context, enabled: Boolean) {
        val p = load(ctx)
        p.preferNoScriptedGreetings = enabled
        save(ctx, p)
    }

    fun setPreferScanMatchFirst(ctx: Context, enabled: Boolean) {
        val p = load(ctx)
        p.preferScanMatchFirst = enabled
        save(ctx, p)
    }

    fun setAssumeUniqueSolutionByDefault(ctx: Context, enabled: Boolean) {
        val p = load(ctx)
        p.assumeUniqueSolutionByDefault = enabled
        save(ctx, p)
    }

    fun setPreferMinimalVerificationTurns(ctx: Context, enabled: Boolean) {
        val p = load(ctx)
        p.preferMinimalVerificationTurns = enabled
        save(ctx, p)
    }

    fun setSudokuSkillHint(ctx: Context, hint: String?) {
        val p = load(ctx)
        p.sudokuSkillHint = hint?.trimOrNullTop()
        save(ctx, p)
    }

    // --- Private utils ---------------------------------------------------------

    private fun String.trimOrNullTop(): String? = this.trim().takeIf { it.isNotEmpty() }
    private fun String.nullIfBlankTop(): String? = this.takeIf { it.isNotBlank() }?.trim()

    private fun csvToSet(csv: String): MutableSet<String> =
        csv.split(',')
            .mapNotNull { it.trim().takeIf { s -> s.isNotEmpty() } }
            .toMutableSet()

    private fun setToCsv(set: Set<String>): String =
        set.asSequence()
            .mapNotNull { it.trim().takeIf { s -> s.isNotEmpty() } }
            .distinct()
            .joinToString(",")
}