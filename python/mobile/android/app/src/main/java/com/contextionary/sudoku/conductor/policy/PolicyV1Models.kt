package com.contextionary.sudoku.conductor.policy

import com.contextionary.sudoku.conductor.GridPhase
import org.json.JSONArray
import org.json.JSONObject

/**
 * Frozen v1 models for the new architecture:
 * Tick 1: IntentEnvelopeV1 (NLU only, multi-intent)
 * Tick 2: ReplyRequestV1 (NLG only; spoken reply always LLM-generated)
 *
 * - Strict-ish JSON parsing + validation for IntentEnvelopeV1
 * - Easy JSON serialization for ReplyRequestV1 + Fact bundles
 */

// -----------------------------------------------------------------------------
// Common helpers
// -----------------------------------------------------------------------------

private val CELL_RE = Regex("^r[1-9]c[1-9]$")

private fun String?.isValidCell(): Boolean = this != null && CELL_RE.matches(this)

// -----------------------------------------------------------------------------
// ✅ Standardized payload refs (V1+)
// These are used inside FactBundle payloads to make Tick2 “dumb + reliable”.
// -----------------------------------------------------------------------------

/** Returns 0..80 for "r1c1".."r9c9", or null if invalid. */
fun cellIndexOf(cell: String?): Int? {
    val s = cell?.trim() ?: return null
    if (!CELL_RE.matches(s)) return null
    // "rNcM"
    val r = s[1].digitToIntOrNull() ?: return null
    val c = s[3].digitToIntOrNull() ?: return null
    return ((r - 1) * 9 + (c - 1)).takeIf { it in 0..80 }
}

/** {"cell":"r4c2","index":29} */
fun cellRefV1(cell: String): JSONObject = JSONObject().apply {
    put("cell", cell)
    put("index", cellIndexOf(cell) ?: JSONObject.NULL)
}

/** {"kind":"ROW|COL|BOX","index":7} (1..9) */
fun houseRefV1(kind: RegionKindV1, index: Int): JSONObject = JSONObject().apply {
    put("kind", kind.name)
    put("index", index)
}

/** {"digit":6} */
fun digitRefV1(digit: Int): JSONObject = JSONObject().apply {
    put("digit", digit.coerceIn(0, 9))
}

/** JSONArray of strings */
fun jsonArrayOfStrings(xs: List<String>): JSONArray =
    JSONArray().apply { xs.forEach { put(it) } }

/** JSONArray of JSONObjects */
fun jsonArrayOfObjects(xs: List<JSONObject>): JSONArray =
    JSONArray().apply { xs.forEach { put(it) } }

private fun clamp01(x: Double): Double = when {
    x < 0.0 -> 0.0
    x > 1.0 -> 1.0
    else -> x
}

private fun JSONObject.optStringOrNull(k: String): String? =
    if (has(k) && !isNull(k)) optString(k, null) else null

private fun JSONObject.optIntOrNull(k: String): Int? =
    if (has(k) && !isNull(k)) optInt(k) else null

private fun JSONObject.optDoubleOrNull(k: String): Double? =
    if (has(k) && !isNull(k)) optDouble(k) else null

private fun JSONObject.optJSONObjectOrNull(k: String): JSONObject? =
    if (has(k) && !isNull(k)) optJSONObject(k) else null

private fun JSONObject.optJSONArrayOrNull(k: String): JSONArray? =
    if (has(k) && !isNull(k)) optJSONArray(k) else null

private fun JSONArray.toStringList(maxItems: Int): List<String> {
    val out = ArrayList<String>(minOf(length(), maxItems))
    val n = minOf(length(), maxItems)
    for (i in 0 until n) {
        val v = optString(i, null) ?: continue
        out.add(v)
    }
    return out
}

private fun JSONObject.optStringListOrEmpty(k: String, maxItems: Int = 100): List<String> =
    optJSONArrayOrNull(k)?.toStringList(maxItems) ?: emptyList()

private fun JSONObject.putStringListIfNotEmpty(k: String, xs: List<String>) {
    if (xs.isNotEmpty()) put(k, jsonArrayOfStrings(xs))
}

private fun JSONObject.putEnumIfNotNull(k: String, e: Enum<*>?) {
    if (e != null) put(k, e.name)
}

private inline fun <reified T : Enum<T>> enumValueOrNull(raw: String?): T? {
    val s = raw?.trim()?.takeIf { it.isNotEmpty() } ?: return null
    return enumValues<T>().firstOrNull { it.name.equals(s, ignoreCase = true) }
}

private fun <T> JSONObject.optObjectListOrEmpty(
    k: String,
    maxItems: Int = 100,
    parse: (JSONObject) -> T?
): List<T> {
    val arr = optJSONArrayOrNull(k) ?: return emptyList()
    val out = ArrayList<T>(minOf(arr.length(), maxItems))
    val n = minOf(arr.length(), maxItems)
    for (i in 0 until n) {
        val obj = arr.optJSONObject(i) ?: continue
        parse(obj)?.let { out.add(it) }
    }
    return out
}

private fun <T> JSONObject.putObjectListIfNotEmpty(
    k: String,
    xs: List<T>,
    toJson: (T) -> JSONObject
) {
    if (xs.isEmpty()) return
    put(k, JSONArray().apply { xs.forEach { put(toJson(it)) } })
}

// -----------------------------------------------------------------------------
// Profile tallies + transcript (v1)
// -----------------------------------------------------------------------------

private fun capWords(s: String?, maxWords: Int): String? {
    val t = s?.trim()?.takeIf { it.isNotEmpty() } ?: return null
    val parts = t.split(Regex("\\s+"))
    if (parts.size <= maxWords) return t
    return parts.take(maxWords).joinToString(" ")
}

private fun JSONObject.optStringOrNullAny(vararg keys: String): String? {
    for (k in keys) {
        val v = optStringOrNull(k)
        if (v != null) return v
    }
    return null
}

// IMPORTANT: for “telemetry JSON” we want empty-string fields to exist.
// This keeps null internal, but allows a “full JSON” with "" defaults.
private fun String?.orEmptyField(): String = this ?: ""

data class UserTallyV1(
    val name: String? = null,                  // cap ~10w
    val age: String? = null,                   // cap ~6w
    val facts: String? = null,                 // cap ~60w
    val sudokuLevel: String? = null,           // cap ~18w
    val thinkingProcess: String? = null,       // cap ~45w
    val dislikes: String? = null,              // cap ~30w
    val preferences: String? = null,           // cap ~30w
    val personality: String? = null,           // cap ~30w
    val firstSpeech: String? = null            // cap ~30w (immutable by store)
) {
    fun merge(delta: UserTallyV1): UserTallyV1 = UserTallyV1(
        name = delta.name ?: name,
        age = delta.age ?: age,
        facts = delta.facts ?: facts,
        sudokuLevel = delta.sudokuLevel ?: sudokuLevel,
        thinkingProcess = delta.thinkingProcess ?: thinkingProcess,
        dislikes = delta.dislikes ?: dislikes,
        preferences = delta.preferences ?: preferences,
        personality = delta.personality ?: personality,
        firstSpeech = delta.firstSpeech ?: firstSpeech
    ).capped()

    private fun capped(): UserTallyV1 = UserTallyV1(
        name = capWords(name, 30),
        age = capWords(age, 30),
        facts = capWords(facts, 100),
        sudokuLevel = capWords(sudokuLevel, 30),
        thinkingProcess = capWords(thinkingProcess, 100),
        dislikes = capWords(dislikes, 100),
        preferences = capWords(preferences, 100),
        personality = capWords(personality, 100),
        firstSpeech = capWords(firstSpeech, 200)
    )

    fun toJson(): JSONObject = JSONObject().apply {
        name?.let { put("name", it) }
        age?.let { put("age", it) }
        facts?.let { put("facts", it) }
        sudokuLevel?.let { put("sudoku_level", it) }
        thinkingProcess?.let { put("thinking_process", it) }
        dislikes?.let { put("dislikes", it) }
        preferences?.let { put("preferences", it) }
        personality?.let { put("personality", it) }
        firstSpeech?.let { put("first_speech", it) }
    }

    fun toFullJson(): JSONObject = JSONObject().apply {
        put("name", name.orEmptyField())
        put("age", age.orEmptyField())
        put("facts", facts.orEmptyField())
        put("sudoku_level", sudokuLevel.orEmptyField())
        put("thinking_process", thinkingProcess.orEmptyField())
        put("dislikes", dislikes.orEmptyField())
        put("preferences", preferences.orEmptyField())
        put("personality", personality.orEmptyField())
        put("first_speech", firstSpeech.orEmptyField())
    }

    companion object {
        fun parse(o: JSONObject?): UserTallyV1 {
            if (o == null) return UserTallyV1()

            val name = o.optStringOrNullAny("name")
            val age = o.optStringOrNullAny("age")
            val facts = o.optStringOrNullAny("facts")
            val sudokuLevel = o.optStringOrNullAny("sudoku_level", "sudokuLevel")
            val thinkingProcess = o.optStringOrNullAny("thinking_process", "thinkingProcess")
            val dislikes = o.optStringOrNullAny("dislikes")
            val preferences = o.optStringOrNullAny("preferences")
            val personality = o.optStringOrNullAny("personality")
            val firstSpeech = o.optStringOrNullAny("first_speech", "firstSpeech")

            return UserTallyV1(
                name = name,
                age = age,
                facts = facts,
                sudokuLevel = sudokuLevel,
                thinkingProcess = thinkingProcess,
                dislikes = dislikes,
                preferences = preferences,
                personality = personality,
                firstSpeech = firstSpeech
            ).capped()
        }
    }
}

data class AssistantTallyV1(
    val name: String? = null,              // cap ~10w
    val age: String? = null,               // cap ~6w
    val about: String? = null,             // cap ~80w
    val dislikes: String? = null,          // cap ~30w
    val preferences: String? = null,       // cap ~30w
    val personality: String? = null,       // cap ~30w
    val firstSpeech: String? = null        // cap ~30w (immutable by store)
) {
    fun merge(delta: AssistantTallyV1): AssistantTallyV1 = AssistantTallyV1(
        name = delta.name ?: name,
        age = delta.age ?: age,
        about = delta.about ?: about,
        dislikes = delta.dislikes ?: dislikes,
        preferences = delta.preferences ?: preferences,
        personality = delta.personality ?: personality,
        firstSpeech = delta.firstSpeech ?: firstSpeech
    ).capped()

    private fun capped(): AssistantTallyV1 = AssistantTallyV1(
        name = capWords(name, 30),
        age = capWords(age, 30),
        about = capWords(about, 100),
        dislikes = capWords(dislikes, 30),
        preferences = capWords(preferences, 100),
        personality = capWords(personality, 100),
        firstSpeech = capWords(firstSpeech, 200)
    )

    fun toJson(): JSONObject = JSONObject().apply {
        name?.let { put("name", it) }
        age?.let { put("age", it) }
        about?.let { put("about", it) }
        dislikes?.let { put("dislikes", it) }
        preferences?.let { put("preferences", it) }
        personality?.let { put("personality", it) }
        firstSpeech?.let { put("first_speech", it) }
    }

    fun toFullJson(): JSONObject = JSONObject().apply {
        put("name", name.orEmptyField())
        put("age", age.orEmptyField())
        put("about", about.orEmptyField())
        put("dislikes", dislikes.orEmptyField())
        put("preferences", preferences.orEmptyField())
        put("personality", personality.orEmptyField())
        put("first_speech", firstSpeech.orEmptyField())
    }

    companion object {
        fun defaults(): AssistantTallyV1 = AssistantTallyV1(
            name = "Sudo",
            age = "timeless",
            about = "PERSONA — “Witty Sidekick”. Funny Sudoku coach: clear steps, specific feedback, motivating accountability. Light wit level 3. Tiny affectionate snark only; never when user is frustrated. No emojis.",
            dislikes = "Guessing, vague confirmations, and app-internal jargon (e.g., unresolved/low-confidence as-is).",
            preferences = "Speak like a human Sudoku player: say 'row 7 column 2' (not r7c2) and 'top-left box' (not box 1). One-question max. Short crisp turns. Always anchor statements to the current agenda step (scan-match → rule-validity → your-answers-correct → solve).",
            personality = "Witty, calm, coach-like, truth-first."
        ).capped()

        fun parse(o: JSONObject?): AssistantTallyV1 {
            if (o == null) return AssistantTallyV1()
            return AssistantTallyV1(
                name = o.optStringOrNull("name"),
                age = o.optStringOrNull("age"),
                about = o.optStringOrNull("about"),
                dislikes = o.optStringOrNull("dislikes"),
                preferences = o.optStringOrNull("preferences"),
                personality = o.optStringOrNull("personality"),
                firstSpeech = o.optStringOrNull("first_speech")
            ).capped()
        }
    }
}

data class TranscriptTurnV1(
    val turnId: Long,
    val user: String,
    val assistant: String
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("turn_id", turnId)
        put("user", user)
        put("assistant", assistant)
    }

    companion object {
        fun jsonArray(items: List<TranscriptTurnV1>): JSONArray =
            JSONArray().apply { items.forEach { put(it.toJson()) } }
    }
}

// -----------------------------------------------------------------------------
// Relationship memory foundation (Wave 1)
// -----------------------------------------------------------------------------
// NOTE:
// - Foundation models only; no behavior wiring yet.
// - Keep some fields as free text / free lists for flexibility.
// - JSON should stay compact: omit nulls and empty collections where reasonable.

enum class RelationshipToneV1 {
    CALM_COMPANION,
    ENERGETIC_TEAMMATE,
    GENTLE_COACH,
    SHARP_SIDEKICK,
    WARM_GUIDE,
    FRIENDLY_PARTNER
}

enum class FamiliarityPreferenceV1 {
    FORMAL,
    WARM,
    FRIENDLY,
    VERY_FAMILIAR
}

enum class PacePreferenceV1 {
    COMPACT,
    BALANCED,
    DETAILED
}

enum class HumorPreferenceV1 {
    NONE,
    DRY,
    LIGHT,
    PLAYFUL
}

enum class ProofPreferenceV1 {
    MINIMAL,
    MODERATE,
    EXPLICIT
}

enum class ExplanationStyleV1 {
    CONCRETE_STEPWISE,
    VISUAL,
    LOGIC_FIRST,
    ANALOGY_SUPPORTED,
    INTUITIVE,
    STORYLIKE
}

enum class TechniqueFamiliarityV1 {
    UNKNOWN,
    NEWLY_LEARNED,
    FRAGILE,
    CHALLENGING,
    MEDIUM,
    EASY,
    MASTERED
}

enum class ConfidenceLevelV1 {
    LOW,
    MEDIUM,
    HIGH
}

enum class EvidenceTypeV1 {
    EXPLICIT,
    INFERRED,
    REPEATED_PATTERN,
    SESSION_ONLY
}

enum class DurabilityV1 {
    TEMPORARY,
    PROMOTE_IF_REPEATED,
    DURABLE
}

data class MetaphorPolicyV1(
    val allowed: Boolean = false,
    val domains: List<String> = emptyList(),
    val frequency: String? = null,
    val naturalOnly: Boolean = true
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("allowed", allowed)
        putStringListIfNotEmpty("domains", domains)
        frequency?.let { put("frequency", it) }
        put("natural_only", naturalOnly)
    }

    companion object {
        fun parse(o: JSONObject?): MetaphorPolicyV1 {
            if (o == null) return MetaphorPolicyV1()
            return MetaphorPolicyV1(
                allowed = o.optBoolean("allowed", false),
                domains = o.optStringListOrEmpty("domains"),
                frequency = o.optStringOrNullAny("frequency"),
                naturalOnly = o.optBoolean("natural_only", true)
            )
        }
    }
}

data class BucketConfidenceV1(
    val bucket: String,
    val confidence: ConfidenceLevelV1 = ConfidenceLevelV1.LOW
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("bucket", bucket)
        putEnumIfNotNull("confidence", confidence)
    }

    companion object {
        fun parse(o: JSONObject?): BucketConfidenceV1? {
            if (o == null) return null
            val bucket = o.optStringOrNull("bucket") ?: return null
            return BucketConfidenceV1(
                bucket = bucket,
                confidence = enumValueOrNull<ConfidenceLevelV1>(o.optStringOrNull("confidence"))
                    ?: ConfidenceLevelV1.LOW
            )
        }
    }
}

data class EvidenceCountEntryV1(
    val key: String,
    val count: Int = 0
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("key", key)
        put("count", count)
    }

    companion object {
        fun parse(o: JSONObject?): EvidenceCountEntryV1? {
            if (o == null) return null
            val key = o.optStringOrNull("key") ?: return null
            return EvidenceCountEntryV1(
                key = key,
                count = o.optInt("count", 0).coerceAtLeast(0)
            )
        }
    }
}

data class TechniqueComfortEntryV1(
    val technique: String,
    val familiarity: TechniqueFamiliarityV1 = TechniqueFamiliarityV1.UNKNOWN,
    val notes: String? = null
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("technique", technique)
        putEnumIfNotNull("familiarity", familiarity)
        notes?.let { put("notes", it) }
    }

    companion object {
        fun parse(o: JSONObject?): TechniqueComfortEntryV1? {
            if (o == null) return null
            val technique = o.optStringOrNull("technique") ?: return null
            return TechniqueComfortEntryV1(
                technique = technique,
                familiarity = enumValueOrNull<TechniqueFamiliarityV1>(o.optStringOrNull("familiarity"))
                    ?: TechniqueFamiliarityV1.UNKNOWN,
                notes = o.optStringOrNull("notes")
            )
        }
    }
}

data class RelationshipObservationV1(
    val note: String,
    val confidence: ConfidenceLevelV1 = ConfidenceLevelV1.LOW,
    val evidenceType: EvidenceTypeV1 = EvidenceTypeV1.INFERRED,
    val durability: DurabilityV1 = DurabilityV1.PROMOTE_IF_REPEATED
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("note", note)
        putEnumIfNotNull("confidence", confidence)
        putEnumIfNotNull("evidence_type", evidenceType)
        putEnumIfNotNull("durability", durability)
    }

    companion object {
        fun parse(o: JSONObject?): RelationshipObservationV1? {
            if (o == null) return null
            val note = o.optStringOrNull("note") ?: return null
            return RelationshipObservationV1(
                note = note,
                confidence = enumValueOrNull<ConfidenceLevelV1>(o.optStringOrNull("confidence"))
                    ?: ConfidenceLevelV1.LOW,
                evidenceType = enumValueOrNull<EvidenceTypeV1>(o.optStringOrNull("evidence_type"))
                    ?: EvidenceTypeV1.INFERRED,
                durability = enumValueOrNull<DurabilityV1>(o.optStringOrNull("durability"))
                    ?: DurabilityV1.PROMOTE_IF_REPEATED
            )
        }
    }
}

data class RelationshipCandidateUpdateV1(
    val bucket: String,
    val key: String,
    val value: String,
    val confidence: ConfidenceLevelV1 = ConfidenceLevelV1.LOW,
    val evidenceType: EvidenceTypeV1 = EvidenceTypeV1.INFERRED,
    val durability: DurabilityV1 = DurabilityV1.PROMOTE_IF_REPEATED
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("bucket", bucket)
        put("key", key)
        put("value", value)
        putEnumIfNotNull("confidence", confidence)
        putEnumIfNotNull("evidence_type", evidenceType)
        putEnumIfNotNull("durability", durability)
    }

    companion object {
        fun parse(o: JSONObject?): RelationshipCandidateUpdateV1? {
            if (o == null) return null
            val bucket = o.optStringOrNull("bucket") ?: return null
            val key = o.optStringOrNull("key") ?: return null
            val value = o.optStringOrNull("value") ?: return null
            return RelationshipCandidateUpdateV1(
                bucket = bucket,
                key = key,
                value = value,
                confidence = enumValueOrNull<ConfidenceLevelV1>(o.optStringOrNull("confidence"))
                    ?: ConfidenceLevelV1.LOW,
                evidenceType = enumValueOrNull<EvidenceTypeV1>(o.optStringOrNull("evidence_type"))
                    ?: EvidenceTypeV1.INFERRED,
                durability = enumValueOrNull<DurabilityV1>(o.optStringOrNull("durability"))
                    ?: DurabilityV1.PROMOTE_IF_REPEATED
            )
        }
    }
}

data class RelationshipToneBondV1(
    val relationshipTone: RelationshipToneV1? = null,
    val familiarityPreference: FamiliarityPreferenceV1? = null,
    val nameUsagePreference: String? = null,
    val encouragementStyle: String? = null,
    val humorPreference: HumorPreferenceV1? = null,
    val trustBuilders: List<String> = emptyList(),
    val warmthPreferences: List<String> = emptyList(),
    val bondNotes: List<String> = emptyList()
) {
    fun toJson(): JSONObject = JSONObject().apply {
        putEnumIfNotNull("relationship_tone", relationshipTone)
        putEnumIfNotNull("familiarity_preference", familiarityPreference)
        nameUsagePreference?.let { put("name_usage_preference", it) }
        encouragementStyle?.let { put("encouragement_style", it) }
        putEnumIfNotNull("humor_preference", humorPreference)
        putStringListIfNotEmpty("trust_builders", trustBuilders)
        putStringListIfNotEmpty("warmth_preferences", warmthPreferences)
        putStringListIfNotEmpty("bond_notes", bondNotes)
    }

    companion object {
        fun parse(o: JSONObject?): RelationshipToneBondV1 {
            if (o == null) return RelationshipToneBondV1()
            return RelationshipToneBondV1(
                relationshipTone = enumValueOrNull<RelationshipToneV1>(o.optStringOrNull("relationship_tone")),
                familiarityPreference = enumValueOrNull<FamiliarityPreferenceV1>(o.optStringOrNull("familiarity_preference")),
                nameUsagePreference = o.optStringOrNull("name_usage_preference"),
                encouragementStyle = o.optStringOrNull("encouragement_style"),
                humorPreference = enumValueOrNull<HumorPreferenceV1>(o.optStringOrNull("humor_preference")),
                trustBuilders = o.optStringListOrEmpty("trust_builders"),
                warmthPreferences = o.optStringListOrEmpty("warmth_preferences"),
                bondNotes = o.optStringListOrEmpty("bond_notes")
            )
        }
    }
}

data class CommunicationSpeechStyleV1(
    val pacePreference: PacePreferenceV1? = null,
    val verbosityPreference: String? = null,
    val speechRhythmPreference: String? = null,
    val questionTolerance: String? = null,
    val confirmationStylePreference: String? = null,
    val repetitionSensitivity: String? = null,
    val jargonTolerance: String? = null,
    val ctaStylePreference: String? = null,
    val avoidSpeechPatterns: List<String> = emptyList()
) {
    fun toJson(): JSONObject = JSONObject().apply {
        putEnumIfNotNull("pace_preference", pacePreference)
        verbosityPreference?.let { put("verbosity_preference", it) }
        speechRhythmPreference?.let { put("speech_rhythm_preference", it) }
        questionTolerance?.let { put("question_tolerance", it) }
        confirmationStylePreference?.let { put("confirmation_style_preference", it) }
        repetitionSensitivity?.let { put("repetition_sensitivity", it) }
        jargonTolerance?.let { put("jargon_tolerance", it) }
        ctaStylePreference?.let { put("cta_style_preference", it) }
        putStringListIfNotEmpty("avoid_speech_patterns", avoidSpeechPatterns)
    }

    companion object {
        fun parse(o: JSONObject?): CommunicationSpeechStyleV1 {
            if (o == null) return CommunicationSpeechStyleV1()
            return CommunicationSpeechStyleV1(
                pacePreference = enumValueOrNull<PacePreferenceV1>(o.optStringOrNull("pace_preference")),
                verbosityPreference = o.optStringOrNull("verbosity_preference"),
                speechRhythmPreference = o.optStringOrNull("speech_rhythm_preference"),
                questionTolerance = o.optStringOrNull("question_tolerance"),
                confirmationStylePreference = o.optStringOrNull("confirmation_style_preference"),
                repetitionSensitivity = o.optStringOrNull("repetition_sensitivity"),
                jargonTolerance = o.optStringOrNull("jargon_tolerance"),
                ctaStylePreference = o.optStringOrNull("cta_style_preference"),
                avoidSpeechPatterns = o.optStringListOrEmpty("avoid_speech_patterns")
            )
        }
    }
}

data class LearningExplanationModelV1(
    val learningPreferences: List<ExplanationStyleV1> = emptyList(),
    val proofPreference: ProofPreferenceV1? = null,
    val setupPreference: String? = null,
    val confrontationPreference: String? = null,
    val resolutionPreference: String? = null,
    val analogyPreference: String? = null,
    val metaphorPolicy: MetaphorPolicyV1 = MetaphorPolicyV1(),
    val clarityTriggers: List<String> = emptyList(),
    val confusionTriggers: List<String> = emptyList()
) {
    fun toJson(): JSONObject = JSONObject().apply {
        if (learningPreferences.isNotEmpty()) {
            put("learning_preferences", JSONArray().apply { learningPreferences.forEach { put(it.name) } })
        }
        putEnumIfNotNull("proof_preference", proofPreference)
        setupPreference?.let { put("setup_preference", it) }
        confrontationPreference?.let { put("confrontation_preference", it) }
        resolutionPreference?.let { put("resolution_preference", it) }
        analogyPreference?.let { put("analogy_preference", it) }
        if (metaphorPolicy != MetaphorPolicyV1()) put("metaphor_policy", metaphorPolicy.toJson())
        putStringListIfNotEmpty("clarity_triggers", clarityTriggers)
        putStringListIfNotEmpty("confusion_triggers", confusionTriggers)
    }

    companion object {
        fun parse(o: JSONObject?): LearningExplanationModelV1 {
            if (o == null) return LearningExplanationModelV1()
            val learningPrefs = o.optJSONArrayOrNull("learning_preferences")
                ?.toStringList(50)
                ?.mapNotNull { enumValueOrNull<ExplanationStyleV1>(it) }
                ?: emptyList()
            return LearningExplanationModelV1(
                learningPreferences = learningPrefs,
                proofPreference = enumValueOrNull<ProofPreferenceV1>(o.optStringOrNull("proof_preference")),
                setupPreference = o.optStringOrNull("setup_preference"),
                confrontationPreference = o.optStringOrNull("confrontation_preference"),
                resolutionPreference = o.optStringOrNull("resolution_preference"),
                analogyPreference = o.optStringOrNull("analogy_preference"),
                metaphorPolicy = MetaphorPolicyV1.parse(o.optJSONObjectOrNull("metaphor_policy")),
                clarityTriggers = o.optStringListOrEmpty("clarity_triggers"),
                confusionTriggers = o.optStringListOrEmpty("confusion_triggers")
            )
        }
    }
}

data class SudokuKnowledgeTechniqueMapV1(
    val sudokuIdentity: String? = null,
    val skillSelfView: String? = null,
    val techniqueComfort: List<TechniqueComfortEntryV1> = emptyList(),
    val recentlyLearnedTechniques: List<String> = emptyList(),
    val fragileTechniques: List<String> = emptyList(),
    val challengingTechniques: List<String> = emptyList(),
    val favoriteTechniques: List<String> = emptyList(),
    val masteredPatterns: List<String> = emptyList(),
    val teachingPriorityGaps: List<String> = emptyList()
) {
    fun toJson(): JSONObject = JSONObject().apply {
        sudokuIdentity?.let { put("sudoku_identity", it) }
        skillSelfView?.let { put("skill_self_view", it) }
        putObjectListIfNotEmpty("technique_comfort", techniqueComfort) { it.toJson() }
        putStringListIfNotEmpty("recently_learned_techniques", recentlyLearnedTechniques)
        putStringListIfNotEmpty("fragile_techniques", fragileTechniques)
        putStringListIfNotEmpty("challenging_techniques", challengingTechniques)
        putStringListIfNotEmpty("favorite_techniques", favoriteTechniques)
        putStringListIfNotEmpty("mastered_patterns", masteredPatterns)
        putStringListIfNotEmpty("teaching_priority_gaps", teachingPriorityGaps)
    }

    companion object {
        fun parse(o: JSONObject?): SudokuKnowledgeTechniqueMapV1 {
            if (o == null) return SudokuKnowledgeTechniqueMapV1()
            return SudokuKnowledgeTechniqueMapV1(
                sudokuIdentity = o.optStringOrNull("sudoku_identity"),
                skillSelfView = o.optStringOrNull("skill_self_view"),
                techniqueComfort = o.optObjectListOrEmpty("technique_comfort", 200) { TechniqueComfortEntryV1.parse(it) },
                recentlyLearnedTechniques = o.optStringListOrEmpty("recently_learned_techniques"),
                fragileTechniques = o.optStringListOrEmpty("fragile_techniques"),
                challengingTechniques = o.optStringListOrEmpty("challenging_techniques"),
                favoriteTechniques = o.optStringListOrEmpty("favorite_techniques"),
                masteredPatterns = o.optStringListOrEmpty("mastered_patterns"),
                teachingPriorityGaps = o.optStringListOrEmpty("teaching_priority_gaps")
            )
        }
    }
}

data class SolvingMindsetCognitiveStyleV1(
    val thinkingStyle: List<String> = emptyList(),
    val decisionStyle: String? = null,
    val attentionStyle: String? = null,
    val mentalModelNotes: List<String> = emptyList(),
    val commonReasoningMoves: List<String> = emptyList(),
    val commonBlindSpots: List<String> = emptyList(),
    val errorTendencies: List<String> = emptyList(),
    val confidencePattern: String? = null,
    val autonomyPreference: String? = null
) {
    fun toJson(): JSONObject = JSONObject().apply {
        putStringListIfNotEmpty("thinking_style", thinkingStyle)
        decisionStyle?.let { put("decision_style", it) }
        attentionStyle?.let { put("attention_style", it) }
        putStringListIfNotEmpty("mental_model_notes", mentalModelNotes)
        putStringListIfNotEmpty("common_reasoning_moves", commonReasoningMoves)
        putStringListIfNotEmpty("common_blind_spots", commonBlindSpots)
        putStringListIfNotEmpty("error_tendencies", errorTendencies)
        confidencePattern?.let { put("confidence_pattern", it) }
        autonomyPreference?.let { put("autonomy_preference", it) }
    }

    companion object {
        fun parse(o: JSONObject?): SolvingMindsetCognitiveStyleV1 {
            if (o == null) return SolvingMindsetCognitiveStyleV1()
            return SolvingMindsetCognitiveStyleV1(
                thinkingStyle = o.optStringListOrEmpty("thinking_style"),
                decisionStyle = o.optStringOrNull("decision_style"),
                attentionStyle = o.optStringOrNull("attention_style"),
                mentalModelNotes = o.optStringListOrEmpty("mental_model_notes"),
                commonReasoningMoves = o.optStringListOrEmpty("common_reasoning_moves"),
                commonBlindSpots = o.optStringListOrEmpty("common_blind_spots"),
                errorTendencies = o.optStringListOrEmpty("error_tendencies"),
                confidencePattern = o.optStringOrNull("confidence_pattern"),
                autonomyPreference = o.optStringOrNull("autonomy_preference")
            )
        }
    }
}

data class WorldContextSolvingRealityV1(
    val solvingMedium: String? = null,
    val usualEnvironments: List<String> = emptyList(),
    val interactionConstraints: List<String> = emptyList(),
    val sessionStyle: String? = null,
    val physicalRealityNotes: List<String> = emptyList(),
    val validationNeeds: List<String> = emptyList(),
    val routinePatterns: List<String> = emptyList()
) {
    fun toJson(): JSONObject = JSONObject().apply {
        solvingMedium?.let { put("solving_medium", it) }
        putStringListIfNotEmpty("usual_environments", usualEnvironments)
        putStringListIfNotEmpty("interaction_constraints", interactionConstraints)
        sessionStyle?.let { put("session_style", it) }
        putStringListIfNotEmpty("physical_reality_notes", physicalRealityNotes)
        putStringListIfNotEmpty("validation_needs", validationNeeds)
        putStringListIfNotEmpty("routine_patterns", routinePatterns)
    }

    companion object {
        fun parse(o: JSONObject?): WorldContextSolvingRealityV1 {
            if (o == null) return WorldContextSolvingRealityV1()
            return WorldContextSolvingRealityV1(
                solvingMedium = o.optStringOrNull("solving_medium"),
                usualEnvironments = o.optStringListOrEmpty("usual_environments"),
                interactionConstraints = o.optStringListOrEmpty("interaction_constraints"),
                sessionStyle = o.optStringOrNull("session_style"),
                physicalRealityNotes = o.optStringListOrEmpty("physical_reality_notes"),
                validationNeeds = o.optStringListOrEmpty("validation_needs"),
                routinePatterns = o.optStringListOrEmpty("routine_patterns")
            )
        }
    }
}

data class PersonalLanguageMeaningHooksV1(
    val userJargon: List<String> = emptyList(),
    val preferredTerms: List<String> = emptyList(),
    val dislikedTerms: List<String> = emptyList(),
    val mentalLabels: List<String> = emptyList(),
    val metaphorDomains: List<String> = emptyList(),
    val meaningHooks: List<String> = emptyList(),
    val languageRegister: String? = null,
    val bilingualOrLanguageNotes: String? = null
) {
    fun toJson(): JSONObject = JSONObject().apply {
        putStringListIfNotEmpty("user_jargon", userJargon)
        putStringListIfNotEmpty("preferred_terms", preferredTerms)
        putStringListIfNotEmpty("disliked_terms", dislikedTerms)
        putStringListIfNotEmpty("mental_labels", mentalLabels)
        putStringListIfNotEmpty("metaphor_domains", metaphorDomains)
        putStringListIfNotEmpty("meaning_hooks", meaningHooks)
        languageRegister?.let { put("language_register", it) }
        bilingualOrLanguageNotes?.let { put("bilingual_or_language_notes", it) }
    }

    companion object {
        fun parse(o: JSONObject?): PersonalLanguageMeaningHooksV1 {
            if (o == null) return PersonalLanguageMeaningHooksV1()
            return PersonalLanguageMeaningHooksV1(
                userJargon = o.optStringListOrEmpty("user_jargon"),
                preferredTerms = o.optStringListOrEmpty("preferred_terms"),
                dislikedTerms = o.optStringListOrEmpty("disliked_terms"),
                mentalLabels = o.optStringListOrEmpty("mental_labels"),
                metaphorDomains = o.optStringListOrEmpty("metaphor_domains"),
                meaningHooks = o.optStringListOrEmpty("meaning_hooks"),
                languageRegister = o.optStringOrNull("language_register"),
                bilingualOrLanguageNotes = o.optStringOrNull("bilingual_or_language_notes")
            )
        }
    }
}

data class InteractionHistoryMemoryIntegrityV1(
    val userExperienceSummary: String? = null,
    val currentPriorities: List<String> = emptyList(),
    val recentGrowthEdges: List<String> = emptyList(),
    val recentFrictionEdges: List<String> = emptyList(),
    val confidenceByBucket: List<BucketConfidenceV1> = emptyList(),
    val evidenceCounts: List<EvidenceCountEntryV1> = emptyList(),
    val lastRefreshedAt: String? = null,
    val stalenessFlags: List<String> = emptyList(),
    val sourceNotes: List<String> = emptyList()
) {
    fun toJson(): JSONObject = JSONObject().apply {
        userExperienceSummary?.let { put("user_experience_summary", it) }
        putStringListIfNotEmpty("current_priorities", currentPriorities)
        putStringListIfNotEmpty("recent_growth_edges", recentGrowthEdges)
        putStringListIfNotEmpty("recent_friction_edges", recentFrictionEdges)
        putObjectListIfNotEmpty("confidence_by_bucket", confidenceByBucket) { it.toJson() }
        putObjectListIfNotEmpty("evidence_counts", evidenceCounts) { it.toJson() }
        lastRefreshedAt?.let { put("last_refreshed_at", it) }
        putStringListIfNotEmpty("staleness_flags", stalenessFlags)
        putStringListIfNotEmpty("source_notes", sourceNotes)
    }

    companion object {
        fun parse(o: JSONObject?): InteractionHistoryMemoryIntegrityV1 {
            if (o == null) return InteractionHistoryMemoryIntegrityV1()
            return InteractionHistoryMemoryIntegrityV1(
                userExperienceSummary = o.optStringOrNull("user_experience_summary"),
                currentPriorities = o.optStringListOrEmpty("current_priorities"),
                recentGrowthEdges = o.optStringListOrEmpty("recent_growth_edges"),
                recentFrictionEdges = o.optStringListOrEmpty("recent_friction_edges"),
                confidenceByBucket = o.optObjectListOrEmpty("confidence_by_bucket", 100) { BucketConfidenceV1.parse(it) },
                evidenceCounts = o.optObjectListOrEmpty("evidence_counts", 200) { EvidenceCountEntryV1.parse(it) },
                lastRefreshedAt = o.optStringOrNull("last_refreshed_at"),
                stalenessFlags = o.optStringListOrEmpty("staleness_flags"),
                sourceNotes = o.optStringListOrEmpty("source_notes")
            )
        }
    }
}

data class RelationshipMemoryV1(
    val relationshipToneBond: RelationshipToneBondV1 = RelationshipToneBondV1(),
    val communicationSpeechStyle: CommunicationSpeechStyleV1 = CommunicationSpeechStyleV1(),
    val learningExplanationModel: LearningExplanationModelV1 = LearningExplanationModelV1(),
    val sudokuKnowledgeTechniqueMap: SudokuKnowledgeTechniqueMapV1 = SudokuKnowledgeTechniqueMapV1(),
    val solvingMindsetCognitiveStyle: SolvingMindsetCognitiveStyleV1 = SolvingMindsetCognitiveStyleV1(),
    val worldContextSolvingReality: WorldContextSolvingRealityV1 = WorldContextSolvingRealityV1(),
    val personalLanguageMeaningHooks: PersonalLanguageMeaningHooksV1 = PersonalLanguageMeaningHooksV1(),
    val interactionHistoryMemoryIntegrity: InteractionHistoryMemoryIntegrityV1 = InteractionHistoryMemoryIntegrityV1()
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("relationship_tone_bond", relationshipToneBond.toJson())
        put("communication_speech_style", communicationSpeechStyle.toJson())
        put("learning_explanation_model", learningExplanationModel.toJson())
        put("sudoku_knowledge_technique_map", sudokuKnowledgeTechniqueMap.toJson())
        put("solving_mindset_cognitive_style", solvingMindsetCognitiveStyle.toJson())
        put("world_context_solving_reality", worldContextSolvingReality.toJson())
        put("personal_language_meaning_hooks", personalLanguageMeaningHooks.toJson())
        put("interaction_history_memory_integrity", interactionHistoryMemoryIntegrity.toJson())
    }

    companion object {
        fun defaults(): RelationshipMemoryV1 = RelationshipMemoryV1()

        fun parse(o: JSONObject?): RelationshipMemoryV1 {
            if (o == null) return RelationshipMemoryV1()
            return RelationshipMemoryV1(
                relationshipToneBond = RelationshipToneBondV1.parse(o.optJSONObjectOrNull("relationship_tone_bond")),
                communicationSpeechStyle = CommunicationSpeechStyleV1.parse(o.optJSONObjectOrNull("communication_speech_style")),
                learningExplanationModel = LearningExplanationModelV1.parse(o.optJSONObjectOrNull("learning_explanation_model")),
                sudokuKnowledgeTechniqueMap = SudokuKnowledgeTechniqueMapV1.parse(o.optJSONObjectOrNull("sudoku_knowledge_technique_map")),
                solvingMindsetCognitiveStyle = SolvingMindsetCognitiveStyleV1.parse(o.optJSONObjectOrNull("solving_mindset_cognitive_style")),
                worldContextSolvingReality = WorldContextSolvingRealityV1.parse(o.optJSONObjectOrNull("world_context_solving_reality")),
                personalLanguageMeaningHooks = PersonalLanguageMeaningHooksV1.parse(o.optJSONObjectOrNull("personal_language_meaning_hooks")),
                interactionHistoryMemoryIntegrity = InteractionHistoryMemoryIntegrityV1.parse(o.optJSONObjectOrNull("interaction_history_memory_integrity"))
            )
        }
    }
}

data class RelationshipDeltaV1(
    val observations: List<RelationshipObservationV1> = emptyList(),
    val candidateUpdates: List<RelationshipCandidateUpdateV1> = emptyList(),
    val confidence: ConfidenceLevelV1 = ConfidenceLevelV1.LOW,
    val durability: DurabilityV1 = DurabilityV1.PROMOTE_IF_REPEATED,
    val expiresAfterTurns: Int? = null
) {
    fun toJson(): JSONObject = JSONObject().apply {
        putObjectListIfNotEmpty("observations", observations) { it.toJson() }
        putObjectListIfNotEmpty("candidate_updates", candidateUpdates) { it.toJson() }
        putEnumIfNotNull("confidence", confidence)
        putEnumIfNotNull("durability", durability)
        expiresAfterTurns?.let { put("expires_after_turns", it) }
    }

    companion object {
        fun parse(o: JSONObject?): RelationshipDeltaV1 {
            if (o == null) return RelationshipDeltaV1()
            return RelationshipDeltaV1(
                observations = o.optObjectListOrEmpty("observations", 200) { RelationshipObservationV1.parse(it) },
                candidateUpdates = o.optObjectListOrEmpty("candidate_updates", 200) { RelationshipCandidateUpdateV1.parse(it) },
                confidence = enumValueOrNull<ConfidenceLevelV1>(o.optStringOrNull("confidence"))
                    ?: ConfidenceLevelV1.LOW,
                durability = enumValueOrNull<DurabilityV1>(o.optStringOrNull("durability"))
                    ?: DurabilityV1.PROMOTE_IF_REPEATED,
                expiresAfterTurns = o.optIntOrNull("expires_after_turns")
            )
        }
    }
}

data class PersonalizationMiniV1(
    val relationshipTone: String? = null,
    val speechStyle: String? = null,
    val learningBias: String? = null,
    val techniqueContext: String? = null,
    val metaphorPolicy: MetaphorPolicyV1? = null,
    val avoid: List<String> = emptyList(),
    val userTerms: List<String> = emptyList(),
    val coachingNote: String? = null
) {
    fun toJson(): JSONObject = JSONObject().apply {
        relationshipTone?.let { put("relationship_tone", it) }
        speechStyle?.let { put("speech_style", it) }
        learningBias?.let { put("learning_bias", it) }
        techniqueContext?.let { put("technique_context", it) }
        metaphorPolicy?.let { put("metaphor_policy", it.toJson()) }
        putStringListIfNotEmpty("avoid", avoid)
        putStringListIfNotEmpty("user_terms", userTerms)
        coachingNote?.let { put("coaching_note", it) }
    }

    companion object {
        fun parse(o: JSONObject?): PersonalizationMiniV1 {
            if (o == null) return PersonalizationMiniV1()
            return PersonalizationMiniV1(
                relationshipTone = o.optStringOrNull("relationship_tone"),
                speechStyle = o.optStringOrNull("speech_style"),
                learningBias = o.optStringOrNull("learning_bias"),
                techniqueContext = o.optStringOrNull("technique_context"),
                metaphorPolicy = o.optJSONObjectOrNull("metaphor_policy")?.let { MetaphorPolicyV1.parse(it) },
                avoid = o.optStringListOrEmpty("avoid"),
                userTerms = o.optStringListOrEmpty("user_terms"),
                coachingNote = o.optStringOrNull("coaching_note")
            )
        }
    }
}


// -------------------------------------------------------------------------
// TurnContextV1 — structured Tick-1 input payload (Phase 2 binder depends on it)
// -------------------------------------------------------------------------
data class TurnContextV1(
    val schema: String = "TurnContextV1",
    val v: Int = 1,

    val turnId: Long,
    val mode: String,
    val phase: String,
    val userText: String,

    val pending: PendingCtxV1? = null,
    val focusCell: String? = null,
    val focusCoreferencePolicy: String? = null,
    val lastAssistantQuestionKey: String? = null,
    val canonicalSolvingPositionKind: String? = null,
    val solvingHandoff: SolvingHandoffCtxV1? = null,
    val awaitedAssistantAnswer: AwaitedAssistantAnswerCtxV1? = null,
    val recentTurnsPolicy: String? = null,

    // Keep as strings (already full-shape JSON) for telemetry/audit.
    val userTallyJson: String,
    val assistantTallyJson: String,
    val recentTurnsJson: String
){

    data class PendingCtxV1(
        val pendingBefore: String? = null,
        val expectedAnswerKind: String? = null,
        val targetCell: String? = null
    ) {
        fun toJson(): JSONObject = JSONObject().apply {
            put("pending_before", pendingBefore ?: JSONObject.NULL)
            put("expected_answer_kind", expectedAnswerKind ?: JSONObject.NULL)
            put("target_cell", targetCell ?: JSONObject.NULL)
        }

        companion object {
            fun parse(o: JSONObject?): PendingCtxV1? {
                if (o == null) return null
                return PendingCtxV1(
                    pendingBefore = o.optStringOrNull("pending_before"),
                    expectedAnswerKind = o.optStringOrNull("expected_answer_kind"),
                    targetCell = o.optStringOrNull("target_cell")
                )
            }
        }
    }


    data class SolvingHandoffCtxV1(
        val handoffKind: String? = null,
        val authority: String = "STRUCTURED_APP_STATE",
        val commitAlreadyApplied: Boolean = false,
        val committedCell: String? = null,
        val assistantCtaKind: String? = null,
        val assistantCtaScope: String? = null,
        val genericAssentDefaultIntent: String? = null,
        val detourOverrideRule: String? = null
    ) {
        fun toJson(): JSONObject = JSONObject().apply {
            put("handoff_kind", handoffKind ?: JSONObject.NULL)
            put("authority", authority)
            put("commit_already_applied", commitAlreadyApplied)
            put("committed_cell", committedCell ?: JSONObject.NULL)
            put("assistant_cta_kind", assistantCtaKind ?: JSONObject.NULL)
            put("assistant_cta_scope", assistantCtaScope ?: JSONObject.NULL)
            put("generic_assent_default_intent", genericAssentDefaultIntent ?: JSONObject.NULL)
            put("detour_override_rule", detourOverrideRule ?: JSONObject.NULL)
        }

        companion object {
            fun parse(o: JSONObject?): SolvingHandoffCtxV1? {
                if (o == null) return null
                return SolvingHandoffCtxV1(
                    handoffKind = o.optStringOrNull("handoff_kind"),
                    authority = o.optStringOrNull("authority") ?: "STRUCTURED_APP_STATE",
                    commitAlreadyApplied = o.optBoolean("commit_already_applied", false),
                    committedCell = o.optStringOrNull("committed_cell"),
                    assistantCtaKind = o.optStringOrNull("assistant_cta_kind"),
                    assistantCtaScope = o.optStringOrNull("assistant_cta_scope"),
                    genericAssentDefaultIntent = o.optStringOrNull("generic_assent_default_intent"),
                    detourOverrideRule = o.optStringOrNull("detour_override_rule")
                )
            }
        }
    }

    data class AwaitedAssistantAnswerCtxV1(
        val owner: String = "APP_ROUTE_OWNER",
        val questionKey: String? = null,
        val questionKind: String? = null,
        val expectedAnswerKind: String? = null,
        val followupDisposition: String = "APP_ROUTE_FOLLOWUP"
    ) {
        fun toJson(): JSONObject = JSONObject().apply {
            put("owner", owner)
            put("question_key", questionKey ?: JSONObject.NULL)
            put("question_kind", questionKind ?: JSONObject.NULL)
            put("expected_answer_kind", expectedAnswerKind ?: JSONObject.NULL)
            put("followup_disposition", followupDisposition)
        }

        companion object {
            fun parse(o: JSONObject?): AwaitedAssistantAnswerCtxV1? {
                if (o == null) return null
                return AwaitedAssistantAnswerCtxV1(
                    owner = o.optStringOrNull("owner") ?: "APP_ROUTE_OWNER",
                    questionKey = o.optStringOrNull("question_key"),
                    questionKind = o.optStringOrNull("question_kind"),
                    expectedAnswerKind = o.optStringOrNull("expected_answer_kind"),
                    followupDisposition = o.optStringOrNull("followup_disposition") ?: "APP_ROUTE_FOLLOWUP"
                )
            }
        }
    }

    fun toJsonString(): String {
        val o = JSONObject().apply {
            put("schema", schema)
            put("v", v)
            put("turn_id", turnId)
            put("mode", mode)
            put("phase", phase)
            put("user_text", userText)


            put("pending", pending?.toJson() ?: JSONObject.NULL)
            put("focus_cell", focusCell ?: JSONObject.NULL)
            put("focus_coreference_policy", focusCoreferencePolicy ?: JSONObject.NULL)
            put("last_assistant_question_key", lastAssistantQuestionKey ?: JSONObject.NULL)
            put("canonical_solving_position_kind", canonicalSolvingPositionKind ?: JSONObject.NULL)
            put("solving_handoff", solvingHandoff?.toJson() ?: JSONObject.NULL)
            put("awaited_assistant_answer", awaitedAssistantAnswer?.toJson() ?: JSONObject.NULL)
            put("recent_turns_policy", recentTurnsPolicy ?: JSONObject.NULL)
            put(
                "tally",
                JSONObject().apply {
                    put("user_tally", runCatching { JSONObject(userTallyJson) }.getOrElse { userTallyJson })
                    put("assistant_tally", runCatching { JSONObject(assistantTallyJson) }.getOrElse { assistantTallyJson })
                }
            )

            put("recent_turns", runCatching { JSONArray(recentTurnsJson) }.getOrElse { recentTurnsJson })
        }
        return o.toString()
    }

    companion object {
        fun fromJsonString(json: String): TurnContextV1? {
            if (json.isBlank()) return null
            val o = runCatching { JSONObject(json) }.getOrNull() ?: return null
            val schema = o.optString("schema", "")
            if (!schema.equals("TurnContextV1", ignoreCase = true)) return null

            val pendingObj = o.optJSONObject("pending")
            val solvingHandoffObj = o.optJSONObject("solving_handoff")
            val awaitedObj = o.optJSONObject("awaited_assistant_answer")
            val tallyObj = o.optJSONObject("tally")
            val recentTurns = o.opt("recent_turns")

            return TurnContextV1(
                schema = "TurnContextV1",
                v = o.optInt("v", 1),
                turnId = o.optLong("turn_id", -1L),
                mode = o.optString("mode", ""),
                phase = o.optString("phase", ""),
                userText = o.optString("user_text", ""),

                pending = PendingCtxV1.parse(pendingObj),

                focusCell = o.optStringOrNull("focus_cell"),
                focusCoreferencePolicy = o.optStringOrNull("focus_coreference_policy"),
                lastAssistantQuestionKey = o.optStringOrNull("last_assistant_question_key"),
                canonicalSolvingPositionKind = o.optStringOrNull("canonical_solving_position_kind"),
                solvingHandoff = SolvingHandoffCtxV1.parse(solvingHandoffObj),
                awaitedAssistantAnswer = AwaitedAssistantAnswerCtxV1.parse(awaitedObj),
                recentTurnsPolicy = o.optStringOrNull("recent_turns_policy"),


                userTallyJson = runCatching { tallyObj?.opt("user_tally")?.toString() ?: "{}" }.getOrElse { "{}" },
                assistantTallyJson = runCatching { tallyObj?.opt("assistant_tally")?.toString() ?: "{}" }.getOrElse { "{}" },
                recentTurnsJson = runCatching {
                    when (recentTurns) {
                        is JSONArray -> recentTurns.toString()
                        else -> recentTurns?.toString() ?: "[]"
                    }
                }.getOrElse { "[]" }
            )
        }
    }
}



// -----------------------------------------------------------------------------
// Shared V1 enums/refs used by Tick1 intents and Tick2 decisions
// -----------------------------------------------------------------------------

enum class MutationOpV1 {
    SET, CLEAR;

    companion object {
        fun parse(s: String?): MutationOpV1? =
            values().firstOrNull { it.name == s }
    }
}

enum class RegionKindV1 {
    ROW, COL, BOX;

    companion object {
        fun parse(s: String?): RegionKindV1? =
            values().firstOrNull { it.name == s }
    }
}

data class RegionRefV1(
    val kind: RegionKindV1,
    val index: Int // 1..9
) {
    fun validate(errors: MutableList<String>) {
        if (index !in 1..9) errors.add("region.index out of range: $index")
    }

    fun toJson(): JSONObject = JSONObject().apply {
        put("kind", kind.name)
        put("index", index)
    }

    companion object {
        fun parse(o: JSONObject?, errors: MutableList<String>): RegionRefV1? {
            if (o == null) return null
            val kind = RegionKindV1.parse(o.optStringOrNull("kind"))
            val idx = o.optIntOrNull("index")
            if (kind == null) {
                errors.add("region.kind missing/invalid")
                return null
            }
            if (idx == null) {
                errors.add("region.index missing")
                return null
            }
            val r = RegionRefV1(kind, idx)
            r.validate(errors)
            return r
        }
    }
}

// -----------------------------------------------------------------------------
// Phase 0 — Detour question taxonomy (scaffold only; no behavior yet)
// -----------------------------------------------------------------------------

enum class DetourQuestionClassV1 {
    STEP_CLARIFICATION,
    PROOF_CHALLENGE,
    TARGET_CELL_QUERY,
    NEIGHBOR_CELL_QUERY,
    CANDIDATE_STATE_QUERY,
    USER_REASONING_CHECK,
    ALTERNATIVE_TECHNIQUE_QUERY,
    ROUTE_COMPARISON_QUERY,
    OVERLAY_CONTROL,
    GENERAL_QUESTION,
    REPAIR_QUESTION,
    ROUTE_CONTROL,
    UNSUPPORTED_DETOUR
}

/**
 * Wave 1 — detour demand compression scaffold.
 *
 * Important:
 * - this does NOT replace ReplyDemandCategoryV1 yet
 * - it is a narrower detour-only taxonomy used to prepare the move away
 *   from shared DETOUR_CONTEXT toward typed detour packets
 * - Wave 1 only covers the 3 solver-backed high-value families
 */
enum class DetourDemandCategoryV2 {
    MOVE_PROOF_OR_TARGET_EXPLANATION,
    LOCAL_GRID_INSPECTION,
    USER_PROPOSAL_VERDICT
}

/**
 * Wave 1 subtype for proof / target explanation detours.
 */
enum class DetourMoveProofProfileV1 {
    WHY_TARGET,
    WHY_DIGIT,
    WHY_NOT_DIGIT,
    HOUSE_BLOCKERS,
    DIGIT_LOCATIONS_IN_HOUSE,
    ONLY_PLACE_FOR_DIGIT_IN_HOUSE,
    RIVAL_COMPARISON_IN_HOUSE,
    TECHNIQUE_LEGITIMACY,
    ROUTE_PRIORITY_COMPARISON,
    PROVE_ELIMINATION,
    PROVE_FORCE,
    TARGET_BEFORE_AFTER
}

/**
 * Wave 1 subtype for local inspection detours.
 */
enum class DetourLocalGridInspectionProfileV1 {
    CELL_CANDIDATES,
    HOUSE_CANDIDATE_MAP,
    DIGIT_LOCATIONS,
    LOCAL_EFFECTS,
    NEARBY_CELL_STATUS,
    TARGET_NEIGHBORHOOD
}

/**
 * Wave 1 subtype for user-proposal verdict detours.
 */
enum class DetourUserProposalKindV1 {
    PROPOSED_DIGIT,
    PROPOSED_ELIMINATION,
    PROPOSED_PATTERN,
    PROPOSED_HOUSE_CLAIM,
    PROPOSED_LOCAL_CHAIN,
    GENERAL_REASONING_CHECK
}

/**
 * Wave 1 permanent-design shared detour narrative archetypes.
 *
 * These enums are shared by:
 * - detour packet contracts in policy space
 * - native detour narrative models in solving/DetourNarrativeModelsV1.kt
 *
 * Series-B P2 + Phase-1 vocabulary extension:
 * Proof-challenge detours treat archetype as a first-class speech-shaping
 * family rather than a thin rollout label. The family now includes
 * LOCAL_PERMISSIBILITY_SCAN so richer local candidate-survival narration can
 * be expressed explicitly rather than collapsed into generic insufficiency.
 */
enum class DetourNarrativeArchetypeV1 {
    LOCAL_PROOF_SPOTLIGHT,
    LOCAL_CONTRADICTION_SPOTLIGHT,
    LOCAL_PERMISSIBILITY_SCAN,
    HOUSE_ALREADY_OCCUPIED,
    CELL_ALREADY_FILLED,
    SURVIVOR_LADDER,
    CONTRAST_DUEL,
    PATTERN_LEGITIMACY_CHECK,
    HONEST_INSUFFICIENCY_ANSWER,
    STATE_READOUT,
    PROPOSAL_VERDICT
}

/**
 * Stronger discipline for native detour atoms / replies.
 *
 * These enums are shared by policy contracts and native detour narrative
 * models. Builders are wired in later phases.
 */
enum class DetourAnswerBoundaryV1 {
    DO_NOT_BECOME_PROOF_LADDER,
    DO_NOT_BECOME_BOARD_AUDIT,
    DO_NOT_SWITCH_ROUTE,
    DO_NOT_COMMIT_MOVE,
    DO_NOT_OPEN_NEW_DETOUR_TREE,
    DO_NOT_RETEACH_FULL_TECHNIQUE,
    DO_NOT_OVERRIDE_VISUAL_FOCUS_UNLESS_REQUESTED
}

enum class DetourHandoverModeV1 {
    RETURN_TO_CURRENT_STAGE,
    RETURN_TO_CURRENT_MOVE,
    HOLD_FOCUS_HERE,
    AWAIT_USER_CONTROL,
    REPAIR_THEN_RETURN
}

enum class DetourOverlayModeV1 {
    PRESERVE,
    REPLACE,
    AUGMENT,
    CLEAR
}

enum class SolvingRoadSemanticV2 {
    CONTINUE_FORWARD,
    STAY_AND_ELABORATE,
    REPEAT_CURRENT_STAGE,
    GO_BACKWARD,
    DETOUR_OR_ROUTE_CONTROL
}

enum class ReasoningVerdictV1 {
    VALID,
    INVALID,
    PARTIALLY_VALID,
    VALID_BUT_NOT_CURRENT_ROUTE,
    UNKNOWN
}

enum class AlternativeMoveRelationV1 {
    SAME_MOVE,
    ALTERNATIVE_MOVE,
    WEAKER_MOVE,
    INVALID_IDEA,
    UNKNOWN
}


//-----------------------------------------------------------------------------
// Tick1 V1: IntentEnvelopeV1 (multi-intent NLU)
// -----------------------------------------------------------------------------

enum class IntentTypeV1 {

    // =========================================================================
    // 1) APP-OWNED AGENDA
    // =========================================================================
    // These intents keep the user on the assistant-driven main road.
    // They either:
    // - answer the assistant's current gate,
    // - move the solving road forward,
    // - stay on the same current stage and elaborate it,
    // - repeat the same current stage,
    // - or move backward within the same solving road rather than leaving it.
    //
    // In other words:
    // these intents do NOT create a true side agenda.
    // They cooperate with the current app-owned solving journey.

    // -------------------------------------------------------------------------
    // 1.1) CONFIRMING PHASE — deterministic answers to app prompts
    // -------------------------------------------------------------------------

    CONFIRM_YES,                    // User accepts the app's prompted proposal / confirmation gate.
    // Purpose: deterministic affirmative reply to the assistant's current CTA.
    // Captures: acceptance of a pending confirmation or branch offered by the assistant.
    // Examples: "yes", "okay go ahead", "let's do it", "sure", "yep"
    // Solving-road semantic when used in SOLVING: usually CONTINUE_FORWARD.

    CONFIRM_NO,                     // User rejects the app's prompted proposal / confirmation gate.
    // Purpose: deterministic negative reply to the assistant's current CTA.
    // Captures: rejection of a pending confirmation or branch offered by the assistant.
    // Examples: "no", "not yet", "don't do that", "wait no", "no thanks"
    // Solving-road semantic when used in SOLVING: usually does not create a detour by itself;
    // it resolves the current gate and may keep the road where it is.

    CONFIRM_GRID_MATCH_EXACT,       // User confirms the whole on-screen grid matches the source puzzle exactly.
    // Purpose: answer the exact-match signoff gate for the captured grid as a whole.
    // Captures: confirmation that the entire displayed grid matches the user's book/paper exactly.
    // Examples: "it matches", "perfect match", "exact match", "yes the screen matches the book", "no issues with the match"
    // Solving-road semantic: deterministic confirming-phase gate answer; must NOT be confused with cell/region as-is confirmation.

    CONFIRM_CELL_AS_IS,             // User confirms a prompted cell is already correct.
    // Purpose: answer a cell-verification prompt from the assistant.
    // Captures: explicit confirmation that a targeted cell should remain unchanged.
    // Examples: "that cell is fine", "leave it as is", "yes that one is correct", "this cell is okay", "keep that cell"
    // Solving-road semantic: deterministic app-gate answer.

    CONFIRM_CELL_TO_DIGIT,          // User supplies the corrected digit for a prompted cell.
    // Purpose: answer a cell-correction prompt from the assistant.
    // Captures: explicit corrected digit bound to the prompted cell.
    // Examples: "make it 7", "that one is 3", "put 9 there", "it's a 5", "change that cell to 4"
    // Solving-road semantic: deterministic app-gate answer.

    CONFIRM_REGION_AS_IS,           // User confirms a prompted region is already correct.
    // Purpose: answer a region-verification prompt from the assistant.
    // Captures: confirmation that a row / column / box is already correct.
    // Examples: "that row is fine", "the box looks correct", "leave that region", "column 3 is okay", "this box is right"
    // Solving-road semantic: deterministic app-gate answer.

    CONFIRM_REGION_TO_DIGITS,       // User supplies corrected digits for a prompted region.
    // Purpose: answer a region-correction prompt from the assistant.
    // Captures: corrected digits for the targeted row / column / box.
    // Examples: "row 2 should be 1 4 7", "those cells are 3 and 8", "replace that box with these digits", "column 6 is 9 then 2", "fix this row to these numbers"
    // Solving-road semantic: deterministic app-gate answer.

    PROVIDE_DIGIT,                  // Bare digit answer bound to an app prompt expecting a digit.
    // Purpose: bind a short numeric reply to the assistant's current pending question.
    // Captures: direct digit-only or near-digit-only answers.
    // Examples: "5", "it's 8", "number 2", "the answer is 6", "just 4"
    // Solving-road semantic: deterministic app-gate answer, usually toward CONTINUE_FORWARD.


    // -------------------------------------------------------------------------
    // 1.2) SOLVING PHASE — app-owned solving rail
    // -------------------------------------------------------------------------
    // These intents live inside the main solving road.
    // They are grouped by SolvingRoadSemanticV2.

    // -------------------------------------------------------------------------
    // 1.2.1) CONTINUE_FORWARD
    // -------------------------------------------------------------------------

    SOLVE_CONTINUE,                 // Continue the app-owned roadmap: stage-to-stage or step-to-step.
    // Purpose: advance the main solving rail forward.
    // Captures: general acceptance of progression on the current solving journey.
    // Examples: "continue", "next", "go on", "keep going", "let's move on"
    // Solving-road semantic: CONTINUE_FORWARD.

    SOLVE_STEP_REVEAL_DIGIT,        // Ask for the answer payload only for the current solving step.
    // Purpose: accelerate the current step toward its answer / resolution payload.
    // Captures: explicit request to reveal the digit now.
    // Examples: "just tell me", "what's the digit", "reveal it", "show me the answer", "what goes there"
    // Solving-road semantic: CONTINUE_FORWARD.

    SOLVE_ACCEPT_LOCK_IN,           // Explicit acceptance of the current resolution / commit boundary.
    // Purpose: confirm that the assistant may place / lock in the current answer.
    // Captures: commit acceptance at the current step's resolution boundary.
    // Examples: "lock it in", "apply it", "go ahead and place it", "put it there", "yes place it"
    // Solving-road semantic: CONTINUE_FORWARD.

    SOLVE_ACCEPT_NEXT_STEP,         // Explicit acceptance to move from current resolution to next step setup.
    // Purpose: advance from completed current step into the next solving step.
    // Captures: explicit request for the next step after resolution.
    // Examples: "next step", "keep going", "show me the next move", "let's continue", "what's next"
    // Solving-road semantic: CONTINUE_FORWARD.


    // -------------------------------------------------------------------------
    // 1.2.2) STAY_AND_ELABORATE
    // -------------------------------------------------------------------------

    SOLVE_PAUSE,                    // Pause forward motion on the current step without advancing.
    // Purpose: stop progression temporarily while staying on the same active solving point.
    // Captures: short requests to hold position on the current step.
    // Examples: "wait", "hold on", "let me think", "pause here", "give me a second"
    // Solving-road semantic: STAY_AND_ELABORATE / stay-put.

    REQUEST_CURRENT_STAGE_ELABORATION, // Ask for a deeper explanation of the stage currently being discussed.
    // Purpose: elaborate the currently active stage without leaving the main road.
    // Captures: clarification of current setup / confrontation / resolution.
    // Examples: "logic", "explain that", "walk me through it", "say more", "why does that help"
    // Solving-road semantic: STAY_AND_ELABORATE.

    REQUEST_CURRENT_TECHNIQUE_EXPLANATION, // Ask to explain the currently active technique in-lane.
    // Purpose: explain the exact technique being used in the current active step.
    // Captures: in-context technique explanation, not a broad side lesson.
    // Examples: "what is this naked pair doing", "how does this technique help here", "explain the technique here", "why is this a naked pair", "what does this pattern mean"
    // Solving-road semantic: STAY_AND_ELABORATE.

    REQUEST_CURRENT_STAGE_COLLAPSE, // Ask for a tighter / simpler restatement of the current stage.
    // Purpose: simplify the current stage without advancing or detouring.
    // Captures: requests for the shorter, simpler, or cleaner version of what was just said.
    // Examples: "short version", "simpler", "say it more simply", "just the key idea", "in one sentence"
    // Solving-road semantic: STAY_AND_ELABORATE.

    REQUEST_CURRENT_STAGE_EXAMPLE,  // Ask for a concrete mini-example tied to the current stage.
    // Purpose: anchor the current stage with a concrete illustration.
    // Captures: request for one more example on the same active step.
    // Examples: "give me an example", "show me with this cell", "make that concrete", "can you illustrate it", "show me exactly how"
    // Solving-road semantic: STAY_AND_ELABORATE.


    // -------------------------------------------------------------------------
    // 1.2.3) REPEAT_CURRENT_STAGE
    // -------------------------------------------------------------------------

    REQUEST_CURRENT_STAGE_REPEAT,   // Replay the same current stage again, rather than deepen it.
    // Purpose: replay the current setup / confrontation / resolution wording.
    // Captures: explicit request to hear the same stage again.
    // Examples: "repeat that", "say that again", "one more time", "go over that again", "repeat the setup"
    // Solving-road semantic: REPEAT_CURRENT_STAGE.

    REQUEST_CURRENT_STAGE_REPHRASE, // Replay the current stage with different wording.
    // Purpose: restate the same current stage in new words.
    // Captures: rephrase requests without moving away from the current stage.
    // Examples: "say it differently", "rephrase that", "another way to put it", "explain that differently", "can you word that another way"
    // Solving-road semantic: REPEAT_CURRENT_STAGE.


    // -------------------------------------------------------------------------
    // 1.2.4) GO_BACKWARD
    // -------------------------------------------------------------------------

    REQUEST_GO_TO_PREVIOUS_STAGE,   // Move back to the immediately previous stage of the current step.
    // Purpose: back up within the same solving step.
    // Captures: requests to go from confrontation -> setup or resolution -> confrontation.
    // Examples: "go back to the setup", "back up one stage", "return to the proof", "back to the explanation", "rewind one stage"
    // Solving-road semantic: GO_BACKWARD.

    REQUEST_GO_TO_PREVIOUS_STEP,    // Move back to the previous solving step entirely.
    // Purpose: revisit the last completed solving step.
    // Captures: requests to move back to the prior app-owned step.
    // Examples: "go back a step", "previous step", "rewind to the last move", "show me the last step again", "back to the prior move"
    // Solving-road semantic: GO_BACKWARD.

    REQUEST_STEP_BACK,              // Request to back up one step or undo a route move.
    // Purpose: broad backward umbrella inside the solving road.
    // Captures: generic backward solving-road movement.
    // Examples: "go back", "step back", "return to the previous step", "rewind", "back up"
    // Solving-road semantic: GO_BACKWARD.


    // =========================================================================
    // 2) USER-OWNED AGENDA
    // =========================================================================
    // These intents assert the passenger's own side agenda.
    // They inspect, challenge, redirect, configure, edit, validate, pause, change modes,
    // or otherwise move away from the straight app-owned solving road.
    //
    // If an intent is truly DETOUR_OR_ROUTE_CONTROL, it belongs here.

    // -------------------------------------------------------------------------
    // 2.1) CONFIRMING PHASE — validation / scan inspection / correction detours
    // -------------------------------------------------------------------------

    CHOOSE_RETAKE,                  // User chooses the retake branch when scan quality is questioned.
    // Purpose: take the retake/rescan fork.
    // Captures: explicit rescan choice.
    // Examples: "retake it", "scan again", "let's redo the photo", "take another picture", "new scan"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    CHOOSE_KEEP_SCAN,               // User chooses to keep the current scan instead of retaking.
    // Purpose: resolve the scan fork by staying with the current scan.
    // Captures: explicit keep-this-scan choice.
    // Examples: "keep this one", "use this scan", "don't retake", "it's fine", "let's keep it"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_STRUCTURAL_VALIDITY,        // Ask if the current grid is structurally valid.
    // Purpose: inspect structural correctness of the current grid.
    // Captures: validation questions about whether the board is valid.
    // Examples: "is this valid", "does the grid make sense", "any structural issue", "is the board okay", "is there a rule violation"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_CONFLICTS_GLOBAL,           // Ask for all conflicts in the board.
    // Purpose: inspect conflicts across the full board.
    // Captures: global error/conflict questions.
    // Examples: "are there conflicts", "what's wrong globally", "show all conflicts", "where are the mistakes", "any clashes"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_CONFLICTS_IN_HOUSE,         // Ask for conflicts in a specific row/col/box.
    // Purpose: inspect local house-level conflicts.
    // Captures: row/column/box conflict inspection.
    // Examples: "conflicts in row 4", "what's wrong in this box", "show column conflicts", "any duplicates in this row", "problem in box 6"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_CONFLICT_DETAILS_CELL,      // Ask why a specific cell is conflicting.
    // Purpose: inspect one problematic cell in detail.
    // Captures: cell-level validation challenge.
    // Examples: "why is r4c5 wrong", "what's wrong with this cell", "explain this conflict", "why is this invalid", "what clashes here"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_DUPLICATES_IN_HOUSE,        // Ask for duplicate digits in a house.
    // Purpose: inspect duplicate-value issues in one house.
    // Captures: duplicate-check questions.
    // Examples: "duplicates in row 7", "is there a repeated number here", "which box has duplicates", "repeated digit in this column", "any duplicates"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_UNRESOLVED_CELLS,           // Ask which cells remain unresolved/problematic.
    // Purpose: inspect unresolved or suspicious cells.
    // Captures: broad unresolved/problem-cell queries.
    // Examples: "which cells are unresolved", "what's still wrong", "where are the problem cells", "what remains unclear", "unsolved trouble spots"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_INVALID_REASON_EXPLAIN,     // Ask why the grid is invalid.
    // Purpose: explain why validation is failing.
    // Captures: "why invalid" style questions.
    // Examples: "why is it invalid", "explain the invalidity", "what makes this wrong", "why does it fail", "what rule is broken"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_IF_RETAKE_NEEDED,           // Ask whether a new scan/photo is needed.
    // Purpose: inspect whether the scan quality requires retake.
    // Captures: retake-necessity questions.
    // Examples: "should I retake", "do we need a new scan", "is the photo bad", "should I scan again", "do we need a better picture"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_SEAL_STATUS,                // Ask whether the scan/validation is sealed/confirmed.
    // Purpose: inspect whether the board has passed the current confirmation gate.
    // Captures: completion/seal questions for confirming.
    // Examples: "is it sealed", "is the scan locked", "are we confirmed", "is this finalized", "is the validation done"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_CONFLICTS_OVERVIEW,         // Broad overview of conflicts.
    // Purpose: get a summary view of all current conflict issues.
    // Captures: overview/summarization of conflicts.
    // Examples: "overview of conflicts", "summarize the conflicts", "what conflicts do we have", "conflict summary", "show the conflict picture"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_VALIDATION_OVERVIEW,        // Broad overview of validation state.
    // Purpose: summarize overall validation health.
    // Captures: high-level validation status requests.
    // Examples: "validation overview", "summarize validation", "how valid is it", "what's the validation status", "give me the validation picture"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_PROBLEM_CELLS_OVERVIEW,     // Broad overview of problematic cells.
    // Purpose: summarize suspicious/problem cells.
    // Captures: overview of cells needing attention.
    // Examples: "show problem cells", "which cells are suspicious", "overview of bad cells", "where are the risky cells", "problem-cell summary"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_MISMATCH_OVERVIEW,          // Overview of mismatches between scan and expected truth.
    // Purpose: inspect mismatch picture between app board and expected board truth.
    // Captures: mismatch summary requests.
    // Examples: "show mismatches", "what doesn't match", "overview of wrong cells", "where is the grid off", "what differs"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_RETAKE_GUIDANCE,            // Ask for guidance about retaking.
    // Purpose: get guidance for rescanning / retaking.
    // Captures: how-to-retake questions.
    // Examples: "how should I retake", "guide me on rescanning", "what do I do for a better scan", "how do I take a better picture", "retake tips"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_WHAT_CHANGED_RECENTLY,      // LEGACY_FALLBACK_ONLY.
    // Purpose: coarse recent-change alias retained for compatibility.
    // Preferred replacement: ASK_WHAT_CHANGED_IN_SCOPE_RECENTLY.

    ASK_WHAT_CHANGED_IN_SCOPE_RECENTLY, // Ask what changed recently in a specific local scope.
    // Purpose: exact scoped recent-change readout.
    // Captures: "what changed in row 1", "did anything change in this cell", "what changed around the target just now" questions.

    ASK_SOURCE_OF_DIGIT,            // LEGACY_FALLBACK_ONLY.
    // Purpose: coarse provenance alias retained for compatibility.
    // Preferred replacement: ASK_SOURCE_OF_DIGIT_IN_CELL_EXACT.

    ASK_SOURCE_OF_DIGIT_IN_CELL_EXACT, // Ask the exact provenance of a specific cell's digit.
    // Purpose: exact cell provenance readout.
    // Captures: "where did the 8 in row 3 column 4 come from", "was this digit scanned, corrected, or solved" questions.

    ASK_OCR_CONFIDENCE_CELL,        // Ask OCR confidence for a specific cell.
    // Purpose: inspect OCR confidence for a specific scanned cell.
    // Captures: confidence/trust questions tied to one cell.
    // Examples: "how sure are you about this cell", "OCR confidence here", "are you certain about r3c7", "how reliable is this scan here", "confidence for this cell"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_OCR_CONFIDENCE_SUMMARY,     // Ask OCR confidence summary.
    // Purpose: inspect overall scan confidence.
    // Captures: broad OCR reliability questions.
    // Examples: "OCR summary", "how reliable is the scan", "confidence overview", "scan confidence", "how certain is the recognition"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_TRUST_OVERVIEW,             // Ask broad trust / certainty overview.
    // Purpose: inspect overall certainty / reliability state.
    // Captures: broad trust questions.
    // Examples: "how much do you trust this", "trust overview", "what are you unsure about", "certainty overview", "what feels risky"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_CELL_TRUST_DETAILS,         // Ask trust details for a specific cell.
    // Purpose: inspect certainty of one cell.
    // Captures: cell-level trust questions.
    // Examples: "why do you trust this cell", "confidence on r2c2", "tell me how sure you are here", "how certain is this cell", "trust details here"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_PROVENANCE_OVERVIEW,        // Ask provenance / origin overview.
    // Purpose: inspect how current grid facts were obtained.
    // Captures: source/origin overview requests.
    // Examples: "where did the values come from", "provenance overview", "show me the sources", "what came from scan vs solver", "origin overview"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    EDIT_CELL,                      // Explicit cell mutation.
    // Purpose: change a specific cell value.
    // Captures: direct edit requests.
    // Examples: "change r4c2 to 7", "edit this cell", "put 3 here", "fix that cell", "replace this number"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    CLEAR_CELL,                     // Clear a cell.
    // Purpose: remove a value from a specific cell.
    // Captures: erase/clear requests.
    // Examples: "clear this cell", "erase r3c4", "remove that number", "blank this cell", "delete that value"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    SWAP_TWO_CELLS,                 // Swap contents of two cells.
    // Purpose: exchange two cell values.
    // Captures: swap requests.
    // Examples: "swap these two", "switch r1c2 and r1c3", "exchange those cells", "flip these values", "trade these cells"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    MASS_CLEAR_SCOPE,               // Clear a region/scope.
    // Purpose: clear many values within a row/column/box/scope.
    // Captures: broad erase requests.
    // Examples: "clear the whole row", "erase this box", "remove all these cells", "clear this region", "wipe that column"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    LOCK_GIVENS_FROM_SCAN,          // Lock givens from the scan.
    // Purpose: mark scanned givens as fixed clues.
    // Captures: givens-locking requests.
    // Examples: "lock the givens", "freeze scanned clues", "mark these as fixed", "make the clues permanent", "lock the original numbers"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    UNDO,                           // Revert last action.
    // Purpose: reverse a recent mutation.
    // Captures: undo requests.
    // Examples: "undo", "go back one change", "revert that", "take that back", "undo the last move"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    REDO,                           // Redo reverted action.
    // Purpose: re-apply a recently undone mutation.
    // Captures: redo requests.
    // Examples: "redo", "do it again", "restore the undone move", "redo that", "bring back the change"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    GRID_EDIT_BATCH,                // Multiple edits at once.
    // Purpose: apply many grid edits in one user action.
    // Captures: batch edit requests.
    // Examples: "set these three cells", "batch edit this row", "apply several fixes", "change all these cells", "do these edits together"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    GRID_MISMATCH_REPORT,           // User reports mismatch between app and real grid.
    // Purpose: challenge the app's board interpretation.
    // Captures: mismatch / "your board is wrong" style reports.
    // Examples: "your grid is wrong", "this doesn't match my puzzle", "the board is off", "that's not my grid", "you read this incorrectly"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.


    // -------------------------------------------------------------------------
    // 2.2) SOLVING PHASE — solving detours, redirects, controls, and challenges
    // -------------------------------------------------------------------------

    REQUEST_SUMMARY_DASHBOARD,      // Ask for a higher-level dashboard or summary view.
    // Purpose: temporarily shift from current step narration to a broader summary view.
    // Captures: summary/dashboard requests.
    // Examples: "give me a summary", "show me the dashboard", "where do we stand", "summarize everything", "overall view please"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    REQUEST_FOCUS_ON_ISSUE_TYPE,    // Redirect attention to a kind of issue.
    // Purpose: redirect the assistant toward a specific issue class.
    // Captures: issue-type focus shifts.
    // Examples: "focus on conflicts", "show the mismatches", "let's look at trust issues", "focus on candidates", "only show problems"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    REQUEST_FOCUS_ON_AREA,          // Redirect attention to a region/area.
    // Purpose: redirect the assistant toward a different board area.
    // Captures: area/region focus shifts.
    // Examples: "focus on row 4", "look at the top-left box", "show me column 8", "let's inspect this box", "move focus there"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    REQUEST_FAST_MODE,              // Preference for shorter / faster guidance.
    // Purpose: change the conversational pacing/style.
    // Captures: speed-up requests.
    // Examples: "go faster", "be quick", "use fast mode", "less talking", "just the essentials"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    REQUEST_TEACH_MODE,             // Preference for more educational guidance.
    // Purpose: change the conversational style toward teaching.
    // Captures: more-explanation / coaching style requests.
    // Examples: "teach me more", "explain as we go", "use teach mode", "be more educational", "I want to learn"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    REQUEST_MODE_CHANGE,            // Generic request to change app mode.
    // Purpose: change high-level assistant/app mode.
    // Captures: mode-change requests.
    // Examples: "change mode", "switch modes", "use a different mode", "move to another mode", "switch this mode"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    REQUEST_SOLVE_STAGE,            // Ask to change the solve-stage focus.
    // Purpose: move between broad phases like validating vs solving.
    // Captures: stage/phase shift requests.
    // Examples: "go back to validating", "move to solving", "switch stage", "change phase", "let's validate first"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    REQUEST_REVALIDATE,             // Request to validate again.
    // Purpose: temporarily leave current solving flow and re-run validation.
    // Captures: revalidation requests.
    // Examples: "check it again", "revalidate", "validate once more", "run validation again", "double-check the grid"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    REQUEST_FOCUS_CHANGE,           // Generic request to change focus.
    // Purpose: broadly redirect assistant attention.
    // Captures: focus-shift requests not specific enough for a narrower intent.
    // Examples: "change focus", "look somewhere else", "switch target", "focus elsewhere", "move focus"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    REQUEST_HINT_LEVEL,             // Request to change hint depth/granularity.
    // Purpose: alter the strength/detail of hints.
    // Captures: hint-strength preferences.
    // Examples: "smaller hints", "give me bigger hints", "change hint level", "more subtle hints", "stronger hints"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    REQUEST_ONLY_VALIDATE,          // Request to stay only in validation.
    // Purpose: constrain the assistant to validation tasks.
    // Captures: validation-only workflow requests.
    // Examples: "just validate", "don't solve yet", "only check the grid", "validation only", "just inspect it"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    REQUEST_ONLY_SOLVE,             // Request to stay only in solving.
    // Purpose: constrain the assistant to solving tasks.
    // Captures: solve-only workflow requests.
    // Examples: "just solve", "skip validation", "only do solving", "go straight to solving", "solve only"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    PAUSE_ASSISTANT,                // Ask assistant to stop speaking / stop guiding temporarily.
    // Purpose: temporarily suspend the assistant's active guidance.
    // Captures: pause/stop requests.
    // Examples: "pause", "stop for a second", "be quiet", "hold off", "stop talking"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    RESUME_ASSISTANT,               // Ask assistant to resume after pause.
    // Purpose: resume assistant output after a prior pause.
    // Captures: resume/start-again requests.
    // Examples: "resume", "continue talking", "come back", "start again", "pick it up"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    SET_LANGUAGE,                   // Change spoken language.
    // Purpose: reconfigure assistant language.
    // Captures: language-change requests.
    // Examples: "speak French", "switch to English", "change language", "use Arabic", "talk in French"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    SET_ONE_QUESTION_MAX,           // Change one-question-at-a-time preference.
    // Purpose: reconfigure turn density / number of asks per turn.
    // Captures: one-question-only preferences.
    // Examples: "one question at a time", "ask only one thing", "single question mode", "one prompt only", "don't stack questions"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    SET_EVIDENCE_VERBOSITY,         // Change how detailed the proofs should be.
    // Purpose: reconfigure explanation verbosity.
    // Captures: proof/detail-level preferences.
    // Examples: "be more detailed", "less proof please", "show more evidence", "keep it concise", "more detail"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    SET_NOTATION_STYLE,             // Change coordinate / notation style.
    // Purpose: reconfigure coordinate vocabulary.
    // Captures: notation-style preferences.
    // Examples: "use r1c1 notation", "say row and column", "change notation style", "use plain coordinates", "different notation"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    SET_DIGIT_STYLE,                // Change how digits are spoken / formatted.
    // Purpose: reconfigure number-speaking format.
    // Captures: digit-style preferences.
    // Examples: "say the digits differently", "change digit style", "use another digit format", "read digits one by one", "change the number style"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_SOLVABILITY_STATUS,         // Ask whether the puzzle is solvable / current state solvable.
    // Purpose: inspect solvability rather than advance current step.
    // Captures: solvability questions.
    // Examples: "is this solvable", "can this still be solved", "are we stuck permanently", "is there a valid solution", "does this still work"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_PROGRESS_METRICS,           // Ask progress status.
    // Purpose: inspect progress rather than continue the step.
    // Captures: progress/remaining-work questions.
    // Examples: "how far are we", "what's our progress", "how much is left", "how many steps left", "what percent done"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_IF_STUCK,                   // Ask whether puzzle is stuck.
    // Purpose: inspect whether forward logical progress exists.
    // Captures: stuckness questions.
    // Examples: "are we stuck", "is there no move", "do we have a next move", "is this blocked", "can we continue logically"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_IF_GUESS_REQUIRED,          // Ask whether guessing is required.
    // Purpose: inspect whether logic-only solving remains possible.
    // Captures: guess-vs-logic questions.
    // Examples: "do we need to guess", "is this forced or guessy", "must we speculate", "can this be solved logically", "is guessing necessary"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_DIFFICULTY_ESTIMATE,        // Ask difficulty level.
    // Purpose: inspect puzzle difficulty rather than progress current step.
    // Captures: difficulty questions.
    // Examples: "how hard is this", "difficulty estimate", "what level puzzle is this", "easy or hard", "what difficulty are we in"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_TECHNIQUES_NEEDED,          // Ask which techniques are needed overall.
    // Purpose: inspect the overall technique set for the puzzle.
    // Captures: broad strategy/tool questions.
    // Examples: "what techniques will we need", "which methods solve this", "what patterns are involved", "what tools do we need", "what strategies appear"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_SOLVING_OVERVIEW,           // Broad overview of solving state / path.
    // Purpose: step out of the current micro-step and inspect the broader solve.
    // Captures: solving-overview questions.
    // Examples: "give me the solving overview", "what is the plan", "how do we solve from here", "what's the road ahead", "overview of the solve"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_TECHNIQUE_OVERVIEW,         // Ask about the technique currently in play or a named technique.
    // Purpose: broad technique lesson or overview, not necessarily limited to current stage wording.
    // Captures: broad technique explanation requests.
    // Examples: "what is a naked pair", "remind me of the technique", "how does this method work", "technique overview", "what pattern is this"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.
    // Note: if the request is clearly tied to the current active stage, prefer REQUEST_CURRENT_TECHNIQUE_EXPLANATION instead.

    ASK_STUCK_HELP,                 // Ask for help when stuck.
    // Purpose: request an alternate assistance mode due to stuckness.
    // Captures: help-me-now solving detours.
    // Examples: "help, I'm stuck", "what should I look at", "give me guidance", "what now", "where do I go from here"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_ADVANCED_PATTERN_HELP,      // Ask for advanced technique help.
    // Purpose: inspect advanced pattern possibilities outside the current straight lane.
    // Captures: advanced-technique fishing questions.
    // Examples: "is there an x-wing", "any advanced pattern here", "do we need something harder", "show advanced options", "is this an advanced puzzle now"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_HIDDEN_SINGLE_LOCATIONS,    // Ask where hidden singles are.
    // Purpose: inspect locations of a named technique.
    // Captures: "find hidden singles" requests.
    // Examples: "where is the hidden single", "show hidden singles", "any hidden single here", "find hidden singles", "which cell is a hidden single"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_NAKED_SINGLE_LOCATIONS,     // Ask where naked singles are.
    // Purpose: inspect naked-single locations.
    // Captures: "find naked singles" requests.
    // Examples: "where are the naked singles", "show easy singles", "any one-candidate cells", "find naked singles", "which cells have one option"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_NAKED_PAIR_LOCATIONS,       // Ask where naked pairs are.
    // Purpose: inspect naked-pair locations.
    // Captures: "find naked pairs" requests.
    // Examples: "where is the naked pair", "show naked pairs", "any pair pattern", "find a naked pair", "which two cells form the pair"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_HIDDEN_PAIR_LOCATIONS,      // Ask where hidden pairs are.
    // Purpose: inspect hidden-pair locations.
    // Captures: "find hidden pairs" requests.
    // Examples: "where is a hidden pair", "show hidden pairs", "any hidden pair here", "find hidden pairs", "which cells make the hidden pair"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_POINTING_PAIR_TRIPLE,       // Ask about pointing pair/triple situations.
    // Purpose: inspect pointing interactions.
    // Captures: pointing-pair/pointing-triple discovery questions.
    // Examples: "any pointing pair", "do we have a pointing triple", "show box-line pointing", "find pointing pairs", "is there a pointing pattern"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_BOX_LINE_REDUCTION,         // Ask about box-line reduction.
    // Purpose: inspect box-line interactions.
    // Captures: claiming/box-line discovery questions.
    // Examples: "any box-line reduction", "line-box interaction here", "do we have claiming", "any box-line claim", "find a box-line reduction"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_XWING_CANDIDATE,            // Ask whether an x-wing candidate exists.
    // Purpose: inspect possible x-wing opportunities.
    // Captures: x-wing discovery questions.
    // Examples: "is there an x-wing", "show x-wing", "any x-wing pattern", "find an x-wing", "could this be x-wing"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    SOLVE_STEP_WHY_NOT_DIGIT,       // LEGACY_FALLBACK_ONLY.
    // Purpose: legacy coarse alias for local why-not-digit questions tied to the current step.
    // Preferred new intents:
    // - ASK_WHY_NOT_DIGIT_IN_CELL
    // - ASK_WHAT_BLOCKS_DIGIT_IN_HOUSE
    // Solving-road semantic today in compatibility code: often proof-challenge flavored.

    SOLVE_SKIP_N_STEPS,             // Request to auto-skip multiple solve steps.
    // Purpose: temporarily change pacing and jump ahead.
    // Captures: fast-forward requests.
    // Examples: "skip 3 steps", "jump ahead", "solve a few moves for me", "advance a bit", "fast-forward five moves"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    WALKBACK_REQUEST,               // Ask to walk back reasoning / previous moves.
    // Purpose: request backward review / historical reconstruction of reasoning.
    // Captures: walkback/rewind-the-logic requests.
    // Examples: "walk that back", "show how we got here", "trace the logic backward", "rewind the reasoning", "go back through the logic"
    // Solving-road semantic: usually DETOUR_OR_ROUTE_CONTROL as a broad review request.
    // It may later be split more finely into in-lane backward vs detour review.


    // -------------------------------------------------------------------------
    // 2.2.1) SOLVING / GRID TALK — specific proof / target / verdict intents
    // -------------------------------------------------------------------------

    ASK_WHY_NOT_DIGIT_IN_CELL,      // Ask why a specific digit cannot go in a specific cell.
    // Purpose: explicit local proof challenge.
    // Captures: cell+digit "why not" questions.
    // Examples: "why can't r1c6 be 5", "why not 7 in this cell", "what blocks 3 from row 4 column 2", "why is 9 impossible here"

    ASK_WHY_DIGIT_IN_CELL,          // Ask why a specific digit can go in a specific cell.
    // Purpose: explicit local proof / support question.
    // Captures: cell+digit "why yes" questions.
    // Examples: "why can r1c6 be 2", "what makes 4 valid here", "why does this cell allow 8", "why is 6 still possible"

    ASK_WHY_THIS_CELL_IS_TARGET_FOR_DIGIT, // Ask why a specific cell is the current target/home for a digit.
    // Purpose: target-cell justification.
    // Captures: "why this cell for digit X" questions.
    // Examples: "why are we focusing on r1c6 for 2", "why this square for 7", "what makes this the live target"

    ASK_WHY_THIS_CELL_NOT_OTHER_CELL, // Ask why one cell is chosen over another.
    // Purpose: target-vs-rival comparison.
    // Captures: "why this cell, not that one" questions.
    // Examples: "why r1c6 and not r1c3", "why this square instead of the one next to it", "why did this one survive"

    ASK_WHY_THIS_TECHNIQUE_APPLIES_HERE, // Ask why the named/current technique applies here.
    // Purpose: technique-fit justification.
    // Captures: "why is this a hidden single / naked pair / x-wing" questions.

    ASK_WHY_CURRENT_MOVE_BEFORE_OTHER_MOVE, // Ask why the current move/route is preferred before another.
    // Purpose: route-order justification.
    // Captures: "why this move first", "why now", "why before checking another idea" questions.

    ASK_WHAT_BLOCKS_DIGIT_IN_HOUSE, // Ask what blocks a digit in a row/col/box.
    // Purpose: house-scoped blocking proof.
    // Captures: "what blocks 2 in row 1", "why can't 7 go anywhere else in this box" questions.

    ASK_ONLY_PLACE_FOR_DIGIT_IN_HOUSE, // Ask whether a cell is the only place left for a digit in a house.
    // Purpose: exact target/uniqueness question.
    // Captures: "is r1c6 the only place left for 2 in row 1" questions.

    ASK_DIGIT_LOCATIONS_IN_HOUSE_EXACT, // Ask for the exact remaining locations of a digit in a specific house.
    // Purpose: exact digit-location readout.
    // Captures: "which cells in row 1 can still take 2", "where can 7 go in column 5" questions.

    ASK_NEXT_MOVE_EXACT,            // Ask for the exact next move.
    // Purpose: exact next-step request rather than broad solving overview.
    // Captures: "what's the next move", "which cell next", "what do we solve now" questions.

    CHECK_PROPOSED_DIGIT_IN_CELL,   // Ask whether a proposed digit in a cell is valid.
    // Purpose: explicit verdict on a user proposal.
    // Captures: "I think r1c6 is 2, am I right" questions.

    CHECK_PROPOSED_CANDIDATE_SET_IN_CELL, // Ask whether a proposed candidate set in a cell is valid.
    // Purpose: candidate-set verdict.
    // Captures: "I think r1c6 could still be 4 or 5, am I right" questions.

    CHECK_PROPOSED_ELIMINATION_IN_SCOPE, // Ask whether a proposed elimination is valid.
    // Purpose: elimination verdict.
    // Captures: "can we remove 5 from r1c6", "is 3 eliminated here" questions.

    CHECK_PROPOSED_TECHNIQUE_APPLIES_HERE, // Ask whether a proposed technique classification is valid.
    // Purpose: technique-verdict question.
    // Captures: "is this really a hidden single", "is this a naked pair" questions.

    CHECK_PROPOSED_ROUTE_EQUIVALENCE, // Ask whether the user's route is equivalent to the assistant's route.
    // Purpose: route-equivalence verdict.
    // Captures: "is my logic basically the same as yours", "does checking the box reach the same move" questions.

    ASK_ALTERNATIVE_TECHNIQUE_FOR_CURRENT_SPOT, // Ask whether another technique could solve the current spot.
    // Purpose: alternative-technique comparison.
    // Captures: "could this be solved as a naked single instead", "is there another technique here" questions.

    ASK_OTHER_LOCAL_MOVE_EXISTS,    // Ask whether another local move exists nearby.
    // Purpose: bounded local move search.
    // Captures: "is there any other move nearby", "any other local move besides this one" questions.

    ASK_COMPARE_CURRENT_ROUTE_WITH_ALTERNATIVE_ROUTE, // Ask how the current route compares with an alternative route.
    // Purpose: route-comparison question.
    // Captures: "is your route the same as checking the box first", "how does your route compare with mine" questions.

    REQUEST_EXPLANATION,            // LEGACY_FALLBACK_ONLY.
    // Purpose: broad explanation umbrella retained for compatibility.
    // Preferred replacements:
    // - ASK_WHY_NOT_DIGIT_IN_CELL
    // - ASK_WHY_DIGIT_IN_CELL
    // - ASK_WHY_THIS_CELL_IS_TARGET_FOR_DIGIT
    // - ASK_WHY_THIS_CELL_NOT_OTHER_CELL
    // - ASK_WHY_THIS_TECHNIQUE_APPLIES_HERE
    // - ASK_WHY_CURRENT_MOVE_BEFORE_OTHER_MOVE
    // - ASK_WHAT_BLOCKS_DIGIT_IN_HOUSE

    REQUEST_REASONING_CHECK,        // LEGACY_FALLBACK_ONLY.
    // Purpose: broad reasoning-check umbrella retained for compatibility.
    // Preferred replacements:
    // - CHECK_PROPOSED_DIGIT_IN_CELL
    // - CHECK_PROPOSED_CANDIDATE_SET_IN_CELL
    // - CHECK_PROPOSED_ELIMINATION_IN_SCOPE
    // - CHECK_PROPOSED_TECHNIQUE_APPLIES_HERE
    // - CHECK_PROPOSED_ROUTE_EQUIVALENCE

    ASK_WHY_THIS_CELL,              // LEGACY_FALLBACK_ONLY.
    // Purpose: coarse target-cell umbrella retained for compatibility.
    // Preferred replacements:
    // - ASK_WHY_THIS_CELL_IS_TARGET_FOR_DIGIT
    // - ASK_WHY_THIS_CELL_NOT_OTHER_CELL

    REPORT_BUG_OR_WRONG_ASSERTION,  // User says the assistant is wrong.
    // Purpose: challenge correctness of the assistant's statement or logic.
    // Captures: bug/incorrectness claims.
    // Examples: "that's wrong", "bug", "your logic is incorrect", "I don't think that's true", "that seems mistaken"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.


    // -------------------------------------------------------------------------
    // 2.3) CONFIRMING OR SOLVING — grid inspection / candidates / navigation
    // -------------------------------------------------------------------------

    ASK_CELL_VALUE,                 // LEGACY_FALLBACK_ONLY.
    // Purpose: coarse cell-value alias retained for compatibility.
    // Preferred replacement: ASK_CELL_VALUE_EXACT.

    ASK_CELL_STATUS,                // LEGACY_FALLBACK_ONLY.
    // Purpose: coarse cell-status umbrella retained for compatibility.
    // Preferred replacements:
    // - ASK_CELL_VALUE_EXACT
    // - ASK_CELL_CANDIDATES_EXACT
    // - ASK_CELL_CANDIDATE_COUNT_EXACT
    // - ASK_ONLY_PLACE_FOR_DIGIT_IN_HOUSE
    // - ASK_WHY_THIS_CELL_IS_TARGET_FOR_DIGIT
    // - ASK_SOURCE_OF_DIGIT_IN_CELL_EXACT

    ASK_CELL_VALUE_EXACT,           // Ask the exact current value of a specific cell.
    // Purpose: exact cell-value readout.
    // Captures: "what digit is in r5c8", "what is in this cell right now" questions.

    ASK_CELL_CANDIDATES_EXACT,      // Ask the exact candidate set of a specific cell.
    // Purpose: exact candidate readout.
    // Captures: "what candidates does r1c3 have", "what pencil marks are left here" questions.

    ASK_CELL_CANDIDATE_COUNT_EXACT, // Ask how many candidates remain in a specific cell.
    // Purpose: candidate-count readout.
    // Captures: "how many candidates are left here", "is this cell down to two options" questions.

    ASK_COMPARE_CANDIDATES_BETWEEN_CELLS, // Ask to compare candidates between two cells.
    // Purpose: side-by-side candidate comparison.
    // Captures: "what candidates does r1c3 have compared with r1c6" questions.

    ASK_ROW_CONTENTS,               // Ask contents of a row.
    // Purpose: inspect row contents.
    // Captures: row-readout requests.
    // Examples: "what's in row 5", "read row 2", "show row contents", "tell me row 8", "what numbers are in this row"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_COL_CONTENTS,               // Ask contents of a column.
    // Purpose: inspect column contents.
    // Captures: column-readout requests.
    // Examples: "what's in column 7", "read this column", "show col contents", "tell me column 3", "what is in this column"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_BOX_CONTENTS,               // Ask contents of a box.
    // Purpose: inspect box contents.
    // Captures: box-readout requests.
    // Examples: "what's in box 3", "show this box", "read the bottom-left box", "tell me box 7", "what numbers are in this box"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_HOUSE_COMPLETION,           // Ask house completion status.
    // Purpose: inspect progress of one house.
    // Captures: completion-status requests for row/column/box.
    // Examples: "how complete is row 4", "is this box nearly done", "completion of column 2", "how full is this row", "what's the completion here"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_HOUSE_MISSING_DIGITS,       // Ask missing digits in a house.
    // Purpose: inspect the missing-number set of one house.
    // Captures: missing-digit requests.
    // Examples: "what's missing in row 6", "missing digits here", "which numbers are absent in this box", "what digits are left", "what is missing in column 4"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_DIGIT_LOCATIONS,            // LEGACY_FALLBACK_ONLY.
    // Purpose: coarse digit-location alias retained for compatibility.
    // Preferred replacement: ASK_DIGIT_LOCATIONS_IN_HOUSE_EXACT.

    ASK_DIGIT_COUNT_GLOBAL,         // Ask count of a digit globally.
    // Purpose: inspect global frequency of a digit.
    // Captures: digit-count requests.
    // Examples: "how many 7s are placed", "count digit 3", "how many 9s do we have", "number of 2s", "how many 5s are on the board"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_DIGIT_COUNT_IN_HOUSE,       // Ask count of a digit in a house.
    // Purpose: inspect house-local frequency of a digit.
    // Captures: house-specific digit-count requests.
    // Examples: "how many 4s in this row", "count 2s in box 8", "number of 6s in this column", "how many 9s in row 1", "digit count in this box"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_EMPTY_HOUSES,               // Ask which houses are empty / emptiest.
    // Purpose: inspect emptiness distribution.
    // Captures: empty-house requests.
    // Examples: "which rows are empty", "any empty boxes", "show empty houses", "which areas have nothing", "empty regions"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_NEARLY_COMPLETE_HOUSES,     // Ask which houses are nearly complete.
    // Purpose: inspect near-finished houses.
    // Captures: nearly-complete-house requests.
    // Examples: "which houses are nearly done", "almost complete rows", "show near-finished boxes", "which column is closest", "near completion"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_HOUSE_OVERVIEW,             // Broad overview of a house.
    // Purpose: inspect one house at a summary level.
    // Captures: broad house-summary requests.
    // Examples: "overview of this row", "summarize this box", "tell me about this column", "house overview", "what's going on in this row"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_HOUSE_RANKING,              // Rank houses by completion or promise.
    // Purpose: inspect comparative promise/completion of houses.
    // Captures: house-ranking requests.
    // Examples: "rank the rows by completeness", "which box is best", "what area is closest to done", "best house to work on", "house ranking"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_DIGIT_OVERVIEW,             // Overview of one digit across the grid.
    // Purpose: inspect one digit globally.
    // Captures: broad digit-summary requests.
    // Examples: "overview of digit 5", "tell me about the 7s", "digit distribution overview", "what about digit 9", "where does 4 stand"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_DIGIT_IN_HOUSE_OVERVIEW,    // Overview of one digit inside one house.
    // Purpose: inspect one digit locally within one house.
    // Captures: local digit-summary requests.
    // Examples: "overview of digit 6 in row 2", "tell me about 9 in this box", "digit in-house summary", "what about 3 in this column", "digit overview here"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_GRID_CONTENTS_OVERVIEW,     // Overview of the board contents.
    // Purpose: inspect the board at a high level.
    // Captures: board-summary requests.
    // Examples: "overview of the grid", "read the board", "show the board contents", "summarize the puzzle", "what's on the board"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_COMPLETE_HOUSES_COUNT,      // Ask how many houses are complete.
    // Purpose: inspect solved-house count.
    // Captures: complete-house count requests.
    // Examples: "how many houses are complete", "count complete rows", "number of finished boxes", "how many solved houses", "completed-house count"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_COMPLETE_HOUSES_LIST,       // Ask which houses are complete.
    // Purpose: inspect which houses are fully solved.
    // Captures: complete-house listing requests.
    // Examples: "which houses are complete", "list finished rows", "show solved boxes", "which rows are complete", "completed-house list"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_HOUSES_COMPLETION_RANKING,  // Rank houses by completion.
    // Purpose: inspect comparative completion ranking.
    // Captures: completion-ranking requests.
    // Examples: "rank all houses", "which house is closest", "completion ranking", "house completion order", "which house is best next"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_CELL_CANDIDATES,            // Ask candidates of one cell.
    // Purpose: inspect candidate set of a cell.
    // Captures: candidate-readout requests.
    // Examples: "what are the candidates here", "possible digits for r2c8", "what can go in this cell", "pencil marks here", "options for this cell"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_CANDIDATE_COUNT_CELL,       // Ask candidate count of one cell.
    // Purpose: inspect how many candidates a cell has.
    // Captures: candidate-count requests.
    // Examples: "how many candidates here", "candidate count in this cell", "number of options for r5c1", "how many choices here", "count the candidates"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_CELLS_WITH_N_CANDIDATES,    // Ask for cells with a given candidate count.
    // Purpose: inspect grid-wide candidate-count distribution.
    // Captures: "show cells with N candidates" requests.
    // Examples: "which cells have two candidates", "show bivalue cells", "cells with 3 options", "find all two-candidate cells", "which cells have one option"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_HOUSE_CANDIDATE_MAP,        // LEGACY_FALLBACK_ONLY.
    // Purpose: coarse house-candidate-map alias retained for compatibility.
    // Preferred replacement: ASK_HOUSE_CANDIDATE_MAP_EXACT.

    ASK_HOUSE_CANDIDATE_MAP_EXACT,  // Ask the exact candidate map for a specific row/col/box.
    // Purpose: exact house-level candidate layout.
    // Captures: "show me the candidate map for row 1", "which cells in this box take which digits" questions.

    ASK_CANDIDATE_FREQUENCY,        // Ask candidate frequency distribution.
    // Purpose: inspect how often candidates appear.
    // Captures: candidate-frequency requests.
    // Examples: "how frequent is digit 8 as a candidate", "candidate frequency", "which candidate appears most", "how often does 4 appear", "frequency of candidates"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_CANDIDATES_OVERVIEW,        // Broad candidate overview.
    // Purpose: inspect the candidate landscape broadly.
    // Captures: broad candidate-summary requests.
    // Examples: "candidate overview", "summarize all candidates", "tell me about the pencil marks", "what's the candidate picture", "candidate summary"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_CANDIDATES_CELL_OVERVIEW,   // Candidate overview of one cell.
    // Purpose: inspect a cell's options at a summary level.
    // Captures: cell-candidate-summary requests.
    // Examples: "candidate overview for this cell", "summarize this cell's options", "tell me the pencil marks here", "overview of this cell's candidates", "what are this cell's options"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_CANDIDATES_DISTRIBUTION,    // Distribution of candidates across the grid/house.
    // Purpose: inspect candidate spread globally or locally.
    // Captures: candidate-distribution requests.
    // Examples: "distribution of candidates", "where are the 5 candidates", "candidate spread", "how are candidates distributed", "spread of the pencil marks"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    CAND_ADD,                       // Add one or more candidates.
    // Purpose: mutate candidate set by adding marks.
    // Captures: candidate-add requests.
    // Examples: "add candidate 4 here", "mark 7 as possible", "put pencil 2", "add 3 to the candidates", "include 8 here"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    CAND_REMOVE,                    // Remove one or more candidates.
    // Purpose: mutate candidate set by removing marks.
    // Captures: candidate-remove requests.
    // Examples: "remove candidate 3", "erase 5 from here", "drop that pencil mark", "take out 7", "remove 2 as an option"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    CAND_SET,                       // Fully set candidate list for a cell.
    // Purpose: overwrite candidate set explicitly.
    // Captures: candidate-reset requests.
    // Examples: "set candidates to 2 and 8", "make these the only options", "replace the pencil marks", "set this cell to 1 4 9", "reset the candidates to these"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    SWITCH_FOCUS_CELL,              // Move focus to a different cell.
    // Purpose: redirect attention to another cell.
    // Captures: cell-focus redirection requests.
    // Examples: "focus on r6c2", "look at this cell", "switch target cell", "move focus here", "let's inspect this cell"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    SWITCH_FOCUS_REGION,            // Move focus to a different region.
    // Purpose: redirect attention to another row/column/box.
    // Captures: region-focus redirection requests.
    // Examples: "focus on row 3", "look at this box", "switch to column 9", "move focus to this region", "inspect this box"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_UI_LEGEND,                  // Ask about visual legend / statuses.
    // Purpose: inspect UI semantics rather than solving content.
    // Captures: color/highlight/icon meaning questions.
    // Examples: "what do these colors mean", "legend please", "what does the border mean", "what does this highlight mean", "what is this icon"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_HOW_TO_LOCATE_CELL,         // Ask how to find a cell by coordinates.
    // Purpose: inspect coordinate-navigation rules.
    // Captures: "how do I locate this cell" questions.
    // Examples: "how do I find r4c7", "how do coordinates work", "which one is row 8 column 1", "how do I locate this cell", "where is r2c5"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_BOX_INDEX_MAPPING,          // Ask how box indices map to the grid.
    // Purpose: inspect box-numbering convention.
    // Captures: "which box is box N" questions.
    // Examples: "which box is box 7", "how are boxes numbered", "box index mapping", "which one is box 3", "how do you count the boxes"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_COORDINATE_TRANSLATION,     // Ask to translate coordinate notation.
    // Purpose: inspect or convert coordinate vocabulary.
    // Captures: coordinate-translation requests.
    // Examples: "translate row 2 column 5", "what is r3c9 in plain words", "coordinate translation", "say r7c2 normally", "what does r5c4 mean"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_WHAT_DO_YOU_MEAN_BY_HOUSE,  // Ask meaning of Sudoku terminology like house.
    // Purpose: inspect Sudoku vocabulary/terminology.
    // Captures: terminology-definition requests.
    // Examples: "what is a house", "what do you mean by house", "define house", "what is a Sudoku house", "house means row or box?"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_UI_HELP,                    // Broad UI help request.
    // Purpose: ask how to use the interface.
    // Captures: broad UI-help requests.
    // Examples: "help with the UI", "what does this screen do", "explain the interface", "UI help", "how do I use this screen"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.

    ASK_COORDINATES_HELP,           // Broad coordinate help request.
    // Purpose: ask for help understanding board coordinates.
    // Captures: broad coordinate-help requests.
    // Examples: "help with coordinates", "I don't understand r1c1", "coordinate help", "how do rows and columns work", "teach me the coordinates"
    // Solving-road semantic: DETOUR_OR_ROUTE_CONTROL.


    // =========================================================================
    // 3) APP-OR-USER-OWNED
    // =========================================================================
    // These intents are genuinely mixed.
    // Ownership depends on context, phase, and surrounding pending/agenda.

    ASK_WHAT_YOU_KNOW_NOW,          // Meta question about current known state.
    // Purpose: ask the assistant for its current state understanding.
    // Captures: state-summary meta questions.
    // Examples: "what do you know now", "what's your current view", "summarize your state", "what's your current understanding", "what are you tracking"
    // Solving-road semantic: context-sensitive; often defaults to forward-safe handling unless a stronger route signal wins.

    ASK_WHAT_YOU_DONT_KNOW,         // Meta question about uncertainty.
    // Purpose: ask the assistant what remains unknown or uncertain.
    // Captures: uncertainty meta questions.
    // Examples: "what don't you know", "what are you unsure about", "what is missing", "where are the uncertainties", "what are you not certain about"
    // Solving-road semantic: context-sensitive.

    REQUEST_EXPORT_STATE,           // Request to export/share current internal state.
    // Purpose: ask the system to produce an exportable state representation.
    // Captures: export/share state requests.
    // Examples: "export the state", "save this session", "give me the current state", "dump the state", "export this puzzle state"
    // Solving-road semantic: typically DETOUR_OR_ROUTE_CONTROL in practice, though ownership can vary.

    CAPABILITY_CHECK,               // Check whether assistant/app can do something.
    // Purpose: test assistant capability or availability.
    // Captures: "can you..." checks.
    // Examples: "can you hear me", "can you do candidates", "are you able to explain", "can you help with x-wing", "do you understand the grid"
    // Solving-road semantic: often safe/non-blocking and may default toward CONTINUE_FORWARD behavior.

    META_APP_QUESTION,              // Ask about the app/flow rather than the Sudoku fact itself.
    // Purpose: ask about mission state, process, or flow.
    // Captures: app-process meta questions.
    // Examples: "what are we doing", "what's next", "how does this app work", "where are we in the process", "what stage is this"
    // Solving-road semantic: usually non-blocking; can be handled without opening a full detour.

    SMALL_TALK,                     // Short social filler / acknowledgment.
    // Purpose: allow lightweight social continuity without breaking the route.
    // Captures: acknowledgments / brief social replies.
    // Examples: "nice", "cool", "got it", "okay", "great"
    // Solving-road semantic: often treated as forward-safe / non-blocking.

    FREE_TALK,                      // Open conversational detour or non-grid chat.
    // Purpose: allow true free conversation outside grid-solving.
    // Captures: non-grid conversational turns.
    // Examples: "tell me a joke", "how's it going", "let's chat", "talk to me", "something unrelated"
    // Solving-road semantic: functionally DETOUR_OR_ROUTE_CONTROL, though special conductor handling may wrap it safely.

    UNKNOWN                         // Parse produced an intent shell but no supported meaning was identified.
    // Purpose: fallback bucket when no clearer supported intent wins.
    // Captures: vague / underspecified / uninterpretable utterances.
    // Examples: "hmm", "uh okay", "right", "mmm", "well"
    // Solving-road semantic in current compatibility behavior: usually defaults to CONTINUE_FORWARD
    // in SOLVING if no stronger semantic wins.
}

enum class AgendaIntentConstitutionBucketV1 {
    DIRECT_APP_OWNED,
    USER_DETOUR,
    USER_ROUTE_JUMP,
    REPAIR_CANDIDATE,
    NONE
}

fun agendaIntentConstitutionBucketV1(t: IntentTypeV1): AgendaIntentConstitutionBucketV1 =
    when (t) {
        IntentTypeV1.CONFIRM_YES,
        IntentTypeV1.CONFIRM_NO,
        IntentTypeV1.CONFIRM_GRID_MATCH_EXACT,
        IntentTypeV1.CONFIRM_CELL_AS_IS,
        IntentTypeV1.CONFIRM_CELL_TO_DIGIT,
        IntentTypeV1.CONFIRM_REGION_AS_IS,
        IntentTypeV1.CONFIRM_REGION_TO_DIGITS,
        IntentTypeV1.PROVIDE_DIGIT,
        IntentTypeV1.SOLVE_CONTINUE,
        IntentTypeV1.SOLVE_STEP_REVEAL_DIGIT,
        IntentTypeV1.SOLVE_ACCEPT_LOCK_IN,
        IntentTypeV1.SOLVE_ACCEPT_NEXT_STEP,
        IntentTypeV1.EDIT_CELL,
        IntentTypeV1.CLEAR_CELL,
        IntentTypeV1.SWAP_TWO_CELLS,
        IntentTypeV1.MASS_CLEAR_SCOPE,
        IntentTypeV1.LOCK_GIVENS_FROM_SCAN,
        IntentTypeV1.UNDO,
        IntentTypeV1.REDO,
        IntentTypeV1.GRID_EDIT_BATCH,
        IntentTypeV1.CAND_ADD,
        IntentTypeV1.CAND_REMOVE,
        IntentTypeV1.CAND_SET ->
            AgendaIntentConstitutionBucketV1.DIRECT_APP_OWNED

        IntentTypeV1.GRID_MISMATCH_REPORT ->
            AgendaIntentConstitutionBucketV1.REPAIR_CANDIDATE

        IntentTypeV1.CHOOSE_RETAKE,
        IntentTypeV1.CHOOSE_KEEP_SCAN,
        IntentTypeV1.REQUEST_SUMMARY_DASHBOARD,
        IntentTypeV1.REQUEST_FOCUS_ON_ISSUE_TYPE,
        IntentTypeV1.REQUEST_FOCUS_ON_AREA,
        IntentTypeV1.REQUEST_FAST_MODE,
        IntentTypeV1.REQUEST_TEACH_MODE,
        IntentTypeV1.REQUEST_MODE_CHANGE,
        IntentTypeV1.REQUEST_SOLVE_STAGE,
        IntentTypeV1.REQUEST_REVALIDATE,
        IntentTypeV1.REQUEST_FOCUS_CHANGE,
        IntentTypeV1.REQUEST_HINT_LEVEL,
        IntentTypeV1.REQUEST_ONLY_VALIDATE,
        IntentTypeV1.REQUEST_ONLY_SOLVE,
        IntentTypeV1.PAUSE_ASSISTANT,
        IntentTypeV1.RESUME_ASSISTANT,
        IntentTypeV1.SET_LANGUAGE,
        IntentTypeV1.SET_ONE_QUESTION_MAX,
        IntentTypeV1.SET_EVIDENCE_VERBOSITY,
        IntentTypeV1.SET_NOTATION_STYLE,
        IntentTypeV1.SET_DIGIT_STYLE,
        IntentTypeV1.REQUEST_GO_TO_PREVIOUS_STAGE,
        IntentTypeV1.REQUEST_GO_TO_PREVIOUS_STEP,
        IntentTypeV1.REQUEST_STEP_BACK,
        IntentTypeV1.SOLVE_SKIP_N_STEPS,
        IntentTypeV1.WALKBACK_REQUEST ->
            AgendaIntentConstitutionBucketV1.USER_ROUTE_JUMP

        IntentTypeV1.CAPABILITY_CHECK,
        IntentTypeV1.META_APP_QUESTION,
        IntentTypeV1.FREE_TALK,
        IntentTypeV1.SOLVE_PAUSE,
        IntentTypeV1.REQUEST_CURRENT_STAGE_ELABORATION,
        IntentTypeV1.REQUEST_CURRENT_TECHNIQUE_EXPLANATION,
        IntentTypeV1.REQUEST_CURRENT_STAGE_COLLAPSE,
        IntentTypeV1.REQUEST_CURRENT_STAGE_EXAMPLE,
        IntentTypeV1.REQUEST_CURRENT_STAGE_REPEAT,
        IntentTypeV1.REQUEST_CURRENT_STAGE_REPHRASE,

        IntentTypeV1.ASK_WHY_NOT_DIGIT_IN_CELL,
        IntentTypeV1.ASK_WHY_DIGIT_IN_CELL,
        IntentTypeV1.ASK_WHY_THIS_CELL_IS_TARGET_FOR_DIGIT,
        IntentTypeV1.ASK_WHY_THIS_CELL_NOT_OTHER_CELL,
        IntentTypeV1.ASK_WHY_THIS_TECHNIQUE_APPLIES_HERE,
        IntentTypeV1.ASK_WHY_CURRENT_MOVE_BEFORE_OTHER_MOVE,
        IntentTypeV1.ASK_WHAT_BLOCKS_DIGIT_IN_HOUSE,
        IntentTypeV1.ASK_ONLY_PLACE_FOR_DIGIT_IN_HOUSE,
        IntentTypeV1.ASK_DIGIT_LOCATIONS_IN_HOUSE_EXACT,
        IntentTypeV1.ASK_NEXT_MOVE_EXACT,
        IntentTypeV1.CHECK_PROPOSED_DIGIT_IN_CELL,
        IntentTypeV1.CHECK_PROPOSED_CANDIDATE_SET_IN_CELL,
        IntentTypeV1.CHECK_PROPOSED_ELIMINATION_IN_SCOPE,
        IntentTypeV1.CHECK_PROPOSED_TECHNIQUE_APPLIES_HERE,
        IntentTypeV1.CHECK_PROPOSED_ROUTE_EQUIVALENCE,
        IntentTypeV1.ASK_ALTERNATIVE_TECHNIQUE_FOR_CURRENT_SPOT,
        IntentTypeV1.ASK_OTHER_LOCAL_MOVE_EXISTS,
        IntentTypeV1.ASK_COMPARE_CURRENT_ROUTE_WITH_ALTERNATIVE_ROUTE,

        IntentTypeV1.REQUEST_EXPLANATION,
        IntentTypeV1.SOLVE_STEP_WHY_NOT_DIGIT,
        IntentTypeV1.ASK_WHY_THIS_CELL,
        IntentTypeV1.REQUEST_REASONING_CHECK,
        IntentTypeV1.REPORT_BUG_OR_WRONG_ASSERTION,
        IntentTypeV1.ASK_SOLVABILITY_STATUS,
        IntentTypeV1.ASK_PROGRESS_METRICS,
        IntentTypeV1.ASK_IF_STUCK,
        IntentTypeV1.ASK_IF_GUESS_REQUIRED,
        IntentTypeV1.ASK_DIFFICULTY_ESTIMATE,
        IntentTypeV1.ASK_TECHNIQUES_NEEDED,
        IntentTypeV1.ASK_SOLVING_OVERVIEW,
        IntentTypeV1.ASK_TECHNIQUE_OVERVIEW,
        IntentTypeV1.ASK_STUCK_HELP,
        IntentTypeV1.ASK_ADVANCED_PATTERN_HELP,
        IntentTypeV1.ASK_HIDDEN_SINGLE_LOCATIONS,
        IntentTypeV1.ASK_NAKED_SINGLE_LOCATIONS,
        IntentTypeV1.ASK_NAKED_PAIR_LOCATIONS,
        IntentTypeV1.ASK_HIDDEN_PAIR_LOCATIONS,
        IntentTypeV1.ASK_POINTING_PAIR_TRIPLE,
        IntentTypeV1.ASK_BOX_LINE_REDUCTION,
        IntentTypeV1.ASK_XWING_CANDIDATE,

        IntentTypeV1.ASK_CELL_VALUE,
        IntentTypeV1.ASK_CELL_STATUS,
        IntentTypeV1.ASK_CELL_VALUE_EXACT,
        IntentTypeV1.ASK_CELL_CANDIDATES_EXACT,
        IntentTypeV1.ASK_CELL_CANDIDATE_COUNT_EXACT,
        IntentTypeV1.ASK_COMPARE_CANDIDATES_BETWEEN_CELLS,
        IntentTypeV1.ASK_ROW_CONTENTS,
        IntentTypeV1.ASK_COL_CONTENTS,
        IntentTypeV1.ASK_BOX_CONTENTS,
        IntentTypeV1.ASK_HOUSE_COMPLETION,
        IntentTypeV1.ASK_HOUSE_MISSING_DIGITS,
        IntentTypeV1.ASK_DIGIT_LOCATIONS,
        IntentTypeV1.ASK_DIGIT_LOCATIONS_IN_HOUSE_EXACT,
        IntentTypeV1.ASK_DIGIT_COUNT_GLOBAL,
        IntentTypeV1.ASK_DIGIT_COUNT_IN_HOUSE,
        IntentTypeV1.ASK_EMPTY_HOUSES,
        IntentTypeV1.ASK_NEARLY_COMPLETE_HOUSES,
        IntentTypeV1.ASK_HOUSE_OVERVIEW,
        IntentTypeV1.ASK_HOUSE_RANKING,
        IntentTypeV1.ASK_DIGIT_OVERVIEW,
        IntentTypeV1.ASK_DIGIT_IN_HOUSE_OVERVIEW,
        IntentTypeV1.ASK_GRID_CONTENTS_OVERVIEW,
        IntentTypeV1.ASK_COMPLETE_HOUSES_COUNT,
        IntentTypeV1.ASK_COMPLETE_HOUSES_LIST,
        IntentTypeV1.ASK_HOUSES_COMPLETION_RANKING,
        IntentTypeV1.ASK_CELL_CANDIDATES,
        IntentTypeV1.ASK_CANDIDATE_COUNT_CELL,
        IntentTypeV1.ASK_CELLS_WITH_N_CANDIDATES,
        IntentTypeV1.ASK_HOUSE_CANDIDATE_MAP,
        IntentTypeV1.ASK_HOUSE_CANDIDATE_MAP_EXACT,
        IntentTypeV1.ASK_CANDIDATE_FREQUENCY,
        IntentTypeV1.ASK_CANDIDATES_OVERVIEW,
        IntentTypeV1.ASK_CANDIDATES_CELL_OVERVIEW,
        IntentTypeV1.ASK_CANDIDATES_DISTRIBUTION,

        IntentTypeV1.ASK_STRUCTURAL_VALIDITY,
        IntentTypeV1.ASK_CONFLICTS_GLOBAL,
        IntentTypeV1.ASK_CONFLICTS_IN_HOUSE,
        IntentTypeV1.ASK_CONFLICT_DETAILS_CELL,
        IntentTypeV1.ASK_DUPLICATES_IN_HOUSE,
        IntentTypeV1.ASK_UNRESOLVED_CELLS,
        IntentTypeV1.ASK_INVALID_REASON_EXPLAIN,
        IntentTypeV1.ASK_IF_RETAKE_NEEDED,
        IntentTypeV1.ASK_SEAL_STATUS,
        IntentTypeV1.ASK_CONFLICTS_OVERVIEW,
        IntentTypeV1.ASK_VALIDATION_OVERVIEW,
        IntentTypeV1.ASK_PROBLEM_CELLS_OVERVIEW,
        IntentTypeV1.ASK_MISMATCH_OVERVIEW,
        IntentTypeV1.ASK_RETAKE_GUIDANCE,
        IntentTypeV1.ASK_WHAT_CHANGED_RECENTLY,
        IntentTypeV1.ASK_WHAT_CHANGED_IN_SCOPE_RECENTLY,
        IntentTypeV1.ASK_SOURCE_OF_DIGIT,
        IntentTypeV1.ASK_SOURCE_OF_DIGIT_IN_CELL_EXACT,
        IntentTypeV1.ASK_OCR_CONFIDENCE_CELL,
        IntentTypeV1.ASK_OCR_CONFIDENCE_SUMMARY,
        IntentTypeV1.ASK_TRUST_OVERVIEW,
        IntentTypeV1.ASK_CELL_TRUST_DETAILS,
        IntentTypeV1.ASK_PROVENANCE_OVERVIEW,
        IntentTypeV1.SWITCH_FOCUS_CELL,
        IntentTypeV1.SWITCH_FOCUS_REGION,
        IntentTypeV1.ASK_UI_LEGEND,
        IntentTypeV1.ASK_HOW_TO_LOCATE_CELL,
        IntentTypeV1.ASK_BOX_INDEX_MAPPING,
        IntentTypeV1.ASK_COORDINATE_TRANSLATION,
        IntentTypeV1.ASK_WHAT_DO_YOU_MEAN_BY_HOUSE,
        IntentTypeV1.ASK_UI_HELP,
        IntentTypeV1.ASK_COORDINATES_HELP,
        IntentTypeV1.ASK_WHAT_YOU_KNOW_NOW,
        IntentTypeV1.ASK_WHAT_YOU_DONT_KNOW,
        IntentTypeV1.REQUEST_EXPORT_STATE ->
            AgendaIntentConstitutionBucketV1.USER_DETOUR

        else ->
            AgendaIntentConstitutionBucketV1.NONE
    }

fun isDirectAppOwnedIntentForAgendaV1(t: IntentTypeV1): Boolean =
    agendaIntentConstitutionBucketV1(t) == AgendaIntentConstitutionBucketV1.DIRECT_APP_OWNED

fun isUserRouteJumpIntentForAgendaV1(t: IntentTypeV1): Boolean =
    agendaIntentConstitutionBucketV1(t) == AgendaIntentConstitutionBucketV1.USER_ROUTE_JUMP

fun isUserDetourIntentForAgendaV1(t: IntentTypeV1): Boolean =
    agendaIntentConstitutionBucketV1(t) == AgendaIntentConstitutionBucketV1.USER_DETOUR

fun isRepairCandidateIntentForAgendaV1(t: IntentTypeV1): Boolean =
    agendaIntentConstitutionBucketV1(t) == AgendaIntentConstitutionBucketV1.REPAIR_CANDIDATE

// -----------------------------------------------------------------------------
// Phase 2 — Decision memory latches (route commits)
// -----------------------------------------------------------------------------

/**
 * Retake decision fork for the current captured grid:
 * - UNDECIDED: we haven't committed the fork yet
 * - RETAKE: user chose to retake (typically followed by WAIT_FOR_TAP)
 * - KEEP_SCAN: user chose to keep current scan and verify mismatches
 */
enum class RetakeDecisionV1 { UNDECIDED, RETAKE, KEEP_SCAN }

data class IntentTargetV1(
    val cell: String? = null,                 // "r4c2"
    val cell2: String? = null,                // "r4c3" (for swaps, comparisons)
    val region: RegionRefV1? = null           // row/col/box
) {
    fun validate(errors: MutableList<String>) {
        if (cell != null && !cell.isValidCell()) errors.add("target.cell invalid: '$cell'")
        if (cell2 != null && !cell2.isValidCell()) errors.add("target.cell2 invalid: '$cell2'")
        region?.validate(errors)
    }

    fun toJson(): JSONObject = JSONObject().apply {
        if (!cell.isNullOrBlank()) put("cell", cell)
        if (!cell2.isNullOrBlank()) put("cell2", cell2)
        if (region != null) put("region", region.toJson())
    }

    companion object {
        fun parse(o: JSONObject, errors: MutableList<String>): IntentTargetV1 {
            val cell = o.optStringOrNull("cell")?.takeIf { it.isNotBlank() }
            val cell2 = o.optStringOrNull("cell2")?.takeIf { it.isNotBlank() }
            val region = o.optJSONObject("region")?.let { RegionRefV1.parse(it, errors) }
            return IntentTargetV1(cell = cell, cell2 = cell2, region = region)
        }
    }
}

data class IntentPayloadV1(
    // existing
    val digit: Int? = null,                    // 1..9
    val rawText: String? = null,
    val digits: List<Int>? = null,
    val regionDigits: String? = null,

    // NEW: generic query controls
    val queryKind: String? = null,             // "value"|"count"|"locations"|"missing_digits"|"candidates"|...
    val candidateCount: Int? = null,           // e.g., 2 for bivalue
    val technique: String? = null,             // "xwing"|"naked_pair"|...
    val scope: String? = null,                 // "GLOBAL"|"ROW"|"COL"|"BOX"|"REGION"|"NON_GIVENS"|...
    val detourSemanticFamily: DetourSemanticFamilyV1? = null,

    // NEW: workflow prefs
    val detailLevel: String? = null,           // "brief"|"normal"|"deep"
    val notation: String? = null,              // "rXcY"|"A1"
    val language: String? = null,              // "en"|"fr"
    val evidenceVerbosity: String? = null,     // "light"|"normal"|"deep"
    val hintLevel: String? = null,             // "minimal"|"gentle"|"explicit"
    val fastMode: Boolean? = null,
    val teachMode: Boolean? = null,
    val oneQuestionMax: Boolean? = null,

// -------------------------
// Phase 4: solving/workflow controls
// -------------------------
    val skipSteps: Int? = null,                 // e.g. 3 for “skip next 3 steps”
    val whyNotDigit: Int? = null,               // e.g. 2 for “why not 2?”
    val edits: JSONArray? = null,               // optional batch edits [{cell:"r2c5",digit:6},...]

    // Walkback controls
    val walkbackSteps: Int? = null,             // e.g. 2 => undo 2 solve moves
    val walkbackToJourneyStep: Int? = null      // e.g. 48 => go back to "cell 48" (journey step number)
) {
    fun validate(errors: MutableList<String>) {
        if (digit != null && digit !in 1..9) errors.add("payload.digit out of range: $digit")
        digits?.forEach { if (it !in 1..9) errors.add("payload.digits contains out of range: $it") }
        if (digits != null && digits.size > 9) errors.add("payload.digits too long: ${digits.size}")
        if (regionDigits != null && regionDigits.length > 20) errors.add("payload.region_digits too long")
        if (candidateCount != null && candidateCount !in 1..9) errors.add("payload.candidate_count out of range: $candidateCount")

        if (skipSteps != null && skipSteps !in 1..60) errors.add("payload.skip_steps out of range: $skipSteps")
        if (walkbackSteps != null && walkbackSteps !in 1..60) errors.add("payload.walkback_steps out of range: $walkbackSteps")
        if (walkbackToJourneyStep != null && walkbackToJourneyStep !in 1..81) errors.add("payload.walkback_to_journey_step out of range: $walkbackToJourneyStep")

        if (whyNotDigit != null && whyNotDigit !in 1..9) errors.add("payload.why_not_digit out of range: $whyNotDigit")
    }

    fun toJson(): JSONObject = JSONObject().apply {
        if (digit != null) put("digit", digit)
        if (!rawText.isNullOrBlank()) put("raw_text", rawText)
        if (digits != null) put("digits", JSONArray().apply { digits.forEach { put(it) } })
        if (!regionDigits.isNullOrBlank()) put("region_digits", regionDigits)

        if (!queryKind.isNullOrBlank()) put("query_kind", queryKind)
        if (candidateCount != null) put("candidate_count", candidateCount)
        if (!technique.isNullOrBlank()) put("technique", technique)
        if (!scope.isNullOrBlank()) put("scope", scope)
        if (detourSemanticFamily != null) put("detour_semantic_family", detourSemanticFamily.name)

        if (!detailLevel.isNullOrBlank()) put("detail_level", detailLevel)
        if (!notation.isNullOrBlank()) put("notation", notation)
        if (!language.isNullOrBlank()) put("language", language)
        if (!evidenceVerbosity.isNullOrBlank()) put("evidence_verbosity", evidenceVerbosity)
        if (!hintLevel.isNullOrBlank()) put("hint_level", hintLevel)

        if (fastMode != null) put("fast_mode", fastMode)
        if (teachMode != null) put("teach_mode", teachMode)
        if (oneQuestionMax != null) put("one_question_max", oneQuestionMax)
        if (skipSteps != null) put("skip_steps", skipSteps)
        if (whyNotDigit != null) put("why_not_digit", whyNotDigit)
        if (edits != null) put("edits", edits)


        if (walkbackSteps != null) put("walkback_steps", walkbackSteps)
        if (walkbackToJourneyStep != null) put("walkback_to_journey_step", walkbackToJourneyStep)

    }

    companion object {
        /**
         * ✅ PATCH 1.4: parse support for new payload fields.
         * This is required because IntentV1.parse() calls IntentPayloadV1.parse(...).
         */
        fun parse(o: JSONObject, errors: MutableList<String>): IntentPayloadV1 {
            val digit = o.optIntOrNull("digit")
            val rawText = o.optStringOrNull("raw_text")?.takeIf { it.isNotBlank() }

            val digitsArr = o.optJSONArrayOrNull("digits")
            val digits = if (digitsArr != null) {
                val out = mutableListOf<Int>()
                for (i in 0 until digitsArr.length()) {
                    val v = digitsArr.optInt(i, -1)
                    if (v in 1..9) out.add(v) else if (v != -1) errors.add("payload.digits invalid: $v")
                }
                out.takeIf { it.isNotEmpty() }
            } else null

            val regionDigits = o.optStringOrNull("region_digits")?.takeIf { it.isNotBlank() }

            // ---- NEW: query controls ----
            val queryKind = o.optStringOrNull("query_kind")?.takeIf { it.isNotBlank() }
            val candidateCount = o.optIntOrNull("candidate_count")
            val technique = o.optStringOrNull("technique")?.takeIf { it.isNotBlank() }
            val scope = o.optStringOrNull("scope")?.takeIf { it.isNotBlank() }
            val detourSemanticFamily =
                enumValueOrNull<DetourSemanticFamilyV1>(
                    o.optStringOrNull("detour_semantic_family")
                        ?: o.optStringOrNull("detourSemanticFamily")
                )

            // ---- NEW: prefs ----
            val detailLevel = o.optStringOrNull("detail_level")?.takeIf { it.isNotBlank() }
            val notation = o.optStringOrNull("notation")?.takeIf { it.isNotBlank() }
            val language = o.optStringOrNull("language")?.takeIf { it.isNotBlank() }
            val evidenceVerbosity = o.optStringOrNull("evidence_verbosity")?.takeIf { it.isNotBlank() }
            val hintLevel = o.optStringOrNull("hint_level")?.takeIf { it.isNotBlank() }

            val fastMode = if (o.has("fast_mode") && !o.isNull("fast_mode")) o.optBoolean("fast_mode") else null
            val teachMode = if (o.has("teach_mode") && !o.isNull("teach_mode")) o.optBoolean("teach_mode") else null
            val oneQuestionMax = if (o.has("one_question_max") && !o.isNull("one_question_max")) o.optBoolean("one_question_max") else null
            val skipSteps = o.optIntOrNull("skip_steps")
            val whyNotDigit = o.optIntOrNull("why_not_digit")
            val edits = o.optJSONArrayOrNull("edits")

            val walkbackSteps = o.optIntOrNull("walkback_steps")
            val walkbackToJourneyStep = o.optIntOrNull("walkback_to_journey_step")

            val payload = IntentPayloadV1(
                digit = digit,
                rawText = rawText,
                digits = digits,
                regionDigits = regionDigits,

                queryKind = queryKind,
                candidateCount = candidateCount,
                technique = technique,
                scope = scope,
                detourSemanticFamily = detourSemanticFamily,

                detailLevel = detailLevel,
                notation = notation,
                language = language,
                evidenceVerbosity = evidenceVerbosity,
                hintLevel = hintLevel,
                fastMode = fastMode,
                teachMode = teachMode,
                oneQuestionMax = oneQuestionMax,
                skipSteps = skipSteps,
                whyNotDigit = whyNotDigit,
                edits = edits,

                walkbackSteps = walkbackSteps,
                walkbackToJourneyStep = walkbackToJourneyStep
            )

            payload.validate(errors)
            return payload
        }
    }
}

enum class ReferenceResolutionModeV1 {
    NONE,
    FOCUS_CELL,
    PENDING_TARGET,
    CURRENT_STEP_TARGET,
    PREVIOUS_ASSISTANT_REFERENT
}

enum class RepairSignalV1 {
    NONE,
    MISHEARD,
    MISUNDERSTOOD,
    CONTRADICTION,
    LOOP_COMPLAINT,
    REPEATED_QUESTION
}

enum class ContextSpanHintV1 {
    LOCAL,
    MEDIUM,
    LONG
}

enum class DetourSemanticFamilyV1 {
    PROOF_BLOCKER_AT_CELL,
    PROOF_DIGIT_IN_HOUSE,
    PROPOSAL_VERDICT,
    TARGETED_EXPLANATION,
    LOCAL_READOUT,
    CANDIDATE_STATE,
    GENERAL_EXPLANATION
}

data class IntentV1(
    val id: String,
    val type: IntentTypeV1,
    val confidence: Double,
    val targets: List<IntentTargetV1> = emptyList(),
    val payload: IntentPayloadV1 = IntentPayloadV1(),
    val missing: List<String> = emptyList(),
    val evidenceText: String? = null,

    // V1 spec: lets the user explicitly address a prior clarification request.
    val addressesUserAgendaId: String? = null,

    // Series 1 — structured Tick-1 follow-up/reference hints.
    // Downstream consumers may use this instead of re-reading raw user text.
    val referenceResolutionMode: ReferenceResolutionModeV1? = null
) {
    fun validate(errors: MutableList<String>) {
        if (id.isBlank()) errors.add("intent.id is blank")
        if (confidence < 0.0 || confidence > 1.0) errors.add("intent.confidence out of range: $confidence")

        targets.forEach { it.validate(errors) }
        payload.validate(errors)

        if (missing.isNotEmpty() && confidence > 0.85) {
            errors.add("intent.missing is non-empty but confidence is high ($confidence)")
        }
    }

    companion object {
        fun parse(o: JSONObject, errors: MutableList<String>): IntentV1 {
            val id = o.optString("id", "").takeIf { it.isNotBlank() } ?: run {
                errors.add("intent.id missing/blank")
                "t1:missing_id"
            }

            val typeStr = o.optString("type", "").takeIf { it.isNotBlank() }
            val type = runCatching { IntentTypeV1.valueOf(typeStr ?: "UNKNOWN") }
                .getOrElse {
                    if (!typeStr.isNullOrBlank()) errors.add("intent.type invalid: '$typeStr'")
                    IntentTypeV1.UNKNOWN
                }

            val confidence = clamp01(o.optDouble("confidence", 0.0))

            val targetsArr = o.optJSONArray("targets")
            val targets = mutableListOf<IntentTargetV1>()
            if (targetsArr != null) {
                for (i in 0 until targetsArr.length()) {
                    val to = targetsArr.optJSONObject(i) ?: continue
                    targets.add(IntentTargetV1.parse(to, errors))
                }
            }

            val payloadObj = o.optJSONObject("payload")
            val payload = if (payloadObj != null) IntentPayloadV1.parse(payloadObj, errors) else IntentPayloadV1()

            val missingArr = o.optJSONArray("missing")
            val missing = if (missingArr != null) missingArr.toStringList(maxItems = 12) else emptyList()

            val evidenceText = o.optStringOrNull("evidence_text")?.takeIf { it.isNotBlank() }

            val addresses =
                o.optStringOrNull("addresses_user_agenda_id")?.takeIf { it.isNotBlank() }
                    ?: o.optStringOrNull("addressesUserAgendaId")?.takeIf { it.isNotBlank() }

            val referenceResolutionMode =
                enumValueOrNull<ReferenceResolutionModeV1>(
                    o.optStringOrNull("reference_resolution_mode")
                        ?: o.optStringOrNull("referenceResolutionMode")
                )?.takeIf { it != ReferenceResolutionModeV1.NONE }

            val intent = IntentV1(
                id = id,
                type = type,
                confidence = confidence,
                targets = targets,
                payload = payload,
                missing = missing,
                evidenceText = evidenceText,
                addressesUserAgendaId = addresses,
                referenceResolutionMode = referenceResolutionMode
            )

            intent.validate(errors)
            return intent
        }
    }
}

data class IntentEnvelopeV1(

    val version: String = "intent_envelope_v1",
    val intents: List<IntentV1> = emptyList(),
    val freeTalkTopic: String? = null,
    val freeTalkConfidence: Double = 0.0,

    // Series 1 — structured Tick-1 discourse/control hints.
    // These allow downstream layers to consume explicit semantic signals
    // instead of inferring meaning again from raw user text.
    val repairSignal: RepairSignalV1? = null,
    val contextSpanHint: ContextSpanHintV1? = null,
    val referencesPriorTurns: Boolean? = null,

    // ✅ NEW (Tick-1 memory updates)
    // These are deltas inferred from the latest user utterance, relative to TurnContextV1.tally.
    // They may be null/empty when no updates were inferred.
    val userTallyDelta: UserTallyV1? = null,
    val assistantTallyDelta: AssistantTallyV1? = null,
    val relationshipDelta: RelationshipDeltaV1? = null,

    val rawUserText: String? = null,
    val language: String? = null,
    val asrQuality: String? = null
) {
    data class ParseResult(
        val value: IntentEnvelopeV1,
        val errors: List<String>
    )

    fun validate(errors: MutableList<String>) {
        if (version != "intent_envelope_v1") errors.add("version must be 'intent_envelope_v1' (got '$version')")
        intents.forEach { it.validate(errors) }
    }

    companion object {
        fun parseJson(json: String): ParseResult {
            val errors = mutableListOf<String>()
            val o = runCatching { JSONObject(json) }.getOrElse {
                return ParseResult(
                    value = IntentEnvelopeV1(intents = emptyList(), rawUserText = null),
                    errors = listOf("json_parse_failed:${it.javaClass.simpleName}")
                )
            }

            val version = o.optString("version", "")
            val intentsArr = o.optJSONArray("intents")
            val intents = mutableListOf<IntentV1>()
            if (intentsArr != null) {
                for (i in 0 until intentsArr.length()) {
                    val io = intentsArr.optJSONObject(i) ?: continue
                    intents.add(IntentV1.parse(io, errors))
                }
            }

            val freeTalk = o.optJSONObject("free_talk")
            val ftTopic = freeTalk?.optStringOrNull("topic")?.takeIf { it.isNotBlank() }
            val ftConf = clamp01(freeTalk?.optDouble("confidence", 0.0) ?: 0.0)

            val repairSignal =
                enumValueOrNull<RepairSignalV1>(
                    o.optStringOrNull("repair_signal")
                        ?: o.optStringOrNull("repairSignal")
                )?.takeIf { it != RepairSignalV1.NONE }

            val contextSpanHint =
                enumValueOrNull<ContextSpanHintV1>(
                    o.optStringOrNull("context_span_hint")
                        ?: o.optStringOrNull("contextSpanHint")
                )

            val referencesPriorTurns =
                when {
                    o.has("references_prior_turns") && !o.isNull("references_prior_turns") ->
                        o.optBoolean("references_prior_turns")
                    o.has("referencesPriorTurns") && !o.isNull("referencesPriorTurns") ->
                        o.optBoolean("referencesPriorTurns")
                    else -> null
                }

            val notes = o.optJSONObject("notes")
            val rawUserText = notes?.optStringOrNull("raw_user_text")?.takeIf { it.isNotBlank() }
            val language = notes?.optStringOrNull("language")?.takeIf { it.isNotBlank() }
            val asrQuality = notes?.optStringOrNull("asr_quality")?.takeIf { it.isNotBlank() }

            // ✅ NEW: tally deltas (optional)
            val userTallyDeltaObj = o.optJSONObjectOrNull("user_tally_delta")
            val assistantTallyDeltaObj = o.optJSONObjectOrNull("assistant_tally_delta")
            val relationshipDeltaObj = o.optJSONObjectOrNull("relationship_delta")

            val userTallyDelta = runCatching {
                userTallyDeltaObj?.let { UserTallyV1.parse(it) }
            }.getOrNull()?.takeIf { it != UserTallyV1() }

            val assistantTallyDelta = runCatching {
                assistantTallyDeltaObj?.let { AssistantTallyV1.parse(it) }
            }.getOrNull()?.takeIf { it != AssistantTallyV1() }

            val relationshipDelta = runCatching {
                relationshipDeltaObj?.let { RelationshipDeltaV1.parse(it) }
            }.getOrNull()?.takeIf {
                it.observations.isNotEmpty() || it.candidateUpdates.isNotEmpty()
            }

            val env = IntentEnvelopeV1(
                version = version.ifBlank { "intent_envelope_v1" },
                intents = intents,
                freeTalkTopic = ftTopic,
                freeTalkConfidence = ftConf,

                repairSignal = repairSignal,
                contextSpanHint = contextSpanHint,
                referencesPriorTurns = referencesPriorTurns,

                userTallyDelta = userTallyDelta,
                assistantTallyDelta = assistantTallyDelta,
                relationshipDelta = relationshipDelta,

                rawUserText = rawUserText,
                language = language,
                asrQuality = asrQuality
            )

            env.validate(errors)
            return ParseResult(value = env, errors = errors)
        }
    }
}

// -----------------------------------------------------------------------------
// Tick 2 — ReplyRequestV1
// -----------------------------------------------------------------------------

enum class DecisionKindV1 {
    ASKED_PENDING,
    REQUESTED_SEAL,
    RECOMMENDED_RETAKE,

    CLEARED_PENDING,
    APPLIED_MUTATION,
    ASKED_CLARIFICATION,
    ANSWERED_GRID_QUESTION,
    SWITCHED_MODE,

    NO_OP
}

enum class ReplyVoiceV1 { friendly, coach, calm, story_coach }
enum class ReplyToneV1 { natural, concise, encouraging, vivid, warm }

/**
 * Phase 1 — first-class reply demand routing key.
 *
 * This does NOT yet change what Tick2 receives.
 * It only gives the architecture an explicit answer to:
 * "What job is this reply actually doing?"
 */
enum class ReplyDemandCategoryV1 {
    ONBOARDING_OPENING,

    // Wave-1 confirming split
    CONFIRM_STATUS_SUMMARY,
    CONFIRM_EXACT_MATCH_GATE,
    CONFIRM_FINALIZE_GATE,

    // Wave-2 confirming expansion
    CONFIRM_RETAKE_GATE,
    CONFIRM_MISMATCH_GATE,
    CONFIRM_CONFLICT_GATE,
    CONFIRM_NOT_UNIQUE_GATE,

    // Transitional legacy confirming bucket.
    // Keep temporarily for staged migration, but new routing should prefer
    // the narrower confirming families above.
    CONFIRMING_VALIDATION_SUMMARY,

    // Wave-1 transactional / inspection families
    PENDING_CLARIFICATION,
    GRID_VALIDATION_ANSWER,
    GRID_CANDIDATE_ANSWER,

    // Wave-2 bounded pending transactional families
    PENDING_CELL_CONFIRM_AS_IS,
    PENDING_CELL_CONFIRM_TO_DIGIT,
    PENDING_REGION_CONFIRM_AS_IS,
    PENDING_REGION_CONFIRM_TO_DIGITS,
    PENDING_DIGIT_PROVIDE,
    PENDING_INTERPRETATION_CONFIRM,

    // Wave-3 grid inspection expansion
    GRID_OCR_TRUST_ANSWER,
    GRID_CONTENTS_ANSWER,
    GRID_CHANGELOG_ANSWER,

    // Wave-3 grid mutation / execution families
    GRID_EDIT_EXECUTION,
    GRID_CLEAR_EXECUTION,
    GRID_SWAP_EXECUTION,
    GRID_BATCH_EDIT_EXECUTION,
    GRID_UNDO_REDO_EXECUTION,
    GRID_LOCK_GIVENS_EXECUTION,

    // Wave-4 solving support families
    SOLVING_STAGE_ELABORATION,
    SOLVING_STAGE_REPEAT,
    SOLVING_STAGE_REPHRASE,
    SOLVING_GO_BACKWARD,
    SOLVING_STEP_REVEAL,
    SOLVING_ROUTE_CONTROL,

    // Wave-4 solving detour families
    DETOUR_PROOF_CHALLENGE,
    DETOUR_TARGET_CELL_QUERY,
    DETOUR_NEIGHBOR_CELL_QUERY,
    DETOUR_REASONING_CHECK,
    DETOUR_ALTERNATIVE_TECHNIQUE,
    DETOUR_LOCAL_MOVE_SEARCH,
    DETOUR_ROUTE_COMPARISON,

    // Wave-5 preferences / control families
    PREFERENCE_CHANGE,
    MODE_CHANGE,
    ASSISTANT_PAUSE_RESUME,
    VALIDATE_ONLY_OR_SOLVE_ONLY,
    FOCUS_REDIRECT,
    HINT_POLICY_CHANGE,

    // Wave-5 meta / capability / glossary / help families
    META_STATE_ANSWER,
    CAPABILITY_ANSWER,
    GLOSSARY_ANSWER,
    UI_HELP_ANSWER,
    COORDINATE_HELP_ANSWER,

    // Wave-5 narrowed free-talk families
    FREE_TALK_NON_GRID,
    SMALL_TALK_BRIDGE,

    SOLVING_SETUP,
    SOLVING_CONFRONTATION,
    SOLVING_RESOLUTION,
    REPAIR_CONTRADICTION,

    // Transitional legacy grid-chat bucket.
    // Keep temporarily for staged migration, but new routing should avoid
    // using this for clarification / validation / candidate work.
    FREE_TALK_IN_GRID_SESSION,

    RECOVERY_REPLY
}

/**
 * Phase 1 — internal setup-demand refinement.
 *
 * Public reply-demand routing stays simple (`SOLVING_SETUP`), but internally we
 * refine SETUP into the actual demand shape the turn needs.
 *
 * Notes:
 * - BASE_SINGLES_SETUP covers hidden/naked singles style setup
 * - FULL_HOUSE_SETUP covers Full House setup explicitly
 * - SUBSETS_SETUP covers subset-family setup generically (pair/triple/quad,
 *   naked/hidden, and any future subset technique already mapped to SUBSETS)
 * - INTERSECTIONS_SETUP covers the intersection family generically:
 *   claiming pair/triple + pointing pair/triple
 * - ADVANCED_PATTERN_SETUP covers fish / wings / chains
 * - FALLBACK_SETUP is the safe default when setup is known but archetype is not
 *
 * Wave-0 constitution for INTERSECTIONS:
 * 1) Setup must prove overlap confinement before naming the pattern.
 * 2) Setup must always audit the source house outside the overlap explicitly,
 *    step by step.
 * 3) Setup must end on territorial control / permission change, not on the
 *    downstream target effect.
 */
enum class SetupDemandProfileV1 {
    BASE_SINGLES_SETUP,
    FULL_HOUSE_SETUP,
    SUBSETS_SETUP,
    INTERSECTIONS_SETUP,
    ADVANCED_PATTERN_SETUP,
    FALLBACK_SETUP
}

enum class SetupNarrationDoctrineV1 {
    LENS_FIRST,
    PATTERN_FIRST,
    NEUTRAL
}



/**
 * Phase 1 — internal confrontation-proof refinement.
 *
 * Public reply-demand routing stays simple (`SOLVING_CONFRONTATION`), but
 * internally we refine CONFRONTATION into the actual proof shape the turn needs.
 *
 * Notes:
 * - BASE_SINGLES_PROOF covers hidden/naked single style collapse proof
 * - FULL_HOUSE_PROOF covers Full House proof explicitly
 * - SUBSETS_PROOF covers subset-family proof generically
 * - INTERSECTIONS_PROOF covers the intersection family generically:
 *   claiming pair/triple + pointing pair/triple
 * - ADVANCED_PATTERN_PROOF covers fish / wings / chains style proof
 * - FALLBACK_PROOF is the safe default when confrontation is known but the
 *   proof archetype is still weak / unknown
 *
 * Wave-0 constitution for INTERSECTIONS:
 * 1) Confrontation must spotlight the downstream target frame.
 * 2) Ordinary witnesses act first.
 * 3) The intersection pattern then enters as a territorial-control hero.
 * 4) The decisive exclusion must be explained as a permission change caused by
 *    overlap confinement.
 * 5) Confrontation stops before commit.
 */
enum class ConfrontationProofProfileV1 {
    BASE_SINGLES_PROOF,
    FULL_HOUSE_PROOF,
    SUBSETS_PROOF,
    INTERSECTIONS_PROOF,
    ADVANCED_PATTERN_PROOF,
    FALLBACK_PROOF
}

enum class ConfrontationNarrationDoctrineV1 {
    LENS_FIRST,
    PATTERN_FIRST,
    NEUTRAL
}

/**
 * Phase 1 — internal resolution-stage refinement.
 *
 * Public reply-demand routing stays simple (`SOLVING_RESOLUTION`), but
 * internally we refine RESOLUTION into the actual compact commit/recap shape
 * the turn needs.
 *
 * Notes:
 * - BASE_SINGLES_RESOLUTION covers direct single-style commit turns
 * - FULL_HOUSE_RESOLUTION covers Full House resolution explicitly
 * - SUBSETS_RESOLUTION covers subset cleanup + final finish recap
 * - INTERSECTIONS_RESOLUTION covers intersection cleanup + final finish recap
 * - ADVANCED_PATTERN_RESOLUTION covers fish / wings / chains style recap
 * - FALLBACK_RESOLUTION is the safe default when resolution is known but the
 *   recap/commit archetype is still weak / unknown
 *
 * Wave-0 constitution for INTERSECTIONS:
 * 1) Resolution commits first.
 * 2) Resolution must distinguish the pattern's indirect territorial power from
 *    the final direct survivor.
 * 3) Resolution should land on structural takeaway + graceful exit.
 */
enum class ResolutionProfileV1 {
    BASE_SINGLES_RESOLUTION,
    FULL_HOUSE_RESOLUTION,
    SUBSETS_RESOLUTION,
    INTERSECTIONS_RESOLUTION,
    ADVANCED_PATTERN_RESOLUTION,
    FALLBACK_RESOLUTION
}


fun isConfirmingPrimaryDemandCategoryV1(
    category: ReplyDemandCategoryV1
): Boolean =
    when (category) {
        ReplyDemandCategoryV1.CONFIRMING_VALIDATION_SUMMARY,
        ReplyDemandCategoryV1.GRID_VALIDATION_ANSWER,
        ReplyDemandCategoryV1.GRID_CANDIDATE_ANSWER,
        ReplyDemandCategoryV1.GRID_OCR_TRUST_ANSWER,
        ReplyDemandCategoryV1.GRID_CONTENTS_ANSWER,
        ReplyDemandCategoryV1.GRID_CHANGELOG_ANSWER -> true

        else -> false
    }

fun isSolvingPrimaryDemandCategoryV1(
    category: ReplyDemandCategoryV1
): Boolean =
    when (category) {
        ReplyDemandCategoryV1.SOLVING_SETUP,
        ReplyDemandCategoryV1.SOLVING_CONFRONTATION,
        ReplyDemandCategoryV1.SOLVING_RESOLUTION,
        ReplyDemandCategoryV1.DETOUR_PROOF_CHALLENGE,
        ReplyDemandCategoryV1.DETOUR_TARGET_CELL_QUERY,
        ReplyDemandCategoryV1.DETOUR_NEIGHBOR_CELL_QUERY,
        ReplyDemandCategoryV1.DETOUR_REASONING_CHECK,
        ReplyDemandCategoryV1.DETOUR_ALTERNATIVE_TECHNIQUE,
        ReplyDemandCategoryV1.DETOUR_LOCAL_MOVE_SEARCH,
        ReplyDemandCategoryV1.DETOUR_ROUTE_COMPARISON,
        ReplyDemandCategoryV1.REPAIR_CONTRADICTION -> true

        else -> false
    }

fun isCrossPhaseDemandCategoryV1(
    category: ReplyDemandCategoryV1
): Boolean =
    when (category) {
        ReplyDemandCategoryV1.META_STATE_ANSWER,
        ReplyDemandCategoryV1.CAPABILITY_ANSWER,
        ReplyDemandCategoryV1.GLOSSARY_ANSWER,
        ReplyDemandCategoryV1.UI_HELP_ANSWER,
        ReplyDemandCategoryV1.COORDINATE_HELP_ANSWER,
        ReplyDemandCategoryV1.FREE_TALK_NON_GRID,
        ReplyDemandCategoryV1.SMALL_TALK_BRIDGE -> true

        else -> false
    }

fun isDemandCategoryLegalForPhaseV1(
    phase: GridPhase,
    category: ReplyDemandCategoryV1
): Boolean =
    when (phase) {
        GridPhase.CONFIRMING,
        GridPhase.SEALING ->
            isConfirmingPrimaryDemandCategoryV1(category) ||
                    isCrossPhaseDemandCategoryV1(category) ||
                    category == ReplyDemandCategoryV1.PENDING_CLARIFICATION ||
                    category == ReplyDemandCategoryV1.CONFIRM_FINALIZE_GATE ||
                    category == ReplyDemandCategoryV1.CONFIRM_RETAKE_GATE ||
                    category == ReplyDemandCategoryV1.CONFIRM_MISMATCH_GATE ||
                    category == ReplyDemandCategoryV1.CONFIRM_CONFLICT_GATE ||
                    category == ReplyDemandCategoryV1.CONFIRM_NOT_UNIQUE_GATE ||
                    category == ReplyDemandCategoryV1.CONFIRM_EXACT_MATCH_GATE ||
                    category == ReplyDemandCategoryV1.CONFIRM_STATUS_SUMMARY ||
                    category == ReplyDemandCategoryV1.ASSISTANT_PAUSE_RESUME ||
                    category == ReplyDemandCategoryV1.VALIDATE_ONLY_OR_SOLVE_ONLY ||
                    category == ReplyDemandCategoryV1.FOCUS_REDIRECT ||
                    category == ReplyDemandCategoryV1.HINT_POLICY_CHANGE ||
                    category == ReplyDemandCategoryV1.MODE_CHANGE ||
                    category == ReplyDemandCategoryV1.PREFERENCE_CHANGE ||
                    category == ReplyDemandCategoryV1.ONBOARDING_OPENING

        GridPhase.SOLVING ->
            isSolvingPrimaryDemandCategoryV1(category) ||
                    isCrossPhaseDemandCategoryV1(category) ||
                    category == ReplyDemandCategoryV1.PENDING_CLARIFICATION ||
                    category == ReplyDemandCategoryV1.SOLVING_ROUTE_CONTROL ||
                    category == ReplyDemandCategoryV1.ASSISTANT_PAUSE_RESUME ||
                    category == ReplyDemandCategoryV1.FOCUS_REDIRECT ||
                    category == ReplyDemandCategoryV1.HINT_POLICY_CHANGE ||
                    category == ReplyDemandCategoryV1.MODE_CHANGE ||
                    category == ReplyDemandCategoryV1.PREFERENCE_CHANGE ||
                    category == ReplyDemandCategoryV1.ONBOARDING_OPENING
    }

/**
 * Phase 2 — prompt modules.
 *
 * These are composable prompt fragments that will later replace the current
 * monolithic Tick2 prompt. Phase 2 only defines the vocabulary.
 */
enum class PromptModuleV1 {
    BASE_JSON_OUTPUT,
    BASE_PERSONA,
    BASE_PERSONA_SOLVING_MAIN_ROAD_MINI,
    BASE_PERSONA_SOLVING_DETOUR_MINI,
    NO_INVENTION_RULES,
    GRID_TRUTH_RULES,
    GRID_TRUTH_SOLVING_MAIN_ROAD_MINI,
    GRID_TRUTH_SOLVING_DETOUR_MINI,
    ONBOARDING_RULES,

    // Transitional legacy confirming rules.
    // Keep temporarily while confirming families are introduced.
    CONFIRMING_RULES,

    // Wave-1 specialized confirming / transactional / inspection rules
    CONFIRM_STATUS_RULES,
    CONFIRM_EXACT_MATCH_RULES,
    CONFIRM_FINALIZE_RULES,
    PENDING_GATE_RULES,
    CLARIFICATION_RULES,
    GRID_VALIDATION_ANSWER_RULES,
    GRID_CANDIDATE_ANSWER_RULES,

    // Wave-2 confirming expansion rules
    CONFIRM_RETAKE_RULES,
    CONFIRM_MISMATCH_RULES,
    CONFIRM_CONFLICT_RULES,
    CONFIRM_NOT_UNIQUE_RULES,

    // Wave-2 bounded pending transactional rules
    PENDING_CELL_CONFIRM_AS_IS_RULES,
    PENDING_CELL_CONFIRM_TO_DIGIT_RULES,
    PENDING_REGION_CONFIRM_AS_IS_RULES,
    PENDING_REGION_CONFIRM_TO_DIGITS_RULES,
    PENDING_DIGIT_PROVIDE_RULES,
    PENDING_INTERPRETATION_CONFIRM_RULES,

    // Wave-3 grid inspection expansion rules
    GRID_OCR_TRUST_ANSWER_RULES,
    GRID_CONTENTS_ANSWER_RULES,
    GRID_CHANGELOG_ANSWER_RULES,

    // Wave-3 grid mutation / execution rules
    GRID_EDIT_EXECUTION_RULES,
    GRID_CLEAR_EXECUTION_RULES,
    GRID_SWAP_EXECUTION_RULES,
    GRID_BATCH_EDIT_EXECUTION_RULES,
    GRID_UNDO_REDO_EXECUTION_RULES,
    GRID_LOCK_GIVENS_EXECUTION_RULES,

    // Wave-4 solving support rules
    SOLVING_STAGE_ELABORATION_RULES,
    SOLVING_STAGE_REPEAT_RULES,
    SOLVING_STAGE_REPHRASE_RULES,
    SOLVING_GO_BACKWARD_RULES,
    SOLVING_STEP_REVEAL_RULES,
    SOLVING_ROUTE_CONTROL_RULES,

    // Wave-4 solving detour rules
    DETOUR_PROOF_CHALLENGE_RULES,
    DETOUR_TARGET_CELL_QUERY_RULES,
    DETOUR_NEIGHBOR_CELL_QUERY_RULES,
    DETOUR_REASONING_CHECK_RULES,
    DETOUR_ALTERNATIVE_TECHNIQUE_RULES,
    DETOUR_LOCAL_MOVE_SEARCH_RULES,
    DETOUR_ROUTE_COMPARISON_RULES,

    // Wave 1 — typed detour packet rules (scaffolded in F3, enforced more
    // strongly in later phases)

    DETOUR_MOVE_PROOF_RULES,
    DETOUR_PROOF_MICRO_STAGE_RULES,
    DETOUR_PROOF_CLOSURE_CTA_RULES,
    DETOUR_PROOF_GEOMETRY_RULES,
    DETOUR_PROOF_CONTRADICTION_SPOTLIGHT_RULES,
    DETOUR_PROOF_LOCAL_PERMISSIBILITY_SCAN_RULES,
    DETOUR_PROOF_HOUSE_ALREADY_OCCUPIED_RULES,
    DETOUR_PROOF_FILLED_CELL_RULES,
    DETOUR_PROOF_SURVIVOR_LADDER_RULES,

    DETOUR_PROOF_CONTRAST_DUEL_RULES,
    DETOUR_PROOF_PATTERN_LEGITIMACY_RULES,
    DETOUR_PROOF_HONEST_INSUFFICIENCY_RULES,
    DETOUR_LOCAL_GRID_INSPECTION_RULES,
    DETOUR_USER_PROPOSAL_VERDICT_RULES,

    // Wave-5 preferences / control rules
    PREFERENCE_CHANGE_RULES,
    MODE_CHANGE_RULES,
    ASSISTANT_PAUSE_RESUME_RULES,
    VALIDATE_ONLY_OR_SOLVE_ONLY_RULES,
    FOCUS_REDIRECT_RULES,
    HINT_POLICY_CHANGE_RULES,

    // Wave-5 meta / capability / glossary / help rules
    META_STATE_ANSWER_RULES,
    CAPABILITY_ANSWER_RULES,
    GLOSSARY_ANSWER_RULES,
    UI_HELP_ANSWER_RULES,
    COORDINATE_HELP_ANSWER_RULES,

    // Wave-5 narrowed free-talk rules
    FREE_TALK_NON_GRID_RULES,
    SMALL_TALK_BRIDGE_RULES,

    SOLVING_SETUP_RULES,
    SETUP_LENS_FIRST_RULES,
    SETUP_PATTERN_FIRST_RULES,
    INTERSECTION_SETUP_RULES,          // Wave-0 placeholder: explicit crossroads / overlap-confinement setup law
    CONFRONTATION_LENS_FIRST_RULES,
    CONFRONTATION_PATTERN_FIRST_RULES,
    INTERSECTION_CONFRONTATION_RULES,  // Wave-0 placeholder: explicit territorial two-actor proof law
    SOLVING_CONFRONTATION_RULES,
    SOLVING_RESOLUTION_RULES,
    RESOLUTION_BASIC_RULES,
    RESOLUTION_ADVANCED_RULES,
    INTERSECTION_RESOLUTION_RULES,     // Wave-0 placeholder: explicit territorial-control payoff law
    REPAIR_RULES,
    FREE_TALK_GRID_RULES,
    CTA_ENDING_RULES,
    CTA_ENDING_SOLVING_MAIN_ROAD_MINI,
    CTA_ENDING_SOLVING_DETOUR_MINI,
    COMMIT_TRUTH_RULES,
    NO_CONTRADICTION_RULES,

    // Wave-7 personalization discipline rules
    PERSONALIZATION_CORE_RULES,
    PERSONALIZATION_MAIN_ROAD_SOLVING_RULES,
    PERSONALIZATION_SOLVING_DETOUR_RULES,
    PERSONALIZATION_VALIDATION_RULES,
    PERSONALIZATION_SOCIAL_RULES,
    PERSONALIZATION_MINIMAL_RULES
}

/**
 * Phase 2 — supply channels.
 *
 * Important distinction:
 * - FactBundleV1.Type describes source bundle families already present today.
 * - ReplySupplyChannelV1 describes what Tick2 is allowed to consume semantically.
 *
 * Later phases will map channels to projected slices of canonical truth.
 */
enum class ReplySupplyChannelV1 {
    TURN_HEADER_MINI,
    STYLE_MINI,
    DECISION_SUMMARY_MINI,
    CONTINUITY_SHORT,
    PERSONALIZATION_MINI,
    CTA_CONTEXT,
    SOLVABILITY_CONTEXT,

    ONBOARDING_CONTEXT,
    CONFIRMING_CONTEXT,

    // Wave-1 / Wave-2 transactional / inspection channels
    PENDING_CONTEXT_CHANNEL,
    GRID_VALIDATION_CONTEXT,
    GRID_CANDIDATE_CONTEXT,

    // Wave-3 inspection / execution channels
    GRID_OCR_TRUST_CONTEXT,
    GRID_CONTENTS_CONTEXT,
    GRID_CHANGELOG_CONTEXT,
    GRID_MUTATION_CONTEXT,

    // Wave-4 solving support / detour channels
    SOLVING_SUPPORT_CONTEXT,
    DETOUR_CONTEXT,

    // Wave 1 / Track 1 — typed detour packet channels
    DETOUR_MOVE_PROOF_PACKET,
    DETOUR_LOCAL_GRID_INSPECTION_PACKET,
    DETOUR_USER_PROPOSAL_VERDICT_PACKET,
    DETOUR_ALTERNATIVE_TECHNIQUE_PACKET,
    DETOUR_LOCAL_MOVE_SEARCH_PACKET,
    DETOUR_ROUTE_COMPARISON_PACKET,
    DETOUR_NARRATIVE_CONTEXT,

    // Wave-5 preferences / meta / help / free-talk channels
    PREFERENCE_CONTEXT,
    META_CONTEXT,
    HELP_CONTEXT,
    FREE_TALK_CONTEXT,

    SETUP_REPLY_PACKET,
    SETUP_STORY_SLICE,
    SETUP_STEP_SLICE,

    CONFRONTATION_REPLY_PACKET,
    CONFRONTATION_STORY_SLICE,
    CONFRONTATION_STEP_SLICE,

    RESOLUTION_REPLY_PACKET,
    RESOLUTION_STORY_SLICE,
    RESOLUTION_STEP_SLICE,

    GLOSSARY_MINI,
    TECHNIQUE_CARD_MINI,
    HANDOVER_NOTE_MINI,
    OVERLAY_MINI,
    REPAIR_CONTEXT
}

/**
 * Phase 1 — lightweight routing result emitted into telemetry and later reused
 * by the assembly planner / contract registry phases.
 */
data class ReplyDemandResolutionV1(
    val category: ReplyDemandCategoryV1,
    val reason: String,
    val phase: String? = null,
    val pendingKind: String? = null,
    val storyStage: String? = null,
    val openingTurn: Boolean = false,
    val canonicalPositionKind: String? = null,
    val canonicalLegal: Boolean? = null,

    /**
     * Internal-only refinement for SOLVING_SETUP.
     * Null for all non-setup demand categories.
     */
    val setupProfile: SetupDemandProfileV1? = null,

    /**
     * Internal-only refinement for SOLVING_CONFRONTATION.
     * Null for all non-confrontation demand categories.
     */
    val confrontationProofProfile: ConfrontationProofProfileV1? = null,

    /**
     * Internal-only refinement for SOLVING_RESOLUTION.
     * Null for all non-resolution demand categories.
     */
    val resolutionProfile: ResolutionProfileV1? = null
) {
    fun toJson(): JSONObject = JSONObject().apply {
        val normalizedCategoryName = when (category) {
            ReplyDemandCategoryV1.CONFIRMING_VALIDATION_SUMMARY -> "CONFIRM_STATUS_SUMMARY"
            ReplyDemandCategoryV1.FREE_TALK_IN_GRID_SESSION -> "META_STATE_ANSWER"
            else -> category.name
        }

        put("category", normalizedCategoryName)
        put("raw_category", category.name)
        put("reason", reason)
        put("phase", phase ?: JSONObject.NULL)
        put("pending_kind", pendingKind ?: JSONObject.NULL)
        put("story_stage", storyStage ?: JSONObject.NULL)
        put("opening_turn", openingTurn)
        put("canonical_position_kind", canonicalPositionKind ?: JSONObject.NULL)
        put("canonical_legal", canonicalLegal ?: JSONObject.NULL)
        put("setup_profile", setupProfile?.name ?: JSONObject.NULL)
        put(
            "confrontation_proof_profile",
            confrontationProofProfile?.name ?: JSONObject.NULL
        )
        put(
            "resolution_profile",
            resolutionProfile?.name ?: JSONObject.NULL
        )
    }
}

/**
 * Phase 1 placeholder for the later contract-driven assembly architecture.
 * Not used yet for request shaping; added now so later phases can extend cleanly.
 */
data class ReplyBudgetV1(
    val softCharBudget: Int? = null,
    val softTokenBudget: Int? = null
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("soft_char_budget", softCharBudget ?: JSONObject.NULL)
        put("soft_token_budget", softTokenBudget ?: JSONObject.NULL)
    }
}

/**
 * Phase 2 — demand contract model.
 *
 * In later phases, every reply demand category will map to one of these
 * contracts, which will define:
 * - which prompt modules are required
 * - which supply channels are required / optional
 * - which channels are forbidden
 * - what the soft budget is
 *
 * Phase 2 only defines the model and serialization.
 */
data class ReplyDemandContractV1(
    val demandCategory: ReplyDemandCategoryV1,
    val requiredPromptModules: Set<PromptModuleV1> = emptySet(),
    val requiredChannels: Set<ReplySupplyChannelV1> = emptySet(),
    val optionalChannels: Set<ReplySupplyChannelV1> = emptySet(),
    val forbiddenChannels: Set<ReplySupplyChannelV1> = emptySet(),
    val budget: ReplyBudgetV1 = ReplyBudgetV1(),
    val notes: String? = null
) {
    fun toJson(): JSONObject = JSONObject().apply {
        val normalizedDemandCategoryName = when (demandCategory) {
            ReplyDemandCategoryV1.CONFIRMING_VALIDATION_SUMMARY -> "CONFIRM_STATUS_SUMMARY"
            ReplyDemandCategoryV1.FREE_TALK_IN_GRID_SESSION -> "META_STATE_ANSWER"
            else -> demandCategory.name
        }

        put("demand_category", normalizedDemandCategoryName)
        put("raw_demand_category", demandCategory.name)

        put(
            "required_prompt_modules",
            JSONArray().apply { requiredPromptModules.forEach { put(it.name) } }
        )
        put(
            "required_channels",
            JSONArray().apply { requiredChannels.forEach { put(it.name) } }
        )
        put(
            "optional_channels",
            JSONArray().apply { optionalChannels.forEach { put(it.name) } }
        )
        put(
            "forbidden_channels",
            JSONArray().apply { forbiddenChannels.forEach { put(it.name) } }
        )

        put("budget", budget.toJson())
        put("notes", notes ?: JSONObject.NULL)
    }
}

/**
 * Phase 2 — assembly-plan placeholder.
 *
 * This is intentionally small for now. Phase 6 will extend it when the actual
 * planner is introduced.
 */
data class ReplyAssemblyPlanV1(
    val demand: ReplyDemandResolutionV1,
    val contract: ReplyDemandContractV1? = null,
    val selectedPromptModules: List<PromptModuleV1> = emptyList(),
    val selectedChannels: List<ReplySupplyChannelV1> = emptyList(),
    val rolloutMode: String = "legacy_payload",
    val notes: String? = null
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("demand", demand.toJson())
        put("contract", contract?.toJson() ?: JSONObject.NULL)
        put(
            "selected_prompt_modules",
            JSONArray().apply { selectedPromptModules.forEach { put(it.name) } }
        )
        put(
            "selected_channels",
            JSONArray().apply { selectedChannels.forEach { put(it.name) } }
        )
        put("rollout_mode", rolloutMode)
        put("notes", notes ?: JSONObject.NULL)
    }
}

data class ReplyStyleV1(
    val voice: ReplyVoiceV1 = ReplyVoiceV1.friendly,
    val tone: ReplyToneV1 = ReplyToneV1.natural,
    val maxWords: Int = 200,
    val askOneQuestionMax: Boolean = true
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("voice", voice.name)
        put("tone", tone.name)
        put("max_words", maxWords)
        put("ask_one_question_max", askOneQuestionMax)
    }

    companion object {
        fun parse(o: JSONObject?): ReplyStyleV1 {
            if (o == null) return ReplyStyleV1()
            val voice = runCatching { ReplyVoiceV1.valueOf(o.optString("voice", "friendly")) }
                .getOrDefault(ReplyVoiceV1.friendly)
            val tone = runCatching { ReplyToneV1.valueOf(o.optString("tone", "natural")) }
                .getOrDefault(ReplyToneV1.natural)
            val maxWords = o.optInt("max_words", 200).coerceIn(40, 500)
            val askOne = o.optBoolean("ask_one_question_max", true)
            return ReplyStyleV1(voice = voice, tone = tone, maxWords = maxWords, askOneQuestionMax = askOne)
        }
    }
}

data class ReplyStoryCtxV1(
    val present: Boolean = false,
    val stage: String? = null,                 // "SETUP"|"CONFRONTATION"|"RESOLUTION"
    val stepId: String? = null,
    val gridHash12: String? = null,
    val atomsCount: Int? = null,
    val focusAtomIndex: Int? = null,
    val discussedAtomIndices: List<Int> = emptyList(),
    val readyForCommit: Boolean? = null,
    val canonicalPositionKind: String? = null,
    val canonicalHeadKind: String? = null,
    val canonicalPendingKind: String? = null
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("present", present)
        if (stage != null) put("stage", stage) else put("stage", JSONObject.NULL)
        if (stepId != null) put("step_id", stepId) else put("step_id", JSONObject.NULL)
        if (gridHash12 != null) put("grid_hash12", gridHash12) else put("grid_hash12", JSONObject.NULL)
        if (atomsCount != null) put("atoms_count", atomsCount) else put("atoms_count", JSONObject.NULL)
        if (focusAtomIndex != null) put("focus_atom_index", focusAtomIndex) else put("focus_atom_index", JSONObject.NULL)

        put(
            "discussed_atom_indices",
            JSONArray().apply { discussedAtomIndices.forEach { put(it) } }
        )

        if (readyForCommit != null) put("ready_for_commit", readyForCommit)
        else put("ready_for_commit", JSONObject.NULL)

        if (canonicalPositionKind != null) put("canonical_position_kind", canonicalPositionKind)
        else put("canonical_position_kind", JSONObject.NULL)

        if (canonicalHeadKind != null) put("canonical_head_kind", canonicalHeadKind)
        else put("canonical_head_kind", JSONObject.NULL)

        if (canonicalPendingKind != null) put("canonical_pending_kind", canonicalPendingKind)
        else put("canonical_pending_kind", JSONObject.NULL)
    }

    companion object {
        fun parse(o: JSONObject?): ReplyStoryCtxV1? {
            if (o == null) return null
            val present = o.optBoolean("present", false)

            val discussed = ArrayList<Int>()
            val arr = o.optJSONArray("discussed_atom_indices")
            if (arr != null) {
                for (i in 0 until arr.length()) {
                    if (!arr.isNull(i)) discussed.add(arr.optInt(i))
                }
            }

            return ReplyStoryCtxV1(
                present = present,
                stage = o.optStringOrNull("stage"),
                stepId = o.optStringOrNull("step_id"),
                gridHash12 = o.optStringOrNull("grid_hash12"),
                atomsCount = o.optIntOrNull("atoms_count"),
                focusAtomIndex = o.optIntOrNull("focus_atom_index"),
                discussedAtomIndices = discussed,
                readyForCommit = if (o.has("ready_for_commit") && !o.isNull("ready_for_commit")) o.optBoolean("ready_for_commit") else null,
                canonicalPositionKind = o.optStringOrNull("canonical_position_kind"),
                canonicalHeadKind = o.optStringOrNull("canonical_head_kind"),
                canonicalPendingKind = o.optStringOrNull("canonical_pending_kind")
            )
        }
    }
}

data class ReplyTurnCtxV1(
    val turnId: Int,
    val tickId: Int = 2,
    val mode: String,
    val phase: String,
    val userText: String,
    val pendingBefore: String? = null,
    val pendingAfter: String? = null,
    val focusBefore: Int? = null,
    val focusAfter: Int? = null,
    val turnAuthorityOwner: String? = null,
    val turnAuthorityReason: String? = null,
    val turnPendingStatus: String? = null,
    val turnResumedPendingKind: String? = null,
    val turnRouteReturnAllowed: Boolean = false,
    val turnBoundaryStatus: String? = null,
    val turnBoundaryReason: String? = null,

    // Series 5 — authoritative reply demand selected upstream by conductor.
    val replyDemandCategory: ReplyDemandCategoryV1? = null,

    // ✅ NEW: explicit SOLVING story header (Driver–Maps contract)
    val story: ReplyStoryCtxV1? = null
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("turn_id", turnId)
        put("tick_id", tickId)
        put("mode", mode)
        put("phase", phase)
        put("user_text", userText)
        if (pendingBefore != null) put("pending_before", pendingBefore) else put("pending_before", JSONObject.NULL)
        if (pendingAfter != null) put("pending_after", pendingAfter) else put("pending_after", JSONObject.NULL)
        if (focusBefore != null) put("focus_before", focusBefore) else put("focus_before", JSONObject.NULL)
        if (focusAfter != null) put("focus_after", focusAfter) else put("focus_after", JSONObject.NULL)
        if (turnAuthorityOwner != null) put("turn_authority_owner", turnAuthorityOwner) else put("turn_authority_owner", JSONObject.NULL)
        if (turnAuthorityReason != null) put("turn_authority_reason", turnAuthorityReason) else put("turn_authority_reason", JSONObject.NULL)
        if (turnPendingStatus != null) put("turn_pending_status", turnPendingStatus) else put("turn_pending_status", JSONObject.NULL)
        if (turnResumedPendingKind != null) put("turn_resumed_pending_kind", turnResumedPendingKind) else put("turn_resumed_pending_kind", JSONObject.NULL)
        put("turn_route_return_allowed", turnRouteReturnAllowed)
        if (turnBoundaryStatus != null) put("turn_boundary_status", turnBoundaryStatus) else put("turn_boundary_status", JSONObject.NULL)
        if (turnBoundaryReason != null) put("turn_boundary_reason", turnBoundaryReason) else put("turn_boundary_reason", JSONObject.NULL)
        if (replyDemandCategory != null) put("reply_demand_category", replyDemandCategory.name) else put("reply_demand_category", JSONObject.NULL)
        if (story != null) put("story", story.toJson()) else put("story", JSONObject.NULL)
    }

    companion object {
        fun parse(o: JSONObject?): ReplyTurnCtxV1 {
            if (o == null) {
                return ReplyTurnCtxV1(
                    turnId = -1,
                    tickId = 2,
                    mode = "GRID_SESSION",
                    phase = "UNKNOWN",
                    userText = "",
                    story = null
                )
            }
            return ReplyTurnCtxV1(
                turnId = o.optInt("turn_id", -1),
                tickId = o.optInt("tick_id", 2),
                mode = o.optString("mode", "GRID_SESSION"),
                phase = o.optString("phase", "UNKNOWN"),
                userText = o.optString("user_text", ""),
                pendingBefore = o.optStringOrNull("pending_before"),
                pendingAfter = o.optStringOrNull("pending_after"),
                focusBefore = o.optIntOrNull("focus_before"),
                focusAfter = o.optIntOrNull("focus_after"),
                turnAuthorityOwner = o.optStringOrNull("turn_authority_owner"),
                turnAuthorityReason = o.optStringOrNull("turn_authority_reason"),
                turnPendingStatus = o.optStringOrNull("turn_pending_status"),
                turnResumedPendingKind = o.optStringOrNull("turn_resumed_pending_kind"),
                turnRouteReturnAllowed = o.optBoolean("turn_route_return_allowed", false),
                turnBoundaryStatus = o.optStringOrNull("turn_boundary_status"),
                turnBoundaryReason = o.optStringOrNull("turn_boundary_reason"),
                replyDemandCategory = runCatching {
                    ReplyDemandCategoryV1.valueOf(o.optString("reply_demand_category", ""))
                }.getOrNull(),
                story = ReplyStoryCtxV1.parse(o.optJSONObjectOrNull("story"))
            )
        }
    }
}

data class MutationAppliedV1(
    val cell: String,
    val op: MutationOpV1,
    val value: Int? = null
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("cell", cell)
        put("op", op.name)
        if (value != null) put("value", value) else put("value", JSONObject.NULL)
    }

    companion object {
        fun parse(o: JSONObject?): MutationAppliedV1? {
            if (o == null) return null
            val cell = o.optString("cell", "").takeIf { it.isNotBlank() } ?: return null
            val op = MutationOpV1.parse(o.optString("op", null)) ?: return null
            val value = if (o.has("value") && !o.isNull("value")) o.optInt("value") else null
            return MutationAppliedV1(cell = cell, op = op, value = value)
        }
    }
}

data class ReplyDecisionV1(
    val decisionKind: DecisionKindV1,
    val summary: String,
    val mutationApplied: MutationAppliedV1? = null
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("decision_kind", decisionKind.name)
        put("summary", summary)
        if (mutationApplied != null) put("mutation_applied", mutationApplied.toJson())
        else put("mutation_applied", JSONObject.NULL)
    }

    companion object {
        fun parse(o: JSONObject?): ReplyDecisionV1 {
            if (o == null) {
                return ReplyDecisionV1(decisionKind = DecisionKindV1.NO_OP, summary = "", mutationApplied = null)
            }
            val dk = runCatching { DecisionKindV1.valueOf(o.optString("decision_kind", "NO_OP")) }
                .getOrDefault(DecisionKindV1.NO_OP)
            val summary = o.optString("summary", "")
            val mut = o.optJSONObject("mutation_applied")?.let { MutationAppliedV1.parse(it) }
            return ReplyDecisionV1(decisionKind = dk, summary = summary, mutationApplied = mut)
        }
    }
}

data class FactBundleV1(
    val type: Type,
    val payload: JSONObject
) {
    enum class Type {

        // -------------------------
        // Legacy v1 bundles (keep)
        // -------------------------
        FOCUS_CELL_SNAPSHOT,
        CONFLICT_EXPLANATION,
        MISMATCH_EXPLANATION,
        REGION_SUMMARY,
        SOLVABILITY_STATUS,
        // Phase 2: NEXT_PENDING_PROMPT is deprecated (do not emit).
        NEXT_PENDING_PROMPT,

        // Phase 2: non-CTA pending context (prompt + targets + expected answer kind).
        PENDING_CONTEXT_V1,

        RETAKE_RECOMMENDATION,
        RECENT_MUTATION_RESULT,

        // -------------------------
        // CONFIRMING agenda packets (NEW)
        // -------------------------
        CONFIRMING_RETAKE_PACKET,
        CONFIRMING_MISMATCH_PACKET,
        CONFIRMING_CONFLICT_PACKET,
        CONFIRMING_VISUAL_VERIFY_PACKET,
        CONFIRMING_NOT_UNIQUE_PACKET,
        CONFIRMING_FINALIZE_PACKET,

        // -------------------------
        // Grid inspection (A)
        // -------------------------
        GRID_SNAPSHOT,
        CELL_STATUS_BUNDLE,
        HOUSE_STATUS_BUNDLE,
        DIGIT_DISTRIBUTION_GLOBAL,
        DIGIT_LOCATIONS_BUNDLE,

        // NEW: ranked houses by completion (rows/cols/boxes)
        HOUSES_COMPLETION_RANKING,

        // Global house completion (NEW)
        COMPLETE_HOUSES_BUNDLE,

        CHANGELOG_RECENT,
        FOCUS_CONTEXT,

        // -------------------------
        // Validation & scan quality (B)
        // -------------------------
        STRUCTURAL_VALIDITY,
        CONFLICT_SET,
        DUPLICATES_BY_HOUSE,
        UNRESOLVED_SET,

        // NEW: mismatch global detail
        MISMATCH_SET,

        OCR_CONFIDENCE_CELL,
        OCR_CONFIDENCE_SUMMARY,
        SEAL_STATUS,

        // -------------------------
        // Candidates (A2/E3 support)
        // -------------------------
        CANDIDATE_STATE_CELL,
        CANDIDATE_COUNTS_GLOBAL,
        CELLS_WITH_N_CANDS_SET,
        BIVALUE_CELLS_SET,
        HOUSE_CANDIDATE_MAP,
        DIGIT_CANDIDATE_FREQUENCY,

        // Phase 0 — Detour packet family (scaffold only; no emitters yet)
        // -------------------------
        STEP_CLARIFICATION_PACKET_V1,
        PROOF_CHALLENGE_PACKET_V1,
        TARGET_CELL_QUERY_PACKET_V1,
        NEIGHBOR_CELL_QUERY_PACKET_V1,
        CANDIDATE_STATE_PACKET_V1,
        USER_REASONING_CHECK_PACKET_V1,
        ALTERNATIVE_TECHNIQUE_PACKET_V1,
        RETURN_TO_ROUTE_PACKET_V1,

        // -------------------------
        // SV-2 — Solver-backed detour packet family
        // -------------------------
        SOLVER_CELL_CANDIDATES_PACKET_V1,
        SOLVER_CELLS_CANDIDATES_PACKET_V1,
        SOLVER_HOUSE_CANDIDATE_MAP_PACKET_V1,
        SOLVER_CELL_DIGIT_BLOCKERS_PACKET_V1,
        SOLVER_REASONING_CHECK_PACKET_V1,
        SOLVER_ALTERNATIVE_TECHNIQUE_PACKET_V1,
        SOLVER_TECHNIQUE_SCOPE_CHECK_PACKET_V1,
        SOLVER_LOCAL_MOVE_SEARCH_PACKET_V1,
        SOLVER_ROUTE_COMPARISON_PACKET_V1,
        SOLVER_SCOPED_SUPPORT_PACKET_V1,

        // -------------------------
        // Wave 1 permanent-design — canonical normalized detour truth
        // -------------------------
        NORMALIZED_DETOUR_MOVE_PROOF_V1,
        NORMALIZED_DETOUR_LOCAL_INSPECTION_V1,
        NORMALIZED_DETOUR_PROPOSAL_VERDICT_V1,

        // -------------------------
        // Wave 1 permanent-design — native detour narrative context
        // -------------------------
        DETOUR_NARRATIVE_CONTEXT_V1,

        // -------------------------
        // Solvability/solving (E)
        PROGRESS_METRICS,
        DIFFICULTY_ESTIMATE,


        // -------------------------
        // SOLVING FactBundles v1 (Milestone 6)
        // -------------------------

        /*
         * SOLVING_STEP_PACKET payload (top-level):
         * {
         *   "step_id": "step:123",
         *   "grid_hash12_before": "abc123...",
         *   "grid_hash12_after": "def456..." | null,
         *   "technique": "Hidden Single",
         *   "target_cell": "r4c7",
         *   "target_digit": 6,
         *   "bundles": ["NEXT_MOVE_RECOMMENDATION","TECHNIQUE_FINDINGS","CANDIDATES_SNAPSHOT","OVERLAY_FRAMES","TEACHING_CARD"],
         *   "note": "All bundles are computed deterministically in-app when step becomes READY."
         * }
         *
         * OVERLAY_FRAMES payload:
         * {
         *   "frames": [
         *     {
         *       "id": "frame:1",
         *       "title": "Scan the box",
         *       "highlights": [
         *         {"kind":"cell","cell":"r4c7"},
         *         {"kind":"house","house":{"kind":"BOX","index":5}},
         *         {"kind":"candidate","cell":"r4c7","digit":6}
         *       ]
         *     }
         *   ]
         * }
         *
         * TEACHING_CARD payload:
         * { "recognition": [...], "application": [...], "pitfalls": [...] }
         */

        SOLVING_STEP_PACKET,        // top-level “proof packet” for the next step (legacy v1)
        CANDIDATES_SNAPSHOT,        // relevant cells before/after (or before only)
        OVERLAY_FRAMES,             // overlay frames + highlights to draw
        TEACHING_CARD,              // recognition / application / pitfalls

        // -------------------------
        // Phase 0 — Frozen Roadmap v1 contracts (NEW)
        // -------------------------
        CTA_PACKET_V1,              // semantic CTA options (no scripts)
        RECOVERY_PACKET_V1,         // recovery facts + semantic CTAs
        SOLVING_STEP_PACKET_V1,     // frozen step packet contract (journey + evidence + overlays + CTAs)
        TEACHING_CARD_V1,           // frozen teaching card contract

        NEXT_MOVE_RECOMMENDATION,
        TECHNIQUE_FINDINGS,
        NO_PROGRESS_REASON,
        UNIQUENESS_WITNESS,

        // -------------------------
        // Explanation / proof (F)
        // -------------------------
        MOVE_JUSTIFICATION_GRAPH,
        ELIMINATION_JUSTIFICATION,
        USER_REASONING_CHECK,

        // -------------------------
        // Workflow / prefs / meta (C/H)
        // -------------------------
        SUMMARY_DASHBOARD,
        PREFERENCES_SNAPSHOT,
        CAPABILITIES_SNAPSHOT,
        LIMITATIONS,

        // ✅ Player-language bridge (NEW)
        GLOSSARY_BUNDLE,

        // -------------------------
        // Milestone 7 — Tick2 driver-voice contract + spoiler policy
        // -------------------------
        PROMPT_CONTRACT_V1,      // hard rules: no invention, must cite proof bundles
        SPOILER_POLICY_V1,       // suppress digits unless user explicitly chose reveal

        // -------------------------
        // ✅ Phase 1 — Narrative Story explicit choreography (NEW)
        // -------------------------
        STORY_CONTEXT_V1,        // stage + atom scope + required CTA options (Maps -> Driver leash)

        // -------------------------
        // ✅ Phase 2 — Canonical stage-spoken packets (NEW)
        // -------------------------
        SETUP_REPLY_PACKET_V1,            // compact setup-only packet derived upstream for SOLVING_SETUP
        CONFRONTATION_REPLY_PACKET_V1,    // compact confrontation-only packet derived upstream for SOLVING_CONFRONTATION
        RESOLUTION_REPLY_PACKET_V1,       // compact resolution-only packet derived upstream for SOLVING_RESOLUTION

        // -------------------------
        // ✅ Phase 4 — Story→Story transition bridge (NEW)
        // -------------------------
        HANDOVER_NOTE_V1,        // structured contrast/continuity between last step technique and next step technique

        STORY_SIGNATURE_V1,      // hash + expanded fields for audit (repeat guard + scope)

        // -------------------------
        // ✅ Phase 5 — Step scorecard evidence bundle (NEW)
        // -------------------------
        STEP_STORY_SCORECARD_V1, // delivered stages/proof depth/commit for this stepId

        // -------------------------
        // Phase 0 — Observability: per-turn contract snapshot (NEW)
        // -------------------------
        TURN_CONTRACT_SNAPSHOT_V1,

        // Fallback (discourage)
        OTHER
    }

    fun toJson(): JSONObject = JSONObject().apply {
        put("type", type.name)
        put("payload", payload)
    }

    companion object {
        fun jsonArray(bundles: List<FactBundleV1>): JSONArray =
            JSONArray().apply { bundles.forEach { put(it.toJson()) } }

        fun parseArray(arr: JSONArray?): List<FactBundleV1> {
            if (arr == null) return emptyList()
            val out = mutableListOf<FactBundleV1>()
            for (i in 0 until arr.length()) {
                val o = arr.optJSONObject(i) ?: continue
                val t = runCatching { Type.valueOf(o.optString("type", "OTHER")) }.getOrDefault(Type.OTHER)
                val p = o.optJSONObject("payload") ?: JSONObject()
                out.add(FactBundleV1(type = t, payload = p))
            }
            return out
        }
    }
}



// -----------------------------------------------------------------------------
// Phase 0 — Frozen Roadmap v1: Fact bundle contracts (typed helpers)
// -----------------------------------------------------------------------------
//
// Invariant: These are FACTS + SEMANTICS, never scripts.
// The LLM is responsible for natural language generation.

data class CtaOptionV1(
    val id: String,
    val label: String? = null,
    val semantics: JSONObject? = null
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("id", id)
        if (label != null) put("label", label) else put("label", JSONObject.NULL)
        if (semantics != null) put("semantics", semantics) else put("semantics", JSONObject.NULL)
    }
}

data class CtaPacketV1(
    val slot: String,
    val options: List<CtaOptionV1>
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", "cta_packet_v1")
        put("slot", slot)
        put("options", JSONArray().apply { options.forEach { put(it.toJson()) } })
    }
}

enum class CtaFamilyV1 {
    APP_SETUP_DISCOVERY,
    APP_CONFRONTATION_PROOF_STEP,
    APP_RESOLUTION_COMMIT,
    APP_POST_COMMIT_CONTINUE,
    USER_DETOUR_FOLLOWUP_OR_RETURN,
    USER_DETOUR_FOLLOWUP_ONLY,
    USER_ROUTE_CONTROL_CONFIRM,
    USER_LOCAL_REPAIR_CONFIRM,
    UNKNOWN
}

enum class CtaRouteMomentV1 {
    SOLVING_SETUP,
    SOLVING_CONFRONTATION,
    SOLVING_RESOLUTION,
    SOLVING_POST_COMMIT,
    USER_DETOUR_COMPLETE,
    USER_DETOUR_PARTIAL,
    USER_ROUTE_MUTATION_DECISION,
    USER_LOCAL_REPAIR_DECISION,
    CONFIRMING,
    SEALING,
    UNKNOWN
}

enum class CtaResponseTypeV1 {
    YES_NO,
    CHOOSE_ONE_OF_TWO,
    NAME_DIGIT,
    NAME_CELL,
    EXPLAIN_WHY,
    CONFIRM_REASONING,
    PERMISSION_TO_APPLY,
    CONTINUE_OR_PAUSE,
    CLARIFY_SCOPE,
    ASK_FOLLOWUP,
    UNKNOWN
}

enum class CtaAskModeV1 {
    DIRECT_QUESTION,
    BINARY_CHOICE,
    GUIDED_PROMPT,
    PERMISSION_ASK,
    CLARIFYING_QUESTION,
    UNKNOWN
}

enum class CtaToneStyleV1 {
    WARM_GUIDE,
    EXPLORATORY,
    DECISIVE
}

data class CtaPolicyV1(
    val family: CtaFamilyV1,
    val expectedResponseType: CtaResponseTypeV1,
    val askMode: CtaAskModeV1,
    val allowInternalJargon: Boolean = false,
    val mustOfferFollowUpChoice: Boolean = false,
    val mustOfferReturnChoice: Boolean = false,
    val mustNotAdvanceStage: Boolean = false,
    val mustReferenceFocusScope: Boolean = false
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("family", family.name)
        put("expected_response_type", expectedResponseType.name)
        put("ask_mode", askMode.name)
        put("allow_internal_jargon", allowInternalJargon)
        put("must_offer_followup_choice", mustOfferFollowUpChoice)
        put("must_offer_return_choice", mustOfferReturnChoice)
        put("must_not_advance_stage", mustNotAdvanceStage)
        put("must_reference_focus_scope", mustReferenceFocusScope)
    }

    companion object {
        fun parse(o: JSONObject?): CtaPolicyV1? {
            if (o == null) return null

            fun <E : Enum<E>> parseEnum(raw: String?, values: Array<E>, fallback: E): E =
                raw?.trim()?.takeIf { it.isNotBlank() }?.let { key ->
                    values.firstOrNull { it.name.equals(key, ignoreCase = true) }
                } ?: fallback

            return CtaPolicyV1(
                family = parseEnum(
                    o.optString("family", null),
                    CtaFamilyV1.values(),
                    CtaFamilyV1.UNKNOWN
                ),
                expectedResponseType = parseEnum(
                    o.optString("expected_response_type", null),
                    CtaResponseTypeV1.values(),
                    CtaResponseTypeV1.UNKNOWN
                ),
                askMode = parseEnum(
                    o.optString("ask_mode", null),
                    CtaAskModeV1.values(),
                    CtaAskModeV1.UNKNOWN
                ),
                allowInternalJargon = o.optBoolean("allow_internal_jargon", false),
                mustOfferFollowUpChoice = o.optBoolean("must_offer_followup_choice", false),
                mustOfferReturnChoice = o.optBoolean("must_offer_return_choice", false),
                mustNotAdvanceStage = o.optBoolean("must_not_advance_stage", false),
                mustReferenceFocusScope = o.optBoolean("must_reference_focus_scope", false)
            )
        }
    }
}

data class CtaContractV1(
    val family: CtaFamilyV1,
    val ownerKind: String,
    val routeMoment: CtaRouteMomentV1,
    val expectedResponseType: CtaResponseTypeV1,
    val askMode: CtaAskModeV1,
    val closureIntent: String? = null,
    val bridgeIntent: String? = null,
    val askIntent: String? = null,
    val focusCellRef: String? = null,
    val focusHouseRef: String? = null,
    val focusDigit: Int? = null,
    val techniqueName: String? = null,
    val allowFollowUp: Boolean = true,
    val allowReturnToRoute: Boolean = false,
    val allowRouteMutation: Boolean = false,
    val bannedPhrases: List<String> = emptyList(),
    val toneStyle: CtaToneStyleV1 = CtaToneStyleV1.WARM_GUIDE,
    val policy: CtaPolicyV1? = null
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("family", family.name)
        put("owner_kind", ownerKind)
        put("route_moment", routeMoment.name)
        put("expected_response_type", expectedResponseType.name)
        put("ask_mode", askMode.name)
        put("closure_intent", closureIntent ?: JSONObject.NULL)
        put("bridge_intent", bridgeIntent ?: JSONObject.NULL)
        put("ask_intent", askIntent ?: JSONObject.NULL)
        put("focus_cell_ref", focusCellRef ?: JSONObject.NULL)
        put("focus_house_ref", focusHouseRef ?: JSONObject.NULL)
        put("focus_digit", focusDigit ?: JSONObject.NULL)
        put("technique_name", techniqueName ?: JSONObject.NULL)
        put("allow_followup", allowFollowUp)
        put("allow_return_to_route", allowReturnToRoute)
        put("allow_route_mutation", allowRouteMutation)
        put("banned_phrases", JSONArray().apply { bannedPhrases.forEach { put(it) } })
        put("tone_style", toneStyle.name)
        put("policy", policy?.toJson() ?: JSONObject.NULL)
    }

    companion object {
        fun parse(o: JSONObject?): CtaContractV1? {
            if (o == null) return null

            fun <E : Enum<E>> parseEnum(raw: String?, values: Array<E>, fallback: E): E =
                raw?.trim()?.takeIf { it.isNotBlank() }?.let { key ->
                    values.firstOrNull { it.name.equals(key, ignoreCase = true) }
                } ?: fallback

            val banned = mutableListOf<String>()
            val bannedArr = o.optJSONArray("banned_phrases")
            if (bannedArr != null) {
                for (i in 0 until bannedArr.length()) {
                    bannedArr.optString(i)?.takeIf { it.isNotBlank() }?.let { banned += it }
                }
            }

            return CtaContractV1(
                family = parseEnum(
                    o.optString("family", null),
                    CtaFamilyV1.values(),
                    CtaFamilyV1.UNKNOWN
                ),
                ownerKind = o.optString("owner_kind", "NONE"),
                routeMoment = parseEnum(
                    o.optString("route_moment", null),
                    CtaRouteMomentV1.values(),
                    CtaRouteMomentV1.UNKNOWN
                ),
                expectedResponseType = parseEnum(
                    o.optString("expected_response_type", null),
                    CtaResponseTypeV1.values(),
                    CtaResponseTypeV1.UNKNOWN
                ),
                askMode = parseEnum(
                    o.optString("ask_mode", null),
                    CtaAskModeV1.values(),
                    CtaAskModeV1.UNKNOWN
                ),
                closureIntent = o.optString("closure_intent", null).takeIf { !it.isNullOrBlank() },
                bridgeIntent = o.optString("bridge_intent", null).takeIf { !it.isNullOrBlank() },
                askIntent = o.optString("ask_intent", null).takeIf { !it.isNullOrBlank() },
                focusCellRef = o.optString("focus_cell_ref", null).takeIf { !it.isNullOrBlank() },
                focusHouseRef = o.optString("focus_house_ref", null).takeIf { !it.isNullOrBlank() },
                focusDigit = if (o.has("focus_digit") && !o.isNull("focus_digit")) o.optInt("focus_digit") else null,
                techniqueName = o.optString("technique_name", null).takeIf { !it.isNullOrBlank() },
                allowFollowUp = o.optBoolean("allow_followup", true),
                allowReturnToRoute = o.optBoolean("allow_return_to_route", false),
                allowRouteMutation = o.optBoolean("allow_route_mutation", false),
                bannedPhrases = banned,
                toneStyle = parseEnum(
                    o.optString("tone_style", null),
                    CtaToneStyleV1.values(),
                    CtaToneStyleV1.WARM_GUIDE
                ),
                policy = CtaPolicyV1.parse(o.optJSONObject("policy"))
            )
        }
    }
}



// ============================================================
// Phase 4 — Claim Map (anti-invention contract)
// ============================================================
data class ClaimMapV1(
    val schema_version: String = "claim_map_v1",
    val claims: List<ClaimV1> = emptyList()
) {
    data class ClaimV1(
        val type: String,                 // "placement" | "elimination" | "status" (extend later)
        val cellIndex: Int? = null,        // 0..80
        val digit: Int? = null,            // 1..9
        val atomIndex: Int? = null,        // which narrative atom supports it
        val witnessIndex: Int? = null      // which witness inside that atom supports it (optional)
    )

    fun toJson(): org.json.JSONObject = org.json.JSONObject().apply {
        put("schema_version", schema_version)
        put("claims", org.json.JSONArray().apply {
            for (c in claims) {
                put(org.json.JSONObject().apply {
                    put("type", c.type)
                    put("cellIndex", c.cellIndex ?: org.json.JSONObject.NULL)
                    put("digit", c.digit ?: org.json.JSONObject.NULL)
                    put("atomIndex", c.atomIndex ?: org.json.JSONObject.NULL)
                    put("witnessIndex", c.witnessIndex ?: org.json.JSONObject.NULL)
                })
            }
        })
    }
}



data class RecoveryPacketV1(
    val kind: String,
    val facts: JSONObject,
    val cta: CtaPacketV1
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", "recovery_packet_v1")
        put("kind", kind)
        put("facts", facts)
        put("cta_packet", cta.toJson())
    }
}

/**
 * Frozen SOLVING_STEP_PACKET v1 (Roadmap v1).
 * This is the driver’s “ground truth packet” for a single solving turn.
 *
 * NOTE: This is a JSON-first contract for forward compatibility.
 * We keep a typed wrapper so callers can build it safely.
 */
data class SolvingStepPacketV1(
    val puzzle: JSONObject,
    val step: JSONObject,
    val evidence: JSONObject,
    val overlays: JSONObject
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", "solving_step_packet_v1")
        put("puzzle", puzzle)
        put("step", step)
        put("evidence", evidence)
        put("overlays", overlays)

        // Phase 2 invariant: CTA contract is ONLY FactBundleV1.Type.CTA_PACKET_V1.
        // Do not embed CTA here.
    }
}

data class TeachingCardV1(
    val technique: JSONObject,
    val definition: JSONObject,
    val howToSpot: JSONArray,
    val commonConfusions: JSONArray,
    val proofShapes: JSONArray,
    val styleHints: JSONObject = JSONObject()
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", "teaching_card_v1")
        put("technique", technique)
        put("definition", definition)
        put("how_to_spot", howToSpot)
        put("common_confusions", commonConfusions)
        put("proof_shapes", proofShapes)
        put("style_hints", styleHints)
    }
}

/**
 * Canonical compact spoken-setup packet.
 *
 * Purpose:
 * - give SOLVING_SETUP one small, archetype-aware payload
 * - derived upstream from current solving truth
 * - optimized for intro narration, not full proof delivery
 *
 * Notes:
 * - `bounded_trigger_rows` supports 0..N rows; for SUBSETS this naturally covers
 *   pair / triple / quad without special casing
 * - `support` carries small policy / coverage hints useful to later contract logic
 */
data class SetupReplyPacketV1(
    val setupProfile: SetupDemandProfileV1,
    val setupDoctrine: SetupNarrationDoctrineV1 = SetupNarrationDoctrineV1.NEUTRAL,
    val archetype: String? = null,
    val techniqueEntranceLabel: String? = null,
    val whyThisTechniqueNow: String? = null,
    val lensQuestion: String? = null,
    val patternBirthSummary: String? = null,
    val patternCompletionMoment: String? = null,
    val technique: JSONObject = JSONObject(),
    val target: JSONObject = JSONObject(),
    val orientation: JSONObject = JSONObject(),
    val lens: JSONObject = JSONObject(),
    val focus: JSONObject = JSONObject(),
    val patternStructure: JSONObject = JSONObject(),
    val triggerOverview: JSONObject = JSONObject(),
    val boundedTriggerRows: JSONArray = JSONArray(),
    val triggerStatement: JSONObject = JSONObject(),
    val bridge: JSONObject = JSONObject(),
    val setupOnlyLine: JSONObject = JSONObject(),
    val cta: JSONObject = JSONObject(),
    val support: JSONObject = JSONObject()
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", "setup_reply_packet_v1")
        put("setup_profile", setupProfile.name)
        put("setup_doctrine", setupDoctrine.name)
        put("archetype", archetype ?: JSONObject.NULL)
        put("technique_entrance_label", techniqueEntranceLabel ?: JSONObject.NULL)
        put("why_this_technique_now", whyThisTechniqueNow ?: JSONObject.NULL)
        put("lens_question", lensQuestion ?: JSONObject.NULL)
        put("pattern_birth_summary", patternBirthSummary ?: JSONObject.NULL)
        put("pattern_completion_moment", patternCompletionMoment ?: JSONObject.NULL)
        put("technique", technique)
        put("target", target)
        put("orientation", orientation)
        put("lens", lens)
        put("focus", focus)
        put("pattern_structure", patternStructure)
        put("trigger_overview", triggerOverview)
        put("bounded_trigger_rows", boundedTriggerRows)
        put("trigger_statement", triggerStatement)
        put("bridge", bridge)
        put("setup_only_line", setupOnlyLine)
        put("cta", cta)
        put("support", support)
    }
}

data class ConfrontationReplyPacketV1(
    val proofProfile: ConfrontationProofProfileV1,
    val confrontationDoctrine: ConfrontationNarrationDoctrineV1 = ConfrontationNarrationDoctrineV1.NEUTRAL,
    val archetype: String? = null,
    val technique: JSONObject = JSONObject(),
    val target: JSONObject = JSONObject(),
    val triggerReference: JSONObject = JSONObject(),
    val triggerEffect: JSONObject = JSONObject(),
    val targetResolutionTruth: JSONObject = JSONObject(),
    val targetProofRows: JSONArray = JSONArray(),
    val collapse: JSONObject = JSONObject(),
    val preCommitLine: JSONObject = JSONObject(),
    val cta: JSONObject = JSONObject(),
    val support: JSONObject = JSONObject()
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", "confrontation_reply_packet_v1")
        put("proof_profile", proofProfile.name)
        put("confrontation_doctrine", confrontationDoctrine.name)
        put("archetype", archetype ?: JSONObject.NULL)
        put("technique", technique)
        put("target", target)
        put("trigger_reference", triggerReference)
        put("trigger_effect", triggerEffect)
        put("target_resolution_truth", targetResolutionTruth)
        put("target_proof_rows", targetProofRows)
        put("collapse", collapse)
        put("pre_commit_line", preCommitLine)
        put("cta", cta)
        put("support", support)
    }
}

data class ResolutionReplyPacketV1(
    val resolutionProfile: ResolutionProfileV1,
    val archetype: String? = null,
    val technique: JSONObject = JSONObject(),
    val commit: JSONObject = JSONObject(),
    val recap: JSONObject = JSONObject(),
    val techniqueContribution: JSONObject = JSONObject(),
    val fullResolutionBasis: JSONObject = JSONObject(),
    val finalForcing: JSONObject = JSONObject(),
    val honesty: JSONObject = JSONObject(),
    val causalRecap: JSONObject = JSONObject(),
    val structuralLesson: JSONObject = JSONObject(),
    val presentStateLine: JSONObject = JSONObject(),
    val postCommit: JSONObject = JSONObject(),
    val cta: JSONObject = JSONObject(),
    val support: JSONObject = JSONObject()
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", "resolution_reply_packet_v1")
        put("resolution_profile", resolutionProfile.name)
        put("archetype", archetype ?: JSONObject.NULL)
        put("technique", technique)
        put("commit", commit)
        put("recap", recap)
        put("technique_contribution", techniqueContribution)
        put("full_resolution_basis", fullResolutionBasis)
        put("final_forcing", finalForcing)
        put("honesty", honesty)
        put("causal_recap", causalRecap)
        put("structural_lesson", structuralLesson)
        put("present_state_line", presentStateLine)
        put("post_commit", postCommit)
        put("cta", cta)
        put("support", support)
    }
}

/**
 * Wave 1 — shared detour overlay policy scaffold.
 *
 * This is intentionally compact:
 * later phases can project richer overlay state into it without changing the
 * detour packet contracts.
 */
data class DetourOverlayPolicyV1(
    val overlayMode: DetourOverlayModeV1 = DetourOverlayModeV1.PRESERVE,
    val primaryFocusCells: List<String> = emptyList(),
    val primaryFocusHouses: List<String> = emptyList(),
    val secondaryFocusCells: List<String> = emptyList(),
    val deemphasizeCells: List<String> = emptyList(),
    val reasonForFocus: String? = null,
    val expectedSpokenAnchor: String? = null
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("overlay_mode", overlayMode.name)
        put("primary_focus_cells", JSONArray(primaryFocusCells))
        put("primary_focus_houses", JSONArray(primaryFocusHouses))
        put("secondary_focus_cells", JSONArray(secondaryFocusCells))
        put("deemphasize_cells", JSONArray(deemphasizeCells))
        put("reason_for_focus", reasonForFocus ?: JSONObject.NULL)
        put("expected_spoken_anchor", expectedSpokenAnchor ?: JSONObject.NULL)
    }
}

/**
 * Wave 1 — explicit handback contract for detour replies.
 *
 * The runtime is not wired to this yet. Phase F1 only freezes the shape.
 */
data class DetourHandbackPolicyV1(
    val handoverMode: DetourHandoverModeV1 = DetourHandoverModeV1.RETURN_TO_CURRENT_MOVE,
    val pausedRouteCheckpoint: String? = null,
    val returnTargetStage: String? = null,
    val returnTargetStepId: String? = null,
    val stayDetachedUntilUserSaysContinue: Boolean = false,
    val spokenReturnLine: String? = null
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("handover_mode", handoverMode.name)
        put("paused_route_checkpoint", pausedRouteCheckpoint ?: JSONObject.NULL)
        put("return_target_stage", returnTargetStage ?: JSONObject.NULL)
        put("return_target_step_id", returnTargetStepId ?: JSONObject.NULL)
        put("stay_detached_until_user_says_continue", stayDetachedUntilUserSaysContinue)
        put("spoken_return_line", spokenReturnLine ?: JSONObject.NULL)
    }
}

/**
 * Wave 1 — typed packet for MOVE_PROOF_OR_TARGET_EXPLANATION.
 *
 * This is the detour equivalent of a compact confrontation packet:
 * narrow, local, solver-backed, and route-safe.
 */
data class DetourMoveProofPacketV1(
    val demandCategory: DetourDemandCategoryV2 = DetourDemandCategoryV2.MOVE_PROOF_OR_TARGET_EXPLANATION,
    val proofProfile: DetourMoveProofProfileV1,
    val archetype: DetourNarrativeArchetypeV1 = DetourNarrativeArchetypeV1.LOCAL_PROOF_SPOTLIGHT,
    val anchorStepId: String? = null,
    val anchorStoryStage: String? = null,
    val targetCell: String? = null,
    val targetDigit: Int? = null,
    val houseScope: String? = null,
    val proofClaim: String? = null,
    val eliminationKind: String? = null,
    val targetBeforeState: JSONObject = JSONObject(),
    val targetAfterState: JSONObject = JSONObject(),
    val boundedProofRows: JSONArray = JSONArray(),
    val witnessRows: JSONArray = JSONArray(),
    val survivorSummary: JSONObject = JSONObject(),
    val directAnswerTruth: JSONObject = JSONObject(),
    val replyDiscipline: JSONObject = JSONObject(),
    val doctrineSurface: JSONObject = JSONObject(),
    val answerShape: String? = null,
    val orderedExplanationLadder: JSONArray = JSONArray(),
    val boundaryLine: String? = null,
    val handbackLine: String? = null,
    val support: JSONObject = JSONObject(),
    val overlayPolicy: DetourOverlayPolicyV1 = DetourOverlayPolicyV1(),
    val handbackPolicy: DetourHandbackPolicyV1 = DetourHandbackPolicyV1(),
    val answerBoundary: List<DetourAnswerBoundaryV1> = listOf(
        DetourAnswerBoundaryV1.DO_NOT_BECOME_BOARD_AUDIT,
        DetourAnswerBoundaryV1.DO_NOT_SWITCH_ROUTE,
        DetourAnswerBoundaryV1.DO_NOT_COMMIT_MOVE
    )
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", "detour_move_proof_packet_v1")
        put("demand_category", demandCategory.name)
        put("proof_profile", proofProfile.name)
        put("archetype", archetype.name)
        put("anchor_step_id", anchorStepId ?: JSONObject.NULL)
        put("anchor_story_stage", anchorStoryStage ?: JSONObject.NULL)
        put("target_cell", targetCell ?: JSONObject.NULL)
        put("target_digit", targetDigit ?: JSONObject.NULL)
        put("house_scope", houseScope ?: JSONObject.NULL)
        put("proof_claim", proofClaim ?: JSONObject.NULL)
        put("elimination_kind", eliminationKind ?: JSONObject.NULL)
        put("target_before_state", targetBeforeState)
        put("target_after_state", targetAfterState)
        put("bounded_proof_rows", boundedProofRows)
        put("witness_rows", witnessRows)
        put("survivor_summary", survivorSummary)
        put("direct_answer_truth", directAnswerTruth)
        put("reply_discipline", replyDiscipline)
        put("doctrine_surface", doctrineSurface)
        put("answer_shape", answerShape ?: JSONObject.NULL)
        put("ordered_explanation_ladder", orderedExplanationLadder)
        put("boundary_line", boundaryLine ?: JSONObject.NULL)
        put("handback_line", handbackLine ?: JSONObject.NULL)
        put("support", support)
        put("overlay_policy", overlayPolicy.toJson())
        put("handback_policy", handbackPolicy.toJson())
        put("answer_boundary", JSONArray(answerBoundary.map { it.name }))
    }
}

data class ProofChallengePacketV1(
    val challengeLane: String? = null,
    val questionFrame: JSONObject = JSONObject(),
    val storyFocus: JSONObject = JSONObject(),
    val storyQuestion: JSONObject = JSONObject(),
    val answerTruth: JSONObject = JSONObject(),
    val proofObject: JSONObject = JSONObject(),
    val proofMethod: JSONObject = JSONObject(),
    val narrativeArchetype: JSONObject = JSONObject(),
    val doctrine: JSONObject = JSONObject(),
    val speechSkeleton: JSONArray = JSONArray(),
    val actorModel: JSONObject = JSONObject(),
    val storyActors: JSONObject = JSONObject(),
    val proofLadder: JSONObject = JSONObject(),
    val proofOutcome: JSONObject = JSONObject(),
    val storyArc: JSONObject = JSONObject(),
    val microStagePlan: JSONObject = JSONObject(),
    val localProofGeometry: JSONObject = JSONObject(),
    val speechBoundary: JSONObject = JSONObject(),
    val closureContract: JSONObject = JSONObject(),
    val handbackContext: JSONObject = JSONObject(),
    val overlayPlan: JSONObject = JSONObject(),
    val visualLanguage: JSONObject = JSONObject(),
    val supportingFacts: JSONObject = JSONObject(),
    val debugSupport: JSONObject = JSONObject()
) {


    fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", "proof_challenge_packet_v5")
        put("packet_kind", "PROOF_CHALLENGE")
        put("challenge_lane", challengeLane ?: JSONObject.NULL)
        put("question_frame", questionFrame)
        put("story_focus", storyFocus)
        put("story_question", storyQuestion)
        put("answer_truth", answerTruth)
        put("proof_object", proofObject)
        put("proof_method", proofMethod)
        put("narrative_archetype", narrativeArchetype)
        put("doctrine", doctrine)
        put("speech_skeleton", speechSkeleton)
        put("actor_model", actorModel)
        put("story_actors", storyActors)
        put("proof_ladder", proofLadder)
        put("proof_outcome", proofOutcome)
        put("story_arc", storyArc)
        put("micro_stage_plan", microStagePlan)
        put("local_proof_geometry", localProofGeometry)
        put("speech_boundary", speechBoundary)
        put("closure_contract", closureContract)
        put("handback_context", handbackContext)
        put("overlay_plan", overlayPlan)
        put("visual_language", visualLanguage)
        put("supporting_facts", supportingFacts)
        if (debugSupport.length() > 0) {
            put("debug_support", debugSupport)
        }
    }
}

/**
 * Wave 1 — typed packet for LOCAL_GRID_INSPECTION.
 *
 * This is descriptive rather than argumentative: local readout over a bounded
 * scope.
 */
data class DetourLocalGridInspectionPacketV1(
    val demandCategory: DetourDemandCategoryV2 = DetourDemandCategoryV2.LOCAL_GRID_INSPECTION,
    val inspectionProfile: DetourLocalGridInspectionProfileV1,
    val archetype: DetourNarrativeArchetypeV1 = DetourNarrativeArchetypeV1.STATE_READOUT,
    val anchorStepId: String? = null,
    val scopeKind: String? = null,
    val scopeRef: String? = null,
    val focusCells: List<String> = emptyList(),
    val focusHouses: List<String> = emptyList(),
    val candidateState: JSONObject = JSONObject(),
    val digitLocations: JSONArray = JSONArray(),
    val localDelta: JSONObject = JSONObject(),
    val nearbyEffectsSummary: JSONObject = JSONObject(),
    val directAnswerTruth: JSONObject = JSONObject(),
    val replyDiscipline: JSONObject = JSONObject(),
    val doctrineSurface: JSONObject = JSONObject(),
    val answerShape: String? = null,
    val orderedExplanationLadder: JSONArray = JSONArray(),
    val boundaryLine: String? = null,
    val handbackLine: String? = null,
    val support: JSONObject = JSONObject(),
    val overlayPolicy: DetourOverlayPolicyV1 = DetourOverlayPolicyV1(),
    val handbackPolicy: DetourHandbackPolicyV1 = DetourHandbackPolicyV1(),
    val answerBoundary: List<DetourAnswerBoundaryV1> = listOf(
        DetourAnswerBoundaryV1.DO_NOT_BECOME_PROOF_LADDER,
        DetourAnswerBoundaryV1.DO_NOT_SWITCH_ROUTE,
        DetourAnswerBoundaryV1.DO_NOT_OPEN_NEW_DETOUR_TREE
    )
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", "detour_local_grid_inspection_packet_v1")
        put("demand_category", demandCategory.name)
        put("inspection_profile", inspectionProfile.name)
        put("archetype", archetype.name)
        put("anchor_step_id", anchorStepId ?: JSONObject.NULL)
        put("scope_kind", scopeKind ?: JSONObject.NULL)
        put("scope_ref", scopeRef ?: JSONObject.NULL)
        put("focus_cells", JSONArray(focusCells))
        put("focus_houses", JSONArray(focusHouses))
        put("candidate_state", candidateState)
        put("digit_locations", digitLocations)
        put("local_delta", localDelta)
        put("nearby_effects_summary", nearbyEffectsSummary)
        put("direct_answer_truth", directAnswerTruth)
        put("reply_discipline", replyDiscipline)
        put("doctrine_surface", doctrineSurface)
        put("answer_shape", answerShape ?: JSONObject.NULL)
        put("ordered_explanation_ladder", orderedExplanationLadder)
        put("boundary_line", boundaryLine ?: JSONObject.NULL)
        put("handback_line", handbackLine ?: JSONObject.NULL)
        put("support", support)
        put("overlay_policy", overlayPolicy.toJson())
        put("handback_policy", handbackPolicy.toJson())
        put("answer_boundary", JSONArray(answerBoundary.map { it.name }))
    }
}

/**
 * Wave 1 — typed packet for USER_PROPOSAL_VERDICT.
 *
 * This is distinct from proof explanation:
 * the assistant is grading the user's idea, not merely defending its own.
 */
data class DetourUserProposalVerdictPacketV1(
    val demandCategory: DetourDemandCategoryV2 = DetourDemandCategoryV2.USER_PROPOSAL_VERDICT,
    val proposalKind: DetourUserProposalKindV1,
    val archetype: DetourNarrativeArchetypeV1 = DetourNarrativeArchetypeV1.PROPOSAL_VERDICT,
    val proposalText: String? = null,
    val proposalScope: String? = null,
    val verdict: ReasoningVerdictV1 = ReasoningVerdictV1.UNKNOWN,
    val verdictReason: String? = null,
    val whatIsCorrect: JSONArray = JSONArray(),
    val whatIsIncorrect: JSONArray = JSONArray(),
    val missingCondition: String? = null,
    val routeAlignment: String? = null,
    val anchorStepId: String? = null,
    val anchorStoryStage: String? = null,
    val solverSupportRows: JSONArray = JSONArray(),
    val doctrineSurface: JSONObject = JSONObject(),
    val answerShape: String? = null,
    val orderedExplanationLadder: JSONArray = JSONArray(),
    val boundaryLine: String? = null,
    val handbackLine: String? = null,
    val support: JSONObject = JSONObject(),
    val overlayPolicy: DetourOverlayPolicyV1 = DetourOverlayPolicyV1(),
    val handbackPolicy: DetourHandbackPolicyV1 = DetourHandbackPolicyV1(),
    val answerBoundary: List<DetourAnswerBoundaryV1> = listOf(
        DetourAnswerBoundaryV1.DO_NOT_BECOME_BOARD_AUDIT,
        DetourAnswerBoundaryV1.DO_NOT_SWITCH_ROUTE,
        DetourAnswerBoundaryV1.DO_NOT_OPEN_NEW_DETOUR_TREE
    )
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", "detour_user_proposal_verdict_packet_v1")
        put("demand_category", demandCategory.name)
        put("proposal_kind", proposalKind.name)
        put("archetype", archetype.name)
        put("proposal_text", proposalText ?: JSONObject.NULL)
        put("proposal_scope", proposalScope ?: JSONObject.NULL)
        put("verdict", verdict.name)
        put("verdict_reason", verdictReason ?: JSONObject.NULL)
        put("what_is_correct", whatIsCorrect)
        put("what_is_incorrect", whatIsIncorrect)
        put("missing_condition", missingCondition ?: JSONObject.NULL)
        put("route_alignment", routeAlignment ?: JSONObject.NULL)
        put("anchor_step_id", anchorStepId ?: JSONObject.NULL)
        put("anchor_story_stage", anchorStoryStage ?: JSONObject.NULL)
        put("solver_support_rows", solverSupportRows)
        put("doctrine_surface", doctrineSurface)
        put("answer_shape", answerShape ?: JSONObject.NULL)
        put("ordered_explanation_ladder", orderedExplanationLadder)
        put("boundary_line", boundaryLine ?: JSONObject.NULL)
        put("handback_line", handbackLine ?: JSONObject.NULL)
        put("support", support)
        put("overlay_policy", overlayPolicy.toJson())
        put("handback_policy", handbackPolicy.toJson())
        put("answer_boundary", JSONArray(answerBoundary.map { it.name }))
    }
}

// Convenience wrappers for FactBundles
fun factCtaPacketV1(slot: String, options: List<CtaOptionV1>): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.CTA_PACKET_V1,
        payload = CtaPacketV1(slot = slot, options = options).toJson()
    )




fun factRecoveryPacketV1(kind: String, facts: JSONObject, cta: CtaPacketV1): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.RECOVERY_PACKET_V1,
        payload = RecoveryPacketV1(kind = kind, facts = facts, cta = cta).toJson()
    )

fun factSolvingStepPacketV1(packet: SolvingStepPacketV1): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.SOLVING_STEP_PACKET_V1,
        payload = packet.toJson()
    )

fun factTeachingCardV1(card: TeachingCardV1): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.TEACHING_CARD_V1,
        payload = card.toJson()
    )

fun factSetupReplyPacketV1(packet: SetupReplyPacketV1): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.SETUP_REPLY_PACKET_V1,
        payload = packet.toJson()
    )

fun factConfrontationReplyPacketV1(packet: ConfrontationReplyPacketV1): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.CONFRONTATION_REPLY_PACKET_V1,
        payload = packet.toJson()
    )

fun factResolutionReplyPacketV1(packet: ResolutionReplyPacketV1): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.RESOLUTION_REPLY_PACKET_V1,
        payload = packet.toJson()
    )

// -----------------------------------------------------------------------------
// Phase 2 — Detour packet convenience wrappers
// -----------------------------------------------------------------------------

fun factStepClarificationPacketV1(payload: JSONObject): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.STEP_CLARIFICATION_PACKET_V1,
        payload = payload
    )

fun factProofChallengePacketV1(payload: JSONObject): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.PROOF_CHALLENGE_PACKET_V1,
        payload = payload
    )


fun factUserReasoningCheckPacketV1(payload: JSONObject): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.USER_REASONING_CHECK_PACKET_V1,
        payload = payload
    )

fun factAlternativeTechniquePacketV1(payload: JSONObject): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.ALTERNATIVE_TECHNIQUE_PACKET_V1,
        payload = payload
    )


fun factTargetCellQueryPacketV1(payload: JSONObject): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.TARGET_CELL_QUERY_PACKET_V1,
        payload = payload
    )

fun factCandidateStatePacketV1(payload: JSONObject): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.CANDIDATE_STATE_PACKET_V1,
        payload = payload
    )

fun factNeighborCellQueryPacketV1(payload: JSONObject): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.NEIGHBOR_CELL_QUERY_PACKET_V1,
        payload = payload
    )

fun factReturnToRoutePacketV1(payload: JSONObject): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.RETURN_TO_ROUTE_PACKET_V1,
        payload = payload
    )

fun factSolverCellCandidatesPacketV1(payload: JSONObject): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.SOLVER_CELL_CANDIDATES_PACKET_V1,
        payload = payload
    )

fun factSolverCellsCandidatesPacketV1(payload: JSONObject): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.SOLVER_CELLS_CANDIDATES_PACKET_V1,
        payload = payload
    )

fun factSolverHouseCandidateMapPacketV1(payload: JSONObject): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.SOLVER_HOUSE_CANDIDATE_MAP_PACKET_V1,
        payload = payload
    )

fun factSolverCellDigitBlockersPacketV1(payload: JSONObject): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.SOLVER_CELL_DIGIT_BLOCKERS_PACKET_V1,
        payload = payload
    )

fun factSolverReasoningCheckPacketV1(payload: JSONObject): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.SOLVER_REASONING_CHECK_PACKET_V1,
        payload = payload
    )

fun factSolverAlternativeTechniquePacketV1(payload: JSONObject): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.SOLVER_ALTERNATIVE_TECHNIQUE_PACKET_V1,
        payload = payload
    )

fun factSolverTechniqueScopeCheckPacketV1(payload: JSONObject): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.SOLVER_TECHNIQUE_SCOPE_CHECK_PACKET_V1,
        payload = payload
    )

fun factSolverLocalMoveSearchPacketV1(payload: JSONObject): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.SOLVER_LOCAL_MOVE_SEARCH_PACKET_V1,
        payload = payload
    )

fun factSolverRouteComparisonPacketV1(payload: JSONObject): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.SOLVER_ROUTE_COMPARISON_PACKET_V1,
        payload = payload
    )

fun factSolverScopedSupportPacketV1(payload: JSONObject): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.SOLVER_SCOPED_SUPPORT_PACKET_V1,
        payload = payload
    )

fun factNormalizedDetourMoveProofV1(payload: JSONObject): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.NORMALIZED_DETOUR_MOVE_PROOF_V1,
        payload = payload
    )

fun factNormalizedDetourLocalInspectionV1(payload: JSONObject): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.NORMALIZED_DETOUR_LOCAL_INSPECTION_V1,
        payload = payload
    )

fun factNormalizedDetourProposalVerdictV1(payload: JSONObject): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.NORMALIZED_DETOUR_PROPOSAL_VERDICT_V1,
        payload = payload
    )

fun factDetourNarrativeContextV1(payload: JSONObject): FactBundleV1 =
    FactBundleV1(
        type = FactBundleV1.Type.DETOUR_NARRATIVE_CONTEXT_V1,
        payload = payload
    )





data class ReplyRequestV1(
    val version: String = "reply_v1",
    val assistantStyleHeader: String? = null,

    // app-owned memory snapshots
    val userTally: UserTallyV1? = null,
    val assistantTally: AssistantTallyV1? = null,
    val relationshipMemory: RelationshipMemoryV1? = null,
    val recentTurns: List<TranscriptTurnV1> = emptyList(),

    // onboarding opening turn
    val openingTurn: Boolean = false,

    // Phase CTA-1: structured CTA contract (policy only; wording still generated later)
    val ctaContract: CtaContractV1? = null,
    val personalizationMini: PersonalizationMiniV1? = null,

    val turn: ReplyTurnCtxV1,
    val decision: ReplyDecisionV1,
    val facts: List<FactBundleV1>,
    val style: ReplyStyleV1
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("version", "reply_v1")
        if (!assistantStyleHeader.isNullOrBlank()) put("assistant_style_header", assistantStyleHeader)
        put("turn", turn.toJson())
        put("decision", decision.toJson())
        put("facts", FactBundleV1.jsonArray(facts))
        put("style", style.toJson())
        if (userTally != null) put("user_tally", userTally.toJson())
        if (assistantTally != null) put("assistant_tally", assistantTally.toJson())
        if (relationshipMemory != null) put("relationship_memory", relationshipMemory.toJson())
        if (recentTurns.isNotEmpty()) put("recent_turns", TranscriptTurnV1.jsonArray(recentTurns))
        put("opening_turn", openingTurn)
        put("cta_contract", ctaContract?.toJson() ?: JSONObject.NULL)
        put("personalization_mini", personalizationMini?.toJson() ?: JSONObject.NULL)
    }

    fun toJsonString(): String = toJson().toString()

    companion object {

        /**
         * ✅ Canonical builder (current shape).
         */
        fun build(
            turnId: Int,
            mode: String,
            phase: String,
            userText: String,
            pendingBefore: String?,
            pendingAfter: String?,
            focusBefore: Int?,
            focusAfter: Int?,
            decisionKind: DecisionKindV1,
            decisionSummary: String,
            mutationApplied: MutationAppliedV1?,
            facts: List<FactBundleV1>,
            style: ReplyStyleV1 = ReplyStyleV1(),
            userTally: UserTallyV1? = null,
            assistantTally: AssistantTallyV1? = null,
            relationshipMemory: RelationshipMemoryV1? = null,
            recentTurns: List<TranscriptTurnV1> = emptyList(),
            openingTurn: Boolean = false,
            assistantStyleHeader: String? = null,
            turnAuthorityOwner: String? = null,
            turnAuthorityReason: String? = null,
            turnPendingStatus: String? = null,
            turnResumedPendingKind: String? = null,
            turnRouteReturnAllowed: Boolean = false,
            turnBoundaryStatus: String? = null,
            turnBoundaryReason: String? = null,
            replyDemandCategory: ReplyDemandCategoryV1? = null,

            // ✅ NEW: explicit story header for SOLVING narration
            story: ReplyStoryCtxV1? = null,

            // ✅ NEW: explicit CTA contract scaffold
            ctaContract: CtaContractV1? = null,
            personalizationMini: PersonalizationMiniV1? = null
        ): ReplyRequestV1 {

            val turn = ReplyTurnCtxV1(
                turnId = turnId,
                tickId = 2,
                mode = mode,
                phase = phase,
                userText = userText,
                pendingBefore = pendingBefore,
                pendingAfter = pendingAfter,
                focusBefore = focusBefore,
                focusAfter = focusAfter,
                turnAuthorityOwner = turnAuthorityOwner,
                turnAuthorityReason = turnAuthorityReason,
                turnPendingStatus = turnPendingStatus,
                turnResumedPendingKind = turnResumedPendingKind,
                turnRouteReturnAllowed = turnRouteReturnAllowed,
                turnBoundaryStatus = turnBoundaryStatus,
                turnBoundaryReason = turnBoundaryReason,
                replyDemandCategory = replyDemandCategory,
                story = story
            )

            val decision = ReplyDecisionV1(
                decisionKind = decisionKind,
                summary = decisionSummary,
                mutationApplied = mutationApplied
            )

            return ReplyRequestV1(
                version = "reply_v1",
                assistantStyleHeader = assistantStyleHeader,
                userTally = userTally,
                assistantTally = assistantTally,
                relationshipMemory = relationshipMemory,
                recentTurns = recentTurns,
                openingTurn = openingTurn,
                ctaContract = ctaContract,
                personalizationMini = personalizationMini,
                turn = turn,
                decision = decision,
                facts = facts,
                style = style
            )
        }

        // ============================================================
        // ✅ PATCH D1: Quick deterministic fix — build request via JSON
        // (works even if some call-sites are temporarily out of sync)
        // ============================================================
        fun buildViaJsonCompat(
            turnId: Int,
            mode: String,
            phase: String,
            userText: String,
            pendingBefore: String?,
            pendingAfter: String?,
            focusBefore: Int?,
            focusAfter: Int?,
            decisionKind: DecisionKindV1,
            decisionSummary: String,
            mutationApplied: MutationAppliedV1?,
            facts: List<FactBundleV1>,
            style: ReplyStyleV1 = ReplyStyleV1(),
            userTally: UserTallyV1? = null,
            assistantTally: AssistantTallyV1? = null,
            relationshipMemory: RelationshipMemoryV1? = null,
            recentTurns: List<TranscriptTurnV1> = emptyList(),
            openingTurn: Boolean = false,
            assistantStyleHeader: String? = null,
            ctaContract: CtaContractV1? = null,
            personalizationMini: PersonalizationMiniV1? = null
        ): ReplyRequestV1 {

            val root = JSONObject()
            root.put("version", "reply_v1")
            if (!assistantStyleHeader.isNullOrBlank()) root.put("assistant_style_header", assistantStyleHeader)

            val turn = JSONObject()
                .put("turn_id", turnId)
                .put("tick_id", 2)
                .put("mode", mode)
                .put("phase", phase)
                .put("user_text", userText)

            turn.put("pending_before", pendingBefore ?: JSONObject.NULL)
            turn.put("pending_after", pendingAfter ?: JSONObject.NULL)
            turn.put("focus_before", focusBefore ?: JSONObject.NULL)
            turn.put("focus_after", focusAfter ?: JSONObject.NULL)
            root.put("turn", turn)

            val decision = JSONObject()
                .put("decision_kind", decisionKind.name)
                .put("summary", decisionSummary)

            decision.put("mutation_applied", mutationApplied?.toJson() ?: JSONObject.NULL)
            root.put("decision", decision)

            root.put("facts", FactBundleV1.jsonArray(facts))
            root.put("style", style.toJson())

            if (userTally != null) root.put("user_tally", userTally.toJson())
            if (assistantTally != null) root.put("assistant_tally", assistantTally.toJson())
            if (relationshipMemory != null) root.put("relationship_memory", relationshipMemory.toJson())
            if (recentTurns.isNotEmpty()) root.put("recent_turns", TranscriptTurnV1.jsonArray(recentTurns))
            root.put("opening_turn", openingTurn)
            root.put("cta_contract", ctaContract?.toJson() ?: JSONObject.NULL)
            root.put("personalization_mini", personalizationMini?.toJson() ?: JSONObject.NULL)

            return parseJsonCompat(root.toString())
        }



        /**
         * Minimal JSON parser for ReplyRequestV1 (compat).
         * Accepts partial JSON and fills defaults.
         */
        fun parseJsonCompat(json: String): ReplyRequestV1 {
            val o = runCatching { JSONObject(json) }.getOrElse { JSONObject() }

            val assistantStyleHeader = o.optString("assistant_style_header", null).takeIf { !it.isNullOrBlank() }

            val turn = ReplyTurnCtxV1.parse(o.optJSONObject("turn"))
            val decision = ReplyDecisionV1.parse(o.optJSONObject("decision"))

            val facts = FactBundleV1.parseArray(o.optJSONArray("facts"))
            val style = ReplyStyleV1.parse(o.optJSONObject("style"))
            val ctaContract = CtaContractV1.parse(o.optJSONObject("cta_contract"))

            val userTally = UserTallyV1.parse(o.optJSONObject("user_tally")).takeIf { it != UserTallyV1() }
            val assistantTally = AssistantTallyV1.parse(o.optJSONObject("assistant_tally")).takeIf { it != AssistantTallyV1() }

            val recentTurnsArr = o.optJSONArray("recent_turns")
            val recentTurns = mutableListOf<TranscriptTurnV1>()
            if (recentTurnsArr != null) {
                for (i in 0 until recentTurnsArr.length()) {
                    val t = recentTurnsArr.optJSONObject(i) ?: continue
                    recentTurns.add(
                        TranscriptTurnV1(
                            turnId = t.optLong("turn_id", -1L),
                            user = t.optString("user", ""),
                            assistant = t.optString("assistant", "")
                        )
                    )
                }
            }

            val openingTurn = o.optBoolean("opening_turn", false)

            return ReplyRequestV1(
                version = "reply_v1",
                assistantStyleHeader = assistantStyleHeader,
                userTally = userTally,
                assistantTally = assistantTally,
                recentTurns = recentTurns,
                openingTurn = openingTurn,
                ctaContract = ctaContract,
                turn = turn,
                decision = decision,
                facts = facts,
                style = style
            )
        }
    }
}

// -----------------------------------------------------------------------------
// App-only DecisionOutcome (internal, deterministic)
// -----------------------------------------------------------------------------

data class StateMutationsV1(
    val newMode: com.contextionary.sudoku.conductor.SudoMode? = null,
    val newPhase: com.contextionary.sudoku.conductor.GridPhase? = null,
    val newPending: com.contextionary.sudoku.conductor.Pending? = null,
    val newFocusIdx: Int? = null
)

data class DecisionOutcomeV1(
    val decisionKind: DecisionKindV1,
    val summary: String,
    val mutationApplied: MutationAppliedV1? = null,

    val stateMutations: StateMutationsV1 = StateMutationsV1(),
    val factBundles: List<FactBundleV1> = emptyList(),

    val replyStyle: ReplyStyleV1 = ReplyStyleV1(),

    val needsClarification: Boolean = false,
    val clarificationQuestionHint: String? = null
)

