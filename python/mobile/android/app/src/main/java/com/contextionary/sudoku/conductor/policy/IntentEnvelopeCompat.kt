package com.contextionary.sudoku.conductor.policy



// ---------- tiny reflection helpers (compile-safe even if fields don’t exist) ----------

private fun Any.getFieldOrNull(name: String): Any? = runCatching {
    val f = this::class.java.declaredFields.firstOrNull { it.name == name } ?: return@runCatching null
    f.isAccessible = true
    f.get(this)
}.getOrNull()

private fun Any.getMethod0OrNull(name: String): Any? = runCatching {
    val m = this::class.java.methods.firstOrNull { it.name == name && it.parameterTypes.isEmpty() } ?: return@runCatching null
    m.isAccessible = true
    m.invoke(this)
}.getOrNull()

private fun Any.boolLike(vararg names: String): Boolean? {
    for (n in names) {
        val v = getFieldOrNull(n) ?: getMethod0OrNull(n)
        when (v) {
            is Boolean -> return v
            is String -> return v.equals("true", ignoreCase = true)
            is Number -> return v.toInt() != 0
        }
    }
    return null
}

private fun Any.intLike(vararg names: String): Int? {
    for (n in names) {
        val v = getFieldOrNull(n) ?: getMethod0OrNull(n)
        when (v) {
            is Int -> return v
            is Number -> return v.toInt()
            is String -> v.toIntOrNull()?.let { return it }
        }
    }
    return null
}

private fun Any.stringLike(vararg names: String): String? {
    for (n in names) {
        val v = getFieldOrNull(n) ?: getMethod0OrNull(n)
        when (v) {
            is String -> return v
        }
    }
    return null
}

private inline fun <reified T : Enum<T>> enumValueOrNullCompat(raw: String?): T? {
    val normalized = raw?.trim()?.takeIf { it.isNotEmpty() } ?: return null
    return enumValues<T>().firstOrNull { it.name.equals(normalized, ignoreCase = true) }
}

// ---------- compat accessors used by planner / store ----------

/**
 * Prefer a real envelope flag if present; otherwise fall back to conservative logic.
 *
 * "Unclear" in v1 is usually: either explicit flag exists OR intents empty OR top intent has missing slots.
 */
fun IntentEnvelopeV1.isUnclearCompat(): Boolean {
    val b = (this as Any).boolLike("isUnclear", "unclear", "needsClarification", "needs_clarification")
    if (b != null) return b

    val intents = runCatching { this.intents }.getOrNull().orEmpty()
    if (intents.isEmpty()) return true

    val top = intents.maxByOrNull { it.confidence }
    return top?.missing?.isNotEmpty() == true
}

/**
 * V1 supports: IntentV1.addressesUserAgendaId (canonical).
 * This accessor stays future-proof if schema changes or aliases appear.
 */
fun IntentV1.addressesUserAgendaIdCompat(): String? {
    // canonical first
    this.addressesUserAgendaId?.let { if (it.isNotBlank()) return it }

    // fallback aliases (reflection safe)
    return (this as Any).stringLike(
        "addressesUserAgendaId",
        "addresses_user_agenda_id",
        "agendaId",
        "agenda_id"
    )?.takeIf { it.isNotBlank() }
}



fun IntentV1.referenceResolutionModeCompat(): ReferenceResolutionModeV1? {
    this.referenceResolutionMode?.let { return it }

    val raw = (this as Any).stringLike(
        "referenceResolutionMode",
        "reference_resolution_mode"
    )
    return enumValueOrNullCompat<ReferenceResolutionModeV1>(raw)?.takeIf { it != ReferenceResolutionModeV1.NONE }
}

fun IntentEnvelopeV1.repairSignalCompat(): RepairSignalV1? {
    this.repairSignal?.let { return it }

    val raw = (this as Any).stringLike(
        "repairSignal",
        "repair_signal"
    )
    return enumValueOrNullCompat<RepairSignalV1>(raw)?.takeIf { it != RepairSignalV1.NONE }
}

fun IntentEnvelopeV1.contextSpanHintCompat(): ContextSpanHintV1? {
    this.contextSpanHint?.let { return it }

    val raw = (this as Any).stringLike(
        "contextSpanHint",
        "context_span_hint"
    )
    return enumValueOrNullCompat<ContextSpanHintV1>(raw)
}

fun IntentEnvelopeV1.referencesPriorTurnsCompat(): Boolean? {
    this.referencesPriorTurns?.let { return it }

    return (this as Any).boolLike(
        "referencesPriorTurns",
        "references_prior_turns"
    )
}

/**
 * Extract 0..80 cell index from the intent target if possible.
 *
 * Preferred path: targets[].cell like "r4c2".
 * Fallback path: older numeric fields (reflection).
 */
fun IntentV1.cellIndexOrNullCompat(): Int? {
    // ✅ Preferred: parse from targets[].cell like "r4c2"
    val cell = this.targets.firstOrNull { !it.cell.isNullOrBlank() }?.cell
    val idxFromCell = cell?.let { c ->
        val m = Regex("""r([1-9])c([1-9])""", RegexOption.IGNORE_CASE).find(c.trim()) ?: return@let null
        val r = m.groupValues[1].toIntOrNull() ?: return@let null
        val col = m.groupValues[2].toIntOrNull() ?: return@let null
        ((r - 1) * 9 + (col - 1)).takeIf { it in 0..80 }
    }
    if (idxFromCell != null) return idxFromCell

    // Fallback: older reflection-based fields
    val idx = (this as Any).intLike("cellIndex", "cell_index", "targetCellIndex", "target_cell_index", "idx")
    return idx?.takeIf { it in 0..80 }
}

/**
 * Extract digit (0..9) from intent payload if present.
 * - For normal edits, payload.digit is 1..9
 * - Some flows may use 0 to represent clear.
 */
fun IntentV1.digitOrNullCompat(): Int? {
    // ✅ Preferred: payload.digit
    val d1 = this.payload.digit
    if (d1 != null && d1 in 0..9) return d1

    // Fallback: older reflection-based fields
    val d = (this as Any).intLike("digit", "value", "targetDigit", "target_digit")
    return d?.takeIf { it in 0..9 }
}

/**
 * Extract region from targets[].region if present.
 * Fallback supports older reflection shapes.
 */
fun IntentV1.regionOrNullCompat(): RegionRefV1? {
    // ✅ Preferred: target.region
    val r = this.targets.firstOrNull { it.region != null }?.region
    if (r != null) return r

    // Fallback: legacy-ish reflection
    val kind = (this as Any).stringLike("regionKind", "region_kind", "kind")?.uppercase()
    val idx = (this as Any).intLike("regionIndex", "region_index", "index")
    if (kind != null && idx != null) {
        val k = runCatching { RegionKindV1.valueOf(kind) }.getOrNull() ?: return null
        return RegionRefV1(kind = k, index = idx)
    }
    return null
}

/**
 * Ensure we store raw user text on the envelope (when present) while staying safe
 * if IntentEnvelopeV1 stops being a data class in the future.
 */
fun IntentEnvelopeV1.normalizeCompat(rawUserTextIn: String): IntentEnvelopeV1 {
    val txt = rawUserTextIn.trim()
    if (txt.isBlank()) return this

    return runCatching {
        // data class => copy exists
        this.copy(rawUserText = txt)
    }.getOrElse {
        // if schema changes away from data class, just return as-is.
        this
    }
}