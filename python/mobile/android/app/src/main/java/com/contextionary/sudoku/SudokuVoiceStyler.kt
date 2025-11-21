package com.contextionary.sudoku

import kotlin.math.min

/**
 * Turns a plain assistant_message into a more human-sounding line for TTS.
 * Works with plain TextToSpeech (no SSML): uses punctuation, ellipses, dashes,
 * and a light sprinkle of interjections + micro-pauses.
 */
object SudoVoiceStyler {
    data class Styled(val text: String, val speechRate: Float?, val pitch: Float?)

    // Small bank of interjections; keep gentle & low-key
    private val okays = listOf("Okay", "Alright", "Got it", "Sounds good")
    private val warms = listOf("Nice", "Great", "Perfect", "Lovely")
    private val softThink = listOf("Hmm", "Mm-hmm")

    fun style(
        raw: String,
        mood: Mood = Mood.NEUTRAL,
        severityHint: Severity = Severity.OK
    ): Styled {
        var s = raw.trim()

        // Normalize spacing and end punctuation
        s = s.replace(Regex("\\s+"), " ")
        if (!s.endsWith(".") && !s.endsWith("!") && !s.endsWith("?")) s += "."

        // Insert micro-pauses after commas / clauses
        s = s.replace(", ", ", … ")
            .replace(" — ", " — … ")

        // Light interjection at start depending on mood
        val prefix = when (mood) {
            Mood.WARM   -> pick(warms) + "… "
            Mood.CALM   -> pick(softThink) + "… "
            Mood.CHEER  -> pick(okays) + "! "
            Mood.NEUTRAL-> ""
        }

        // If message starts too abruptly, prepend the prefix
        if (prefix.isNotEmpty() && s.length > 24) s = prefix + s.replaceFirst("… ", "")

        // If asking to double-check: slow down a bit and lower pitch
        val rate: Float?
        val pitch: Float?
        val v = s.lowercase()

        when {
            "double-check" in v || "confirm" in v || "look at" in v -> {
                rate = 0.95f; pitch = 0.98f
            }
            "ready to play" in v || "looks great" in v -> {
                rate = 1.05f; pitch = 1.03f
            }
            else -> {
                rate = null; pitch = null
            }
        }

        // Severity hint: milder tone if severe, a touch brighter if ok
        val adj = when (severityHint) {
            Severity.SEVERE -> Pair(0.93f, 0.97f)
            Severity.MILD   -> Pair(0.98f, 1.00f)
            Severity.OK     -> Pair(1.00f, 1.02f)
        }
        val finalRate = rate?.let { it * adj.first }
        val finalPitch = pitch?.let { it * adj.second }

        // Limit ellipses density to avoid overdoing pauses
        val capped = capEllipses(s, maxCount = 3)

        return Styled(capped, finalRate, finalPitch)
    }

    private fun capEllipses(s: String, maxCount: Int): String {
        var count = 0
        val out = StringBuilder(s.length)
        var i = 0
        while (i < s.length) {
            if (i+2 < s.length && s[i]=='.' && s[i+1]=='.' && s[i+2]=='.') {
                if (count < maxCount) { out.append("…"); count++ }
                i += 3
            } else {
                out.append(s[i]); i++
            }
        }
        return out.toString()
    }

    private fun pick(lst: List<String>) = lst[(Math.random() * lst.size).toInt()]

    enum class Mood { NEUTRAL, WARM, CALM, CHEER }
    enum class Severity { OK, MILD, SEVERE }
}