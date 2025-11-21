package com.contextionary.sudoku

object SudoSsml {

    // Choose a warm, conversational neural voice per BCP-47 locale.
    fun pickAzureVoice(localeTag: String): String = when (localeTag.lowercase()) {
        "en-us" -> "en-US-JennyNeural"
        "en-gb" -> "en-GB-LibbyNeural"
        "es-es" -> "es-ES-ElviraNeural"
        "fr-fr" -> "fr-FR-DeniseNeural"
        "de-de" -> "de-DE-KatjaNeural"
        else    -> "en-US-JennyNeural"
    }

    // Optional: map our Mood/Severity to Azure "style"
    private fun styleFor(mood: SudoVoiceStyler.Mood, severity: SudoVoiceStyler.Severity): String {
        return when {
            severity == SudoVoiceStyler.Severity.SEVERE -> "empathetic"
            severity == SudoVoiceStyler.Severity.MILD   -> "calm"
            mood == SudoVoiceStyler.Mood.CHEER          -> "cheerful"
            mood == SudoVoiceStyler.Mood.WARM           -> "friendly"
            else                                        -> "chat"  // safe conversational baseline
        }
    }

    // Convert a styled line into Azure SSML
    fun toAzureSsml(
        styled: SudoVoiceStyler.Styled,
        localeTag: String,
        explicitVoice: String? = null,
        mood: SudoVoiceStyler.Mood = SudoVoiceStyler.Mood.NEUTRAL,
        severity: SudoVoiceStyler.Severity = SudoVoiceStyler.Severity.OK
    ): String {
        val voice = explicitVoice ?: pickAzureVoice(localeTag)
        val style = styleFor(mood, severity)
        val ratePct  = styled.speechRate?.let { (it * 100).toInt().coerceIn(70, 130) } ?: 100
        // pitch in semitones: 1.0 ≈ +0 st; we’ll keep subtle
        val pitch = styled.pitch?.let { if (it > 1.01f) "+1st" else if (it < 0.99f) "-1st" else "0st" } ?: "0st"

        val safe = styled.text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")

        // NOTE: requires xmlns:mstts for style
        return """
<speak version="1.0" xml:lang="$localeTag" xmlns:mstts="https://www.w3.org/2001/mstts">
  <voice name="$voice">
    <mstts:express-as style="$style">
      <prosody rate="${ratePct}%" pitch="$pitch">
        $safe
      </prosody>
    </mstts:express-as>
  </voice>
</speak>
        """.trimIndent()
    }
}