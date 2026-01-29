package com.contextionary.sudoku

import android.content.Context
import android.content.SharedPreferences
import java.util.Locale

object UserPrefs {
    private const val FILE = "sudo_prefs"
    private const val KEY_ASR_LOCALE = "asr_locale"
    private const val KEY_AZURE_TTS_ENABLED = "azure_tts_enabled"
    private const val KEY_AZURE_VOICE = "azure_voice"

    private fun prefs(ctx: Context): SharedPreferences =
        ctx.getSharedPreferences(FILE, Context.MODE_PRIVATE)

    fun isAzureTtsEnabled(ctx: Context): Boolean =
        prefs(ctx).getBoolean(KEY_AZURE_TTS_ENABLED, true)

    fun setAzureTtsEnabled(ctx: Context, enabled: Boolean) {
        prefs(ctx).edit().putBoolean(KEY_AZURE_TTS_ENABLED, enabled).apply()
    }

    fun preferredAsrLocale(ctx: Context): Locale? =
        prefs(ctx).getString(KEY_ASR_LOCALE, null)?.let { Locale.forLanguageTag(it) }

    fun setPreferredAsrLocale(ctx: Context, locale: Locale?) {
        prefs(ctx).edit().putString(KEY_ASR_LOCALE, locale?.toLanguageTag()).apply()
    }

    fun azureVoiceName(ctx: Context, default: String = "en-US-JaneNeural"): String =
        prefs(ctx).getString(KEY_AZURE_VOICE, default) ?: default

    fun setAzureVoiceName(ctx: Context, name: String) {
        prefs(ctx).edit().putString(KEY_AZURE_VOICE, name).apply()
    }
}