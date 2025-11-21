package com.contextionary.sudoku

interface SudoTtsEngine {
    fun speakSsml(
        ssml: String,
        localeTag: String,
        onStart: (() -> Unit)? = null,
        onDone:  (() -> Unit)? = null,
        onError: ((Throwable) -> Unit)? = null
    )

    fun stop()
    fun shutdown()
}