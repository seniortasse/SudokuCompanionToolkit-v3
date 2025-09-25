package com.contextionary.sudoku

import org.opencv.android.OpenCVLoader

object OpenCv {
    @Volatile private var ready = false

    fun ensureLoaded() {
        if (ready) return
        if (!OpenCVLoader.initDebug()) {
            throw IllegalStateException("OpenCV init failed")
        }
        ready = true
    }
}