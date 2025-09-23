package com.contextionary.sudoku

import android.app.Application
import android.util.Log
import org.opencv.core.Core

class App : Application() {
    override fun onCreate() {
        super.onCreate()

        // Load native lib first (if it also depends on OpenCV, either order works in practice)
        try {
            System.loadLibrary("native-lib")
        } catch (t: Throwable) {
            Log.e("App", "Failed to load native-lib", t)
        }

        // Initialize OpenCV once at process start
        try {
            OpenCv.ensureLoaded()
            Log.i("OpenCV", "Loaded OpenCV ${Core.getVersionString()}")
        } catch (t: Throwable) {
            Log.e("OpenCV", "OpenCV init failed", t)
            // Consider showing a user-friendly message or disabling features that require OpenCV
        }
    }
}