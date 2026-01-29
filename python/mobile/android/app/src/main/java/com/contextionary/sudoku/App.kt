package com.contextionary.sudoku

import android.app.Application
import android.util.Log
//import org.opencv.core.Core
//import org.opencv.android.OpenCVLoader
import org.opencv.core.Core

class App : Application() {
    override fun onCreate() {
        super.onCreate()

        // 🔹 Run the debug smoke test at startup.
        // It's safe to call in release too, but keep it while you're verifying.
        OpenCVSmoke.run(this)

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