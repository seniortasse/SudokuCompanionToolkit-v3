package com.contextionary.sudoku

import android.app.Application
import android.util.Log
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.opencv.core.Core

class App : Application() {
    override fun onCreate() {
        super.onCreate()

        // ✅ Chaquopy runtime init (required before any Python.getInstance() calls)
        try {
            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(this))
                Log.i("Chaquopy", "Python started OK")
            }
        } catch (t: Throwable) {
            Log.e("Chaquopy", "Python start failed", t)
        }

        // 🔹 Run the debug smoke test at startup.
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
        }
    }
}