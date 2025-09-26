package com.contextionary.sudoku

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core

@RunWith(AndroidJUnit4::class)
class OpenCVInitInstrumentedTest {
    @Test
    fun openCvLoadsAndReportsVersion() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val ok = OpenCVLoader.initDebug()
        Log.i("SudokuSmokeTest", "OpenCV initDebug() = $ok")
        assertTrue("OpenCV failed to load on device/emulator", ok)

        Log.i("SudokuSmokeTest", "OpenCV VERSION=${Core.VERSION}")
        // If we get here, native libs + Java API are intact.
    }
}