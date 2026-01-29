package com.contextionary.sudoku

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Size

object OpenCVSmoke {
    private const val TAG = "SudokuSmoke"

    /**
     * - Ensures OpenCV is loaded
     * - Logs the OpenCV version
     * - Writes a small PNG to the app cache folder and logs its path
     */
    fun run(context: Context) {
        val ok = OpenCVLoader.initDebug()
        Log.i(TAG, "OpenCV initDebug() = $ok")

        if (!ok) {
            Log.e(TAG, "OpenCV failed to initialize. Native libs likely missing.")
            return
        }

        Log.i(TAG, "OpenCV Core.VERSION = ${Core.VERSION}")

        // Make a tiny 64x64 grayscale Mat and convert to Bitmap, then save to cache as PNG.
        val mat = Mat.zeros(64, 64, CvType.CV_8UC1)
        // Put a bright pixel to ensure non-empty content
        // mat.put(32, 32, 255.toByte())

        mat.put(32, 32, byteArrayOf(255.toByte()))
        // or
        mat.put(32, 32, 255.0)

        val bmp = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, bmp)

        val outFile = context.cacheDir.resolve("opencv_smoke.png")
        outFile.outputStream().use { fos ->
            bmp.compress(Bitmap.CompressFormat.PNG, 100, fos)
        }

        Log.i(TAG, "Wrote smoke PNG to: ${outFile.absolutePath}")
        mat.release()
    }
}