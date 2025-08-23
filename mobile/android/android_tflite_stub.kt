// android_tflite_stub.kt
// Tiny example using org.tensorflow:tensorflow-lite:2.14.0
// implementation 'org.tensorflow:tensorflow-lite:2.14.0'

package com.example.sudoku

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class SudokuCellModelTFLite(context: Context, modelAsset: String = "sudoku_cell_model_int8.tflite") {
    private val interpreter: Interpreter

    init {
        val opts = Interpreter.Options().apply { numThreads = 2 }
        val model = context.assets.open(modelAsset).readBytes()
        val bb = ByteBuffer.allocateDirect(model.size).order(ByteOrder.nativeOrder())
        bb.put(model); bb.rewind()
        interpreter = Interpreter(bb, opts)
    }

    fun predict(cell: Bitmap): Triple<FloatArray, FloatArray, FloatArray> {
        val input = preprocess(cell) // [1,64,64,1] uint8
        val outType = Array(1) { FloatArray(4) }
        val outDigit = Array(1) { FloatArray(10) }
        val outNotes = Array(1) { FloatArray(9) }
        val outputs = mapOf(0 to outType, 1 to outDigit, 2 to outNotes)
        interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
        return Triple(outType[0], outDigit[0], outNotes[0])
    }

    private fun preprocess(bm: Bitmap): ByteBuffer {
        val w = 64; val h = 64
        val scaled = Bitmap.createScaledBitmap(bm, w, h, true)
        val buf = ByteBuffer.allocateDirect(w*h).order(ByteOrder.nativeOrder())
        val pixels = IntArray(w*h)
        scaled.getPixels(pixels, 0, w, 0, 0, w, h)
        for (p in pixels) {
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            val gray = ((r*299 + g*587 + b*114) / 1000).toInt() // 0..255
            buf.put(gray.toByte())
        }
        buf.rewind()
        return buf
    }
}
