package com.contextionary.sudoku

import android.content.Context
import android.graphics.*
import android.util.Log
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.support.common.FileUtil
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.sqrt

class DigitClassifier(
    private val context: Context,
    private val modelAsset: String = "models/digit_cnn_fp32.tflite",
    private val numThreads: Int = 2,
    private val useXNNPack: Boolean = true,
) {
    private val TAG = "DigitClassifier"

    // ---- Runtime options / parity controls ---------------------------------

    /** Apply the same normalization as Python: (x - 0.5) / 0.5 → [-1, 1]. */
    private val APPLY_NORMALIZATION = true

    /** Apply calibrated softmax with temperature from calibration.json (T ≈ 4.363). */
    private val APPLY_CALIBRATED_SOFTMAX = true
    private val CALIB_T = 4.363f // 1 / 0.22920914

    /** Enable a one-shot parity dump. */
    var dumpParity: Boolean = false
    /** Example: setOf("r1c5","r4c5","r4c7","r5c6","r7c6","r8c4"). Empty set = dump all 81. */
    var dumpWhitelist: Set<String> = emptySet()
    /** Tag to distinguish multiple dumps in the same run. */
    var dumpSessionTag: String = "session"

    /** Center-fraction inner crop applied BEFORE resize (matches Python --inner-crop). */
    var innerCrop: Float = 1.0f

    // ------------------------------------------------------------------------

    private val interpreter: Interpreter
    private val inputShape: IntArray
    private val inputH: Int
    private val inputW: Int
    private val inputC: Int
    private val isCHW: Boolean

    // Co-locate parity with your regular debug folder
    private val dumpRoot: File by lazy {
        File(context.filesDir, "rectify_debug").apply { mkdirs() }
    }

    init {
        val buffer = FileUtil.loadMappedFile(context, modelAsset)
        val opts = Interpreter.Options().apply {
            setNumThreads(numThreads)
            if (useXNNPack) setUseXNNPACK(true)
        }
        interpreter = Interpreter(buffer, opts)

        val inTensor: Tensor = interpreter.getInputTensor(0)
        inputShape = inTensor.shape() // e.g. [1,1,28,28] or [1,28,28,1]
        val nDims = inputShape.size
        val (h, w, c, chw) = when (nDims) {
            4 -> {
                if (inputShape[1] == 1) {
                    // CHW: [1,1,H,W]
                    Quadruple(inputShape[2], inputShape[3], 1, true)
                } else {
                    // HWC: [1,H,W,1]
                    Quadruple(inputShape[1], inputShape[2], inputShape[3], false)
                }
            }
            else -> Quadruple(28, 28, 1, false)
        }
        inputH = h; inputW = w; inputC = c; isCHW = chw

        Log.i(TAG, "Loaded TFLite model: $modelAsset")
        Log.i(TAG, "Input shape=${inputShape.contentToString()} type=${inTensor.dataType()}")
        val outTensor = interpreter.getOutputTensor(0)
        Log.i(TAG, "Output shape=${outTensor.shape().contentToString()} type=${outTensor.dataType()}")
        Log.i(TAG, "Preproc: normalize=$APPLY_NORMALIZATION, calibrated_softmax=$APPLY_CALIBRATED_SOFTMAX (T=$CALIB_T)")
    }

    /**
     * Classify a grid of cell bitmaps. Each tile should be the **raw rectified cell crop**
     * (pre-resize), preferably ~80–120 px square. We handle resize to model input (28×28).
     *
     * Returns Pair(digits[9][9], probs[9][9]) where probs are **calibrated probabilities**
     * if APPLY_CALIBRATED_SOFTMAX=true; otherwise plain softmax.
     */
    fun classifyTiles(
        tiles: Array<Array<Bitmap>>,
        // kept for API parity; rectifier already performs calibrated inner crop.
        innerCrop: Float = 0.92f
    ): Pair<Array<IntArray>, Array<FloatArray>> {

        val outDigits = Array(9) { IntArray(9) }
        val outProbs  = Array(9) { FloatArray(9) }

        val parityDir: File? = if (dumpParity) {
            val stamp = android.text.format.DateFormat.format("yyyyMMdd_HHmmss", java.util.Date())
            File(dumpRoot, "parity_${dumpSessionTag}_$stamp").apply {
                mkdirs()
                Log.i(TAG, "Parity dump dir: $absolutePath")
            }
        } else null

        for (r in 0 until 9) {
            for (c in 0 until 9) {
                val tileRaw = tiles[r][c]
                val tag = "r${r + 1}c${c + 1}"

                // Dump if parity is enabled and either whitelist is empty (dump all) or contains the tag.
                val doDump = dumpParity && (dumpWhitelist.isEmpty() || dumpWhitelist.contains(tag))

                val pre = preprocess(tileRaw, doDump, parityDir, tag)

                val logits2d = Array(1) { FloatArray(10) }
                if (isCHW) {
                    interpreter.run(pre.postNormCHW, logits2d)
                } else {
                    interpreter.run(pre.postNormNHWC, logits2d)
                }
                val logits = logits2d[0]
                val probs = if (APPLY_CALIBRATED_SOFTMAX) {
                    softmaxCalibrated(logits, CALIB_T)
                } else {
                    softmax(logits)
                }

                val bestIdx = argmax(probs)
                outDigits[r][c] = bestIdx
                outProbs[r][c]  = probs[bestIdx]

                if (doDump && parityDir != null) {
                    writeText(File(parityDir, "${tag}_logits.txt"), logits.joinToString(","))
                    writeText(File(parityDir, "${tag}_probsCal.txt"), probs.joinToString(","))
                }

                Log.d(TAG, "tile[$r,$c] -> pred=$bestIdx p=${"%.3f".format(outProbs[r][c])}")
            }
        }

        // Histogram for sanity
        val hist = IntArray(10)
        for (r in 0 until 9) for (c in 0 until 9) hist[outDigits[r][c]]++
        Log.i(TAG, "Class histogram (0..9) after batch: ${hist.joinToString(",")}  total=${hist.sum()}")

        return Pair(outDigits, outProbs)
    }

    /**
     * Preprocess a single tile:
     *  - grayscale
     *  - **center-fraction crop (innerCrop)**
     *  - resize to model (typically 28×28)
     *  - convert to float [0,1]
     *  - normalize to [-1,1] if enabled
     *  Produces both NHWC and CHW float buffers for convenience.
     */
    private fun preprocess(
        srcRaw: Bitmap,
        doDump: Boolean,
        parityDir: File?,
        tag: String
    ): PreprocOut {
        // 1) Grayscale
        val gray = toGray(srcRaw)

        // 1.5) CenterFrac crop BEFORE resize (matches Python --inner-crop)
        val cropFrac = innerCrop.coerceIn(0.6f, 1.0f)
        val gw = gray.width
        val gh = gray.height
        val cw = (gw * cropFrac).toInt().coerceAtLeast(1)
        val ch = (gh * cropFrac).toInt().coerceAtLeast(1)
        val cx = (gw - cw) / 2
        val cy = (gh - ch) / 2
        val grayCropped = Bitmap.createBitmap(gray, cx, cy, cw, ch)

        // 2) Resize to model input size (commonly 28×28)
        val resized = Bitmap.createScaledBitmap(grayCropped, inputW, inputH, true)

        // 3) To float [0,1] and stats
        val n = inputW * inputH
        val buf = FloatArray(n)
        val pixels = IntArray(n)
        resized.getPixels(pixels, 0, inputW, 0, 0, inputW, inputH)

        var sum = 0f
        var sum2 = 0f
        for (i in 0 until n) {
            val p = pixels[i]
            val v = ((p ushr 16) and 0xFF) / 255f // r=g=b in gray
            buf[i] = v
            sum += v
            sum2 += v * v
        }
        val mean = sum / n
        val std  = sqrt(max(1e-12f, sum2 / n - mean * mean))
        Log.d(TAG, "preproc mean=${"%.3f".format(mean)} std=${"%.3f".format(std)} crop=${"%.2f".format(cropFrac)}")

        // 4) Normalize to [-1,1] if enabled
        val post = if (APPLY_NORMALIZATION) {
            FloatArray(n) { i -> (buf[i] - 0.5f) / 0.5f }
        } else {
            buf
        }

        // 5) Build ByteBuffers in both layouts for convenience
        val nhwc = toByteBufferNHWC(post, inputH, inputW) // [1,H,W,1]
        val chw  = toByteBufferCHW(post, inputH, inputW)  // [1,1,H,W]

        // 6) Optional parity dump
        if (doDump && parityDir != null) {
            saveBitmapPng(File(parityDir, "${tag}_tile_raw.png"), srcRaw)
            saveBitmapPng(File(parityDir, "${tag}_tile_cropped.png"), grayCropped)
            saveBitmapPng(File(parityDir, "${tag}_tile_gray${inputW}.png"), resized)
            // raw float arrays (little-endian)
            dumpFloatBin(File(parityDir, "${tag}_input_preNorm.bin"), buf)
            dumpFloatBin(File(parityDir, "${tag}_input_postNorm.bin"), post)
            // metadata
            val meta = JSONObject(
                mapOf(
                    "input_shape_model" to inputShape.toList(),
                    "layout_used" to if (isCHW) "CHW" else "NHWC",
                    "applyNormalization" to APPLY_NORMALIZATION,
                    "mean_preNorm" to mean,
                    "std_preNorm" to std,
                    "calibratedSoftmax" to APPLY_CALIBRATED_SOFTMAX,
                    "temperature" to CALIB_T,
                    "innerCropApplied" to cropFrac
                )
            )
            writeText(File(parityDir, "${tag}_meta.json"), meta.toString())
        }

        return PreprocOut(
            postNormNHWC = nhwc,
            postNormCHW = chw
        )
    }

    // ---- Utilities ----------------------------------------------------------

    private data class PreprocOut(
        val postNormNHWC: ByteBuffer,
        val postNormCHW: ByteBuffer
    )

    private fun toByteBufferNHWC(src: FloatArray, h: Int, w: Int): ByteBuffer {
        val bb = ByteBuffer.allocateDirect(4 * h * w).order(ByteOrder.nativeOrder())
        // NHWC with C=1 → row-major
        for (y in 0 until h) for (x in 0 until w) {
            bb.putFloat(src[y * w + x])
        }
        bb.rewind()
        return bb
    }

    private fun toByteBufferCHW(src: FloatArray, h: Int, w: Int): ByteBuffer {
        val bb = ByteBuffer.allocateDirect(4 * h * w).order(ByteOrder.nativeOrder())
        // CHW with C=1 → same writing order; interpreter reads as [1,1,H,W]
        for (y in 0 until h) for (x in 0 until w) {
            bb.putFloat(src[y * w + x])
        }
        bb.rewind()
        return bb
    }

    private fun argmax(a: FloatArray): Int {
        var bi = 0
        var bv = a[0]
        for (i in 1 until a.size) {
            if (a[i] > bv) { bv = a[i]; bi = i }
        }
        return bi
    }

    private fun softmax(z: FloatArray): FloatArray {
        val m = z.maxOrNull() ?: 0f
        var s = 0.0
        val e = DoubleArray(z.size) { j -> kotlin.math.exp((z[j] - m).toDouble()) }
        for (v in e) s += v
        return FloatArray(z.size) { j -> (e[j] / s).toFloat() }
    }

    private fun softmaxCalibrated(z: FloatArray, T: Float): FloatArray {
        // Probabilities after dividing logits by temperature T
        val m = (z.maxOrNull() ?: 0f) / T
        var s = 0.0
        val e = DoubleArray(z.size) { j -> kotlin.math.exp(((z[j] / T) - m).toDouble()) }
        for (v in e) s += v
        return FloatArray(z.size) { j -> (e[j] / s).toFloat() }
    }

    private fun toGray(src: Bitmap): Bitmap {
        val out = Bitmap.createBitmap(src.width, src.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        val paint = Paint()
        val cm = ColorMatrix()
        cm.setSaturation(0f)
        paint.colorFilter = ColorMatrixColorFilter(cm)
        canvas.drawBitmap(src, 0f, 0f, paint)
        return out
    }

    private fun saveBitmapPng(dst: File, bmp: Bitmap) {
        dst.parentFile?.mkdirs()
        FileOutputStream(dst).use { fos -> bmp.compress(Bitmap.CompressFormat.PNG, 100, fos) }
    }

    private fun dumpFloatBin(dst: File, data: FloatArray) {
        dst.parentFile?.mkdirs()
        val bb = ByteBuffer.allocate(data.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        data.forEach { bb.putFloat(it) }
        FileOutputStream(dst).use { it.write(bb.array()) }
    }

    private fun writeText(dst: File, s: String) {
        dst.parentFile?.mkdirs()
        dst.writeText(s)
    }

    // ---- (Optional) board-slicer helper retained for completeness -----------

    fun classifyBoard(
        boardBitmap: Bitmap,
        perTileInsetFrac: Float = 0.0f
    ): Pair<Array<IntArray>, Array<FloatArray>> {
        val W = boardBitmap.width
        val H = boardBitmap.height
        val cw = W / 9
        val ch = H / 9

        val tiles = Array(9) { Array(9) { boardBitmap } }
        for (r in 0 until 9) {
            val y0 = r * ch
            val y1 = if (r == 8) H else (r + 1) * ch
            for (c in 0 until 9) {
                val x0 = c * cw
                val x1 = if (c == 8) W else (c + 1) * cw

                var ix0 = x0
                var iy0 = y0
                var ix1 = x1
                var iy1 = y1
                if (perTileInsetFrac > 0f) {
                    val padX = ((x1 - x0) * perTileInsetFrac * 0.5f).toInt()
                    val padY = ((y1 - y0) * perTileInsetFrac * 0.5f).toInt()
                    ix0 = (x0 + padX).coerceAtMost(x1 - 1)
                    iy0 = (y0 + padY).coerceAtMost(y1 - 1)
                    ix1 = (x1 - padX).coerceAtLeast(ix0 + 1)
                    iy1 = (y1 - padY).coerceAtLeast(iy0 + 1)
                }

                val w = (ix1 - ix0).coerceAtLeast(1)
                val h = (iy1 - iy0).coerceAtLeast(1)
                tiles[r][c] = Bitmap.createBitmap(boardBitmap, ix0, iy0, w, h)
            }
        }

        return classifyTiles(tiles, innerCrop = 0.92f /* kept for parity */)
    }

    // Small helper for destructuring 4-tuple in init
    private data class Quadruple<A,B,C,D>(val first:A,val second:B,val third:C,val fourth:D)
}