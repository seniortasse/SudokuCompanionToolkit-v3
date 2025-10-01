package com.contextionary.sudoku

import android.content.Context
import android.graphics.*
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Locale
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

class CornerRefiner(
    ctx: Context,
    private var modelAsset: String = "models/corner_heatmaps3_fp32.tflite",
    numThreads: Int = 2
) {
    // ===== Debug / Logging config =====
    private var ENABLE_VERBOSE = true
    private var ENABLE_TIMING = true
    private var ENABLE_SANITY = true
    private var ENABLE_FILE_DUMPS = false
    private var DUMP_EVERY_N_FRAMES = 15
    private var MAX_DUMPS = 10

    fun setDebugConfig(
        verbose: Boolean? = null,
        timing: Boolean? = null,
        sanity: Boolean? = null,
        fileDumps: Boolean? = null,
        dumpEveryN: Int? = null,
        maxDumps: Int? = null
    ) {
        verbose?.let { ENABLE_VERBOSE = it }
        timing?.let { ENABLE_TIMING = it }
        sanity?.let { ENABLE_SANITY = it }
        fileDumps?.let { ENABLE_FILE_DUMPS = it }
        dumpEveryN?.let { DUMP_EVERY_N_FRAMES = max(1, it) }
        maxDumps?.let { MAX_DUMPS = max(1, it) }
    }

    // ===== Model / tensor config =====
    data class Pt(val x: Float, val y: Float)
    data class Corners(val tl: Pt, val tr: Pt, val br: Pt, val bl: Pt)

    private data class PeakStats(
        val sx: Float,
        val sy: Float,
        val argmaxX: Int,
        val argmaxY: Int,
        val peak: Float
    )

    private data class LetterboxGeom(
        val padX: Int, val padY: Int,
        val fitW: Int, val fitH: Int,
        val roiLeft: Int, val roiTop: Int,
        val roiW: Int, val roiH: Int
    )

    private val TAG = "CornerRefiner"
    private val H = 128
    private val W = 128
    private val C = 3   // gray, x, y
    private val K = 4   // 4 corners

    // Precomputed coord planes (0..1)
    private val xs = FloatArray(H * W)
    private val ys = FloatArray(H * W)

    // Tensors (NHWC IO)
    private val inputNHWC = Array(1) { Array(H) { Array(W) { FloatArray(C) } } }
    private val logits = Array(1) { Array(H) { Array(W) { FloatArray(K) } } }

    private val it: Interpreter
    private val cacheDir: File = File(ctx.cacheDir, "corner_refiner").apply { mkdirs() }
    private var frameCount: Long = 0
    private var dumpsWritten: Int = 0

    // Expose peaks for UI labels
    @Volatile private var lastPeaks: FloatArray? = null
    fun getLastPeaks(): FloatArray? = lastPeaks

    // NEW: Expose the expanded ROI actually cropped (bitmap coords)
    @Volatile private var lastExpandedRoi: Rect? = null
    fun getLastExpandedRoi(): Rect? = lastExpandedRoi

    init {
        // Build coord maps
        var idx = 0
        for (y in 0 until H) {
            val fy = if (H > 1) y.toFloat() / (H - 1) else 0f
            for (x in 0 until W) {
                val fx = if (W > 1) x.toFloat() / (W - 1) else 0f
                xs[idx] = fx
                ys[idx] = fy
                idx++
            }
        }

        val opts = Interpreter.Options().apply { setNumThreads(numThreads) }
        val mmap = FileUtil.loadMappedFile(ctx, modelAsset)
        it = Interpreter(mmap, opts)

        // Shape assertions
        val inShape = it.getInputTensor(0).shape()
        val outShape = it.getOutputTensor(0).shape()
        if (ENABLE_VERBOSE) {
            Log.i(TAG, "Loaded corner model: $modelAsset")
            Log.i(TAG, "Input  shape: ${inShape.contentToString()} (expect [1,128,128,3])")
            Log.i(TAG, "Output shape: ${outShape.contentToString()} (expect [1,128,128,4])")
        }
        require(inShape.size == 4 && inShape[1] == H && inShape[2] == W && inShape[3] == C) {
            "Expected input [1,128,128,3], got ${inShape.contentToString()}"
        }
        require(outShape.size == 4 && outShape[1] == H && outShape[2] == W && outShape[3] == K) {
            "Expected output [1,128,128,4], got ${outShape.contentToString()}"
        }
    }

    /** Refine corners inside a detected grid rectangle (bitmap coordinates). */
    fun refine(bmp: Bitmap, roiBmpSpace: RectF): Corners? {
        frameCount++

        val t0 = SystemClock.elapsedRealtimeNanos()

        // Expand ROI by ~5% to match training margin
        val padFrac = 0.05f
        val cx = (roiBmpSpace.left + roiBmpSpace.right) * 0.5f
        val cy = (roiBmpSpace.top + roiBmpSpace.bottom) * 0.5f
        val rw = (roiBmpSpace.right - roiBmpSpace.left)
        val rh = (roiBmpSpace.bottom - roiBmpSpace.top)
        val rwPad = rw * (1f + 2f * padFrac)
        val rhPad = rh * (1f + 2f * padFrac)
        val roiF = RectF(
            cx - rwPad * 0.5f,
            cy - rhPad * 0.5f,
            cx + rwPad * 0.5f,
            cy + rhPad * 0.5f
        )
        val roi = Rect(
            roiF.left.coerceIn(0f, bmp.width - 1f).toInt(),
            roiF.top.coerceIn(0f, bmp.height - 1f).toInt(),
            roiF.right.coerceIn(1f, bmp.width.toFloat()).toInt(),
            roiF.bottom.coerceIn(1f, bmp.height.toFloat()).toInt()
        )

        // <-- store for overlay
        lastExpandedRoi = Rect(roi)

        if (roi.width() < 8 || roi.height() < 8) {
            if (ENABLE_VERBOSE) Log.w(TAG, "ROI too small: $roi")
            return null
        }

        val crop = safeCrop(bmp, roi) ?: run {
            if (ENABLE_VERBOSE) Log.w(TAG, "Failed to crop ROI: $roi")
            return null
        }

        // LETTERBOX to 128x128
        val srcW = crop.width
        val srcH = crop.height
        val scale = if (srcW >= srcH) 128f / srcW else 128f / srcH
        val fitW = (srcW * scale).toInt().coerceAtLeast(1)
        val fitH = (srcH * scale).toInt().coerceAtLeast(1)
        val padX = (128 - fitW) / 2
        val padY = (128 - fitH) / 2

        val scaled = Bitmap.createScaledBitmap(crop, fitW, fitH, true)
        val canvasBmp = Bitmap.createBitmap(W, H, Bitmap.Config.ARGB_8888)
        val draw = Canvas(canvasBmp)
        draw.drawColor(Color.rgb(128, 128, 128))
        draw.drawBitmap(scaled, padX.toFloat(), padY.toFloat(), null)

        if (ENABLE_VERBOSE) {
            Log.d(TAG, "ROI bmp-space: l=${roi.left} t=${roi.top} w=${roi.width()} h=${roi.height()} | fit=${fitW}x${fitH} pad=($padX,$padY) scale=${"%.4f".format(scale)}")
        }

        // Build input NHWC
        val px = IntArray(W * H)
        canvasBmp.getPixels(px, 0, W, 0, 0, W, H)

        var i = 0
        var gMin = 1f; var gMax = 0f; var gSum = 0f
        for (y in 0 until H) {
            for (x in 0 until W) {
                val p = px[i]
                val r = ((p ushr 16) and 0xFF) / 255f
                val g = ((p ushr 8) and 0xFF) / 255f
                val b = (p and 0xFF) / 255f
                val gray = 0.299f * r + 0.587f * g + 0.114f * b
                inputNHWC[0][y][x][0] = gray
                inputNHWC[0][y][x][1] = xs[i]
                inputNHWC[0][y][x][2] = ys[i]
                gMin = min(gMin, gray)
                gMax = max(gMax, gray)
                gSum += gray
                i++
            }
        }
        val t1 = SystemClock.elapsedRealtimeNanos()

        if (ENABLE_VERBOSE) {
            val mean = gSum / (W * H)
            Log.d(TAG, "Input gray stats: min=${"%.3f".format(gMin)} max=${"%.3f".format(gMax)} mean=${"%.3f".format(mean)}")
        }

        // Inference
        val outputs = hashMapOf(0 to logits as Any)
        val t2s = SystemClock.elapsedRealtimeNanos()
        it.runForMultipleInputsOutputs(arrayOf(inputNHWC as Any), outputs)
        val t2e = SystemClock.elapsedRealtimeNanos()

        // Postprocess
        val tau = 0.04f
        fun sigmoid(v: Float) = 1f / (1f + kotlin.math.exp(-v))

        fun softArgmaxXYAndStats(channel: Int): PeakStats {
            var sumW = 0.0
            var sumX = 0.0
            var sumY = 0.0
            var maxProb = -1f
            var maxX = 0
            var maxY = 0
            for (yy in 0 until H) {
                for (xx in 0 until W) {
                    val s = sigmoid(logits[0][yy][xx][channel])
                    if (s > maxProb) { maxProb = s; maxX = xx; maxY = yy }
                    val w = kotlin.math.exp((s / tau).toDouble())
                    sumW += w; sumX += w * xx; sumY += w * yy
                }
            }
            val sx = if (sumW > 0) (sumX / sumW).toFloat() else (W - 1) / 2f
            val sy = if (sumW > 0) (sumY / sumW).toFloat() else (H - 1) / 2f
            return PeakStats(sx, sy, maxX, maxY, maxProb)
        }

        val t3s = SystemClock.elapsedRealtimeNanos()
        val tlQ = softArgmaxXYAndStats(0)
        val trQ = softArgmaxXYAndStats(1)
        val brQ = softArgmaxXYAndStats(2)
        val blQ = softArgmaxXYAndStats(3)
        val t3e = SystemClock.elapsedRealtimeNanos()

        lastPeaks = floatArrayOf(tlQ.peak, trQ.peak, brQ.peak, blQ.peak)

        if (ENABLE_VERBOSE) {
            fun Float.format1(): String = String.format(Locale.US, "%.1f", this)
            fun fmt(q: PeakStats) =
                "peak=${"%.3f".format(q.peak)} argmax=(${q.argmaxX},${q.argmaxY}) soft=(${q.sx.format1()},${q.sy.format1()})"
            Log.d(TAG, "TL: ${fmt(tlQ)} | TR: ${fmt(trQ)} | BR: ${fmt(brQ)} | BL: ${fmt(blQ)}")
        }

        // Map back (undo letterbox)
        val lb = LetterboxGeom(padX, padY, fitW, fitH, roi.left, roi.top, roi.width(), roi.height())
        fun mapToBmp(x128: Float, y128: Float, g: LetterboxGeom): Pt {
            val xC = x128.coerceIn(g.padX.toFloat(), (g.padX + g.fitW - 1).toFloat())
            val yC = y128.coerceIn(g.padY.toFloat(), (g.padY + g.fitH - 1).toFloat())
            val nx = (xC - g.padX) / (g.fitW - 1f).coerceAtLeast(1f)
            val ny = (yC - g.padY) / (g.fitH - 1f).coerceAtLeast(1f)
            val xBmp = g.roiLeft + nx * g.roiW
            val yBmp = g.roiTop + ny * g.roiH
            return Pt(xBmp, yBmp)
        }

        val tl = mapToBmp(tlQ.sx, tlQ.sy, lb)
        val tr = mapToBmp(trQ.sx, trQ.sy, lb)
        val br = mapToBmp(brQ.sx, brQ.sy, lb)
        val bl = mapToBmp(blQ.sx, blQ.sy, lb)

        if (ENABLE_SANITY) {
            fun sane(p: Pt) = p.x.isFinite() && p.y.isFinite() &&
                    p.x in 0f..bmp.width.toFloat() && p.y in 0f..bmp.height.toFloat()

            if (!(sane(tl) && sane(tr) && sane(br) && sane(bl))) {
                Log.w(TAG, "Corner sanity failed; out-of-bounds or NaN.")
                return null
            }
        }

        if (ENABLE_VERBOSE) {
            Log.i(TAG, "TL(${tl.x.format1()},${tl.y.format1()}) TR(${tr.x.format1()},${tr.y.format1()}) BR(${br.x.format1()},${br.y.format1()}) BL(${bl.x.format1()},${bl.y.format1()})")
        }

        if (ENABLE_FILE_DUMPS && dumpsWritten < MAX_DUMPS && (frameCount % DUMP_EVERY_N_FRAMES == 0L)) {
            try {
                val stamp = timestamp()
                dumpBitmapPNG(canvasBmp, "input_${stamp}.png")
                dumpHeatmapSigmoid(logits, 0, "hm_TL_${stamp}.png")
                dumpHeatmapSigmoid(logits, 1, "hm_TR_${stamp}.png")
                dumpHeatmapSigmoid(logits, 2, "hm_BR_${stamp}.png")
                dumpHeatmapSigmoid(logits, 3, "hm_BL_${stamp}.png")
                dumpsWritten++
                if (ENABLE_VERBOSE) Log.i(TAG, "Dumped input/heatmaps to ${cacheDir.absolutePath}")
            } catch (e: Exception) {
                Log.w(TAG, "Dump failed: ${e.message}")
            }
        }

        if (ENABLE_TIMING) {
            fun ms(dn: Long) = (dn / 1e6).toFloat()
            val tPrep = ms(t1 - t0)
            val tInfer = ms(t2e - t2s)
            val tPost = ms(t3e - t3s)
            Log.d(TAG, "timings_ms{ preprocess=${"%.2f".format(tPrep)}, infer=${"%.2f".format(tInfer)}, post=${"%.2f".format(tPost)} }")
        }

        return Corners(tl, tr, br, bl)
    }

    // ===== Helpers =====

    private fun safeCrop(bmp: Bitmap, r: Rect): Bitmap? {
        val x = r.left.coerceIn(0, bmp.width - 1)
        val y = r.top.coerceIn(0, bmp.height - 1)
        val w = (r.right - x).coerceIn(1, bmp.width - x)
        val h = (r.bottom - y).coerceIn(1, bmp.height - y)
        return try {
            Bitmap.createBitmap(bmp, x, y, w, h)
        } catch (e: Exception) {
            Log.w(TAG, "safeCrop failed: ${e.message}")
            null
        }
    }

    private fun Float.format1(): String = String.format(Locale.US, "%.1f", this)

    private fun timestamp(): String {
        val sdf = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US)
        return sdf.format(System.currentTimeMillis())
    }

    private fun dumpBitmapPNG(bmp: Bitmap, name: String) {
        val f = File(cacheDir, name)
        FileOutputStream(f).use { out -> bmp.compress(Bitmap.CompressFormat.PNG, 100, out) }
    }

    private fun dumpHeatmapSigmoid(h: Array<Array<Array<FloatArray>>>, ch: Int, name: String) {
        val img = Bitmap.createBitmap(W, H, Bitmap.Config.ARGB_8888)
        val px = IntArray(W * H)
        var i = 0
        var minV = 1f
        var maxV = 0f
        for (yy in 0 until H) for (xx in 0 until W) {
            val s = (1f / (1f + exp(-h[0][yy][xx][ch]))).coerceIn(0f, 1f)
            if (s < minV) minV = s
            if (s > maxV) maxV = s
        }
        val range = (maxV - minV).coerceAtLeast(1e-6f)
        for (yy in 0 until H) for (xx in 0 until W) {
            val s = (1f / (1f + exp(-h[0][yy][xx][ch]))).coerceIn(0f, 1f)
            val n = ((s - minV) / range * 255f).toInt().coerceIn(0, 255)
            px[i++] = Color.argb(255, n, n, n)
        }
        img.setPixels(px, 0, W, 0, 0, W, H)
        dumpBitmapPNG(img, name)
    }
}