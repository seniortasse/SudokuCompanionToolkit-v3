package com.contextionary.sudoku

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import kotlin.math.max
import kotlin.math.min

/**
 * Describes the letterbox transform used during pre-processing.
 *
 * The original bitmap is scaled by `scale` to fit inside a SIZE×SIZE gray canvas.
 * The scaled image is then placed at offset (padX, padY).
 *
 * Model coordinates (0..SIZE) can be mapped back to bitmap coordinates by:
 *   xBmp = (xModel - padX) / scale
 *   yBmp = (yModel - padY) / scale
 */
data class LbTransform(
    val scale: Float,
    val padX: Int,
    val padY: Int,
    val size: Int = 640
)

/**
 * Detector
 *
 * Responsibilities:
 * 1) Preprocess camera bitmaps via YOLO-style letterboxing to 640×640.
 * 2) Run the TFLite model (shape-agnostic across CF/CL output layouts).
 * 3) Parse model outputs into candidate boxes.
 * 4) 🚩 Rotate model boxes 90° CW (fixes the plane mismatch you discovered).
 * 5) Unletterbox boxes back to bitmap space.
 * 6) Geometric gates + NMS + small cooldown reuse for stability.
 */
class Detector(
    ctx: Context,
    modelAsset: String,
    @Suppress("unused") private val labelsAsset: String
) {

    // ---- Constants / knobs ---------------------------------------------------

    private val TAG = "Detector"

    /** Model input is 640×640×3 float32 (NHWC). */
    private val INPUT_SIZE = 640

    /** Small reuse window when no fresh boxes pass the gates (frames). */
    private val COOLDOWN_FRAMES = 3

    /** Upper bound of how many candidates we’ll hold before NMS. */
    private val MAX_CANDIDATES = 300

    // ---- State for logging / cooldown ---------------------------------------

    private var printedModelInfo = false
    private var frameCount = 0

    private var lastKept: List<Det> = emptyList()
    private var coolDown = 0

    // ---- TFLite interpreter --------------------------------------------------

    private val tflite: Interpreter = run {
        val opts = Interpreter.Options().apply { setNumThreads(4) }
        val model = FileUtil.loadMappedFile(ctx, modelAsset)
        Interpreter(model, opts)
    }

    /**
     * Single detection returned to the UI.
     *
     * @param box   Rectangle in ORIGINAL BITMAP space (after unletterboxing)
     * @param score Confidence score in [0,1]
     * @param cls   Class index (0 for grids; included for completeness)
     * @param cx640,cy640,w640,h640  Rectangle in MODEL/LETTERBOX space (0..640)
     *                                — we store the FINAL values we used, i.e.
     *                                after the 90° CW rotation fix, so the HUD
     *                                matches the red rectangle visually.
     */
    data class Det(
        val box: RectF,
        val score: Float,
        val cls: Int = 0,
        val cx640: Float,
        val cy640: Float,
        val w640: Float,
        val h640: Float
    )

    // -------------------------------------------------------------------------
    //  Pre-processing (letterbox) : Bitmap -> 640×640 float32 NHWC + transform
    // -------------------------------------------------------------------------

    private fun preprocessLetterbox(
        src: Bitmap
    ): Triple<Array<Array<Array<FloatArray>>>, LbTransform, Bitmap> {

        val size = INPUT_SIZE
        val srcW = src.width
        val srcH = src.height

        // scale so the longer edge touches SIZE, preserving aspect ratio
        val scale = min(size.toFloat() / srcW, size.toFloat() / srcH)
        val newW = (srcW * scale).toInt()
        val newH = (srcH * scale).toInt()
        val padX = (size - newW) / 2
        val padY = (size - newH) / 2

        // 640×640 gray canvas, then draw the scaled src inside it
        val lb = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val c = android.graphics.Canvas(lb)
        c.drawColor(android.graphics.Color.rgb(114, 114, 114)) // YOLO gray
        val dst = android.graphics.Rect(padX, padY, padX + newW, padY + newH)
        c.drawBitmap(src, null, dst, null)

        // NHWC float32 [1,640,640,3], normalized to [0,1]
        val out = Array(1) { Array(size) { Array(size) { FloatArray(3) } } }
        val pixels = IntArray(size * size)
        lb.getPixels(pixels, 0, size, 0, 0, size, size)
        var i = 0
        for (y in 0 until size) {
            for (x in 0 until size) {
                val p = pixels[i++]
                out[0][y][x][0] = ((p ushr 16) and 0xFF) / 255f // R
                out[0][y][x][1] = ((p ushr 8) and 0xFF) / 255f  // G
                out[0][y][x][2] = (p and 0xFF) / 255f           // B
            }
        }

        return Triple(out, LbTransform(scale, padX, padY, size), lb)
    }

    // -------------------------------------------------------------------------
    //  Geometry helpers
    // -------------------------------------------------------------------------

    /** Map a model/letterbox-space rect back to ORIGINAL BITMAP space. */
    private fun unletterboxRect(
        cx640: Float, cy640: Float, w640: Float, h640: Float,
        lb: LbTransform, origW: Int, origH: Int
    ): RectF {
        val x1 = (cx640 - w640 / 2f - lb.padX) / lb.scale
        val y1 = (cy640 - h640 / 2f - lb.padY) / lb.scale
        val x2 = (cx640 + w640 / 2f - lb.padX) / lb.scale
        val y2 = (cy640 + h640 / 2f - lb.padY) / lb.scale

        // Keep inside bitmap bounds; any small overshoot from rounding is trimmed.
        return RectF(
            x1.coerceIn(0f, origW.toFloat()),
            y1.coerceIn(0f, origH.toFloat()),
            x2.coerceIn(0f, origW.toFloat()),
            y2.coerceIn(0f, origH.toFloat())
        )
    }

    /** IoU for NMS. */
    private fun iou(a: RectF, b: RectF): Float {
        val x1 = max(a.left, b.left)
        val y1 = max(a.top, b.top)
        val x2 = min(a.right, b.right)
        val y2 = min(a.bottom, b.bottom)
        val inter = max(0f, x2 - x1) * max(0f, y2 - y1)
        val ua = a.width() * a.height() + b.width() * b.height() - inter
        return if (ua <= 0f) 0f else inter / ua
    }

    /** Simple greedy NMS. */
    private fun nms(dets: List<Det>, thr: Float): List<Det> {
        val sorted = dets.sortedByDescending { it.score }.toMutableList()
        val keep = mutableListOf<Det>()
        while (sorted.isNotEmpty()) {
            val a = sorted.removeAt(0)
            keep += a
            val it = sorted.iterator()
            while (it.hasNext()) if (iou(a.box, it.next().box) > thr) it.remove()
        }
        return keep
    }

    // -------------------------------------------------------------------------
    //  🔧 Rotation fix helpers (model plane → bitmap plane)
    // -------------------------------------------------------------------------

    /**
     * Rotate a box in the 640×640 MODEL plane by +90° (clockwise).
     *
     * You validated on-device that model boxes align with the paper after this
     * transformation. This function is applied BEFORE unletterboxing so the
     * returned bitmap-space rectangle lands where it should.
     *
     * For a 90° CW rotation in a W×H plane (here 640×640):
     *   (cx, cy, w, h) → (cx', cy', w', h') = (W - cy, cx, h, w)
     */
    private inline fun rotateModelBox90CW(
        cx: Float, cy: Float, w: Float, h: Float, planeSize: Float
    ): FloatArray {
        val cxR = planeSize - cy
        val cyR = cx
        val wR  = h
        val hR  = w
        return floatArrayOf(cxR, cyR, wR, hR)
    }

    // -------------------------------------------------------------------------
    //  Inference entry point
    // -------------------------------------------------------------------------

    /**
     * Run detection on a bitmap.
     *
     * @param scoreThresh confidence threshold in [0,1]
     * @param maxDets     maximum boxes to return after NMS
     */
    fun infer(
        bmp: Bitmap,
        scoreThresh: Float = 0.55f,
        maxDets: Int = 6
    ): List<Det> {
        frameCount++

        // 1) Preprocess (letterbox)
        val (input, lb, _) = preprocessLetterbox(bmp)

        // 2) Run model — be robust to output layout differences:
        //    - "channels first":  [1, C, N]
        //    - "channels last" :  [1, N, C]
        val shape = tflite.getOutputTensor(0).shape()
        var channelsFirst = true
        val nA: Int
        val ch: Int
        if (shape.size == 3 && shape[0] == 1) {
            val d1 = shape[1]; val d2 = shape[2]
            if (d1 <= 64 && d2 >= 1000) { // [1, C, N]
                channelsFirst = true; ch = d1; nA = d2
            } else {                      // [1, N, C]
                channelsFirst = false; nA = d1; ch = d2
            }
        } else {
            // Fall back to a common YOLO-ish layout if unknown.
            channelsFirst = true; ch = 5; nA = 8400
        }

        val outCF: Array<Array<FloatArray>>?
        val outCL: Array<Array<FloatArray>>?
        if (channelsFirst) {
            outCF = Array(1) { Array(ch) { FloatArray(nA) } }
            outCL = null
            tflite.run(input, outCF)
        } else {
            outCF = null
            outCL = Array(1) { Array(nA) { FloatArray(ch) } }
            tflite.run(input, outCL)
        }

        fun get(i: Int, c: Int): Float =
            if (channelsFirst) outCF!![0][c][i] else outCL!![0][i][c]

        // 3) One-time model info log + quick scan of the score channel
        val scoreIdx = if (ch >= 5) 4 else (ch - 1).coerceAtLeast(0)
        if (!printedModelInfo) {
            Log.d(TAG, "tflite out[0] shape=${shape.joinToString()} (nA=$nA, ch=$ch, channelsFirst=$channelsFirst)")
            var minS = Float.POSITIVE_INFINITY
            var maxS = Float.NEGATIVE_INFINITY
            val stride = (nA / 400).coerceAtLeast(1)
            var cnt = 0
            for (i in 0 until nA step stride) {
                val s = get(i, scoreIdx)
                if (s < minS) minS = s
                if (s > maxS) maxS = s
                cnt++
            }
            Log.d(TAG, "score channel sample (stride=$stride, n=$cnt): min=$minS, max=$maxS")
            printedModelInfo = true
        }

        // If the model emits logits, squash them; otherwise this is identity.
        fun toProb(x: Float): Float = if (x in 0f..1f) x else (1f / (1f + kotlin.math.exp(-x)))

        // 4) Parse predictions → candidates (cheap gates to keep FPS high)
        val minBoxSizePx = 90f // very small squares are unlikely sudoku grids
        val maxArea640 = 0.75f * lb.size * lb.size
        val cand = ArrayList<Det>(MAX_CANDIDATES)
        val nClassCh = (ch - 5).coerceAtLeast(0)

        for (i in 0 until nA) {
            val objProb = toProb(get(i, scoreIdx))

            // If there are class channels, combine with the best class prob.
            val score = if (nClassCh >= 2) {
                var best = 0f; var c = 5
                while (c < ch) { val p = toProb(get(i, c)); if (p > best) best = p; c++ }
                objProb * best
            } else objProb

            if (score < scoreThresh) continue

            // Raw model/letterbox-space box — coordinates are normalized to [0,1] or already [0,640].
            val cx640 = get(i, 0) * INPUT_SIZE
            val cy640 = get(i, 1) * INPUT_SIZE
            val w640  = get(i, 2) * INPUT_SIZE
            val h640  = get(i, 3) * INPUT_SIZE

            // Geometric sanity gates
            if (w640 < minBoxSizePx || h640 < minBoxSizePx) continue
            val ar = if (h640 <= 0f) Float.POSITIVE_INFINITY else (w640 / h640)
            if (ar < 0.85f || ar > 1.18f) continue // near-square only
            if (w640 * h640 > maxArea640) continue

            // 🔧 ROTATION FIX:
            // Your A/B/C test showed the model plane is rotated w.r.t. bitmap/view.
            // Rotate the model box +90° CW in the 640×640 plane BEFORE unletterboxing.
            val (cxR, cyR, wR, hR) = rotateModelBox90CW(
                cx = cx640, cy = cy640, w = w640, h = h640, planeSize = INPUT_SIZE.toFloat()
            )

            // Now map the rotated model box back to original bitmap space.
            val boxBmp = unletterboxRect(cxR, cyR, wR, hR, lb, bmp.width, bmp.height)

            cand += Det(
                box = boxBmp,
                score = score,
                cls = 0,
                // Store the ROTATED model-space numbers so HUD/green overlays match red.
                cx640 = cxR, cy640 = cyR, w640 = wR, h640 = hR
            )

            if (cand.size >= MAX_CANDIDATES) break
        }

        // 5) Cooldown reuse if nothing passed the gates this frame
        if (cand.isEmpty()) {
            if (coolDown > 0) { coolDown--; return lastKept }
            return emptyList()
        }

        // 6) NMS + cap
        val kept = nms(cand, 0.50f)
        val keptCapped =
            if (kept.size > maxDets) kept.sortedByDescending { it.score }.take(maxDets) else kept

        // 7) Save for cooldown + occasional debug log
        lastKept = keptCapped
        coolDown = COOLDOWN_FRAMES

        if (frameCount % 30 == 0) {
            val top = keptCapped.maxByOrNull { it.score }?.score ?: 0f
            Log.d(TAG, "Kept ${keptCapped.size} boxes after NMS (score≥$scoreThresh, IoU=0.50), top=${"%.3f".format(top)}")
        }

        return keptCapped
    }
}