package com.contextionary.sudoku

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.DataType
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round

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
 * 3) Parse model outputs into candidate boxes (basic YOLO-ish parser).
 * 4) Rotate model boxes 90° CW (plane mismatch fix).
 * 5) Unletterbox boxes back to bitmap space.
 * 6) Geometric gates + NMS + small cooldown reuse for stability.
 */
class Detector(
    ctx: Context,
    modelAsset: String,
    @Suppress("unused") private val labelsAsset: String,
    private val enableNnapi: Boolean = false,
    private val numThreads: Int = 4
) {
    // ---- Constants / knobs ---------------------------------------------------
    private val TAG = "Detector"
    private val INPUT_SIZE = 640
    private val COOLDOWN_FRAMES = 3
    private val MAX_CANDIDATES = 300

    // ---- Public fields some UI code expects ----------------------------------
    var inputW: Int = INPUT_SIZE; private set
    var inputH: Int = INPUT_SIZE; private set
    var inputC: Int = 3;          private set
    var lastMaxDets: Int = 0;     private set

    // ---- Interpreter and IO metadata ----------------------------------------
    private var interpreter: Interpreter
    private var isNCHW: Boolean = false
    private var outShapes: List<IntArray> = emptyList()
    private var printedModelInfo = false

    // ---- Runtime state -------------------------------------------------------
    private var frameCount = 0
    private var lastKept: List<Det> = emptyList()
    private var coolDown = 0

    init {
        interpreter = openInterpreter(ctx, modelAsset, numThreads)
        inspectModelIO()   // fills inputW/H/C, isNCHW, outShapes, etc.
        logModelSummary()
    }

    // -------------------------------------------------------------------------
    //  TFLite interpreter setup
    // -------------------------------------------------------------------------
    private fun openInterpreter(ctx: Context, modelAssetPath: String, threads: Int): Interpreter {
        val options = Interpreter.Options().apply {
            setNumThreads(threads)
            if (enableNnapi) {
                try {
                    addDelegate(NnApiDelegate())
                    Log.d(TAG, "NNAPI delegate added")
                } catch (t: Throwable) {
                    Log.w(TAG, "NNAPI not available; using CPU", t)
                }
            }
        }
        val model = FileUtil.loadMappedFile(ctx, modelAssetPath)
        return Interpreter(model, options)
    }

    /**
     * Read the *actual* model IO and allocate tensors up front.
     */
    private fun inspectModelIO() {
        val it = interpreter

        // Input tensor shape & layout
        val inT = it.getInputTensor(0)
        val inShape = inT.shape() // rank 4 expected
        val rank = inShape.size
        if (rank != 4) {
            throw IllegalStateException("Unsupported input rank=$rank, shape=${inShape.contentToString()}")
        }

        // Guess layout by where channel dimension is (3 or 1)
        when {
            inShape[1] == 3 || inShape[1] == 1 -> {
                // [1, C, H, W]
                isNCHW = true
                inputC = inShape[1]
                inputH = inShape[2]
                inputW = inShape[3]
            }
            inShape[3] == 3 || inShape[3] == 1 -> {
                // [1, H, W, C]
                isNCHW = false
                inputH = inShape[1]
                inputW = inShape[2]
                inputC = inShape[3]
            }
            else -> {
                throw IllegalStateException("Cannot infer layout from input shape=${inShape.contentToString()}")
            }
        }

        // If the model expects something other than our default 640x640, respect it
        if (inputW != INPUT_SIZE || inputH != INPUT_SIZE) {
            Log.w(TAG, "Model expects $inputW x $inputH (C=$inputC, ${if (isNCHW) "NCHW" else "NHWC"}), not $INPUT_SIZE — will resize inputs accordingly.")
        }

        // Ensure tensors are allocated for the current input shape
        it.resizeInput(0, inShape)
        it.allocateTensors()

        // Output tensors
        val outs = ArrayList<IntArray>()
        for (i in 0 until it.outputTensorCount) {
            outs += it.getOutputTensor(i).shape()
        }
        outShapes = outs
    }

    fun logModelSummary() {
        if (printedModelInfo) return
        printedModelInfo = true

        val it = interpreter
        val inT = it.getInputTensor(0)
        val inQ = inT.quantizationParams()
        Log.i(TAG, "=== Model Summary ===")
        Log.i(TAG, "Input0 shape=${inT.shape().contentToString()} type=${inT.dataType()} qScale=${inQ.scale} qZp=${inQ.zeroPoint} layout=${if (isNCHW) "NCHW" else "NHWC"}")

        for (i in 0 until it.outputTensorCount) {
            val t = it.getOutputTensor(i)
            val q = t.quantizationParams()
            Log.i(TAG, "Out[$i] shape=${t.shape().contentToString()} type=${t.dataType()} qScale=${q.scale} qZp=${q.zeroPoint}")
        }
    }

    // -------------------------------------------------------------------------
    //  Pre-processing (letterbox) : Bitmap -> NHWC float32 [1,H,W,C] + transform
    // -------------------------------------------------------------------------
    private fun preprocessLetterbox(src: Bitmap): Triple<Array<Array<Array<FloatArray>>>, LbTransform, Bitmap> {
        val sizeW = inputW
        val sizeH = inputH

        val srcW = src.width
        val srcH = src.height

        // keep aspect ratio inside inputW x inputH
        val scale = min(sizeW.toFloat() / srcW, sizeH.toFloat() / srcH)
        val newW = (srcW * scale).toInt().coerceAtLeast(1)
        val newH = (srcH * scale).toInt().coerceAtLeast(1)
        val padX = (sizeW - newW) / 2
        val padY = (sizeH - newH) / 2

        val lb = Bitmap.createBitmap(sizeW, sizeH, Bitmap.Config.ARGB_8888)
        val c = android.graphics.Canvas(lb)
        c.drawColor(android.graphics.Color.rgb(114, 114, 114)) // YOLO gray
        val dst = android.graphics.Rect(padX, padY, padX + newW, padY + newH)
        c.drawBitmap(src, null, dst, null)

        // Build NHWC float32 [1,H,W,C] normalized to [0,1]
        val out = Array(1) { Array(sizeH) { Array(sizeW) { FloatArray(inputC) } } }
        val pixels = IntArray(sizeW * sizeH)
        lb.getPixels(pixels, 0, sizeW, 0, 0, sizeW, sizeH)
        var i = 0
        for (y in 0 until sizeH) {
            for (x in 0 until sizeW) {
                val p = pixels[i++]
                val r = ((p ushr 16) and 0xFF) / 255f
                val g = ((p ushr 8) and 0xFF) / 255f
                val b = (p and 0xFF) / 255f
                if (inputC == 3) {
                    out[0][y][x][0] = r
                    out[0][y][x][1] = g
                    out[0][y][x][2] = b
                } else { // grayscale
                    val gray = 0.299f * r + 0.587f * g + 0.114f * b
                    out[0][y][x][0] = gray
                }
            }
        }

        // For overlay math we still treat "model plane" size as square reference.
        return Triple(out, LbTransform(scale, padX, padY, min(sizeW, sizeH)), lb)
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
    //  Rotation fix (model plane → bitmap plane)
    // -------------------------------------------------------------------------
    /**
     * Rotate a box in the model plane by +90° CW, within a given plane size.
     * (cx,cy,w,h) -> (W - cy, cx, h, w)
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
    //  Public data model for UI
    // -------------------------------------------------------------------------
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
    //  Inference entry point
    // -------------------------------------------------------------------------
    fun infer(
        bmp: Bitmap,
        scoreThresh: Float = 0.55f,
        maxDets: Int = 6
    ): List<Det> {
        frameCount++
        Log.i(TAG, "infer() start frame=$frameCount")

        // 1) Preprocess to NHWC float32 [1,H,W,C] in [0,1]
        val (inputNHWC, lb, _) = preprocessLetterbox(bmp)

        // 2) Build runtime input to match interpreter expectations (dtype + layout)
        val inT = interpreter.getInputTensor(0)
        val inType = inT.dataType()
        val runtimeInput: Any = when (inType) {
            DataType.FLOAT32 -> {
                if (!isNCHW) {
                    // NHWC already
                    inputNHWC
                } else {
                    // Convert NHWC -> NCHW
                    val arr = Array(1) { Array(inputC) { Array(inputH) { FloatArray(inputW) } } }
                    for (y in 0 until inputH)
                        for (x in 0 until inputW)
                            for (c in 0 until inputC)
                                arr[0][c][y][x] = inputNHWC[0][y][x][c]
                    arr
                }
            }
            DataType.INT8, DataType.UINT8 -> {
                val q = inT.quantizationParams()
                val scale = q.scale
                val zp = q.zeroPoint
                val isUint8 = (inType == DataType.UINT8)
                val bb = java.nio.ByteBuffer
                    .allocateDirect(1 * inputH * inputW * inputC)
                    .order(java.nio.ByteOrder.nativeOrder())
                fun clamp(v: Int, u8: Boolean) = if (u8) v.coerceIn(0, 255) else v.coerceIn(-128, 127)

                if (!isNCHW) {
                    for (y in 0 until inputH) for (x in 0 until inputW) {
                        val rQ = clamp(round(inputNHWC[0][y][x][0] / scale + zp).toInt(), isUint8)
                        val gQ = clamp(round(inputNHWC[0][y][x][1] / scale + zp).toInt(), isUint8)
                        val bQ = clamp(round(inputNHWC[0][y][x][2] / scale + zp).toInt(), isUint8)
                        bb.put(rQ.toByte()); bb.put(gQ.toByte()); bb.put(bQ.toByte())
                    }
                } else {
                    // NHWC -> NCHW quantized write order (C major)
                    val plane = inputH * inputW
                    val tmp = FloatArray(plane * inputC)
                    var k = 0
                    for (c in 0 until inputC)
                        for (y in 0 until inputH)
                            for (x in 0 until inputW)
                                tmp[k++] = inputNHWC[0][y][x][c]

                    // Write channel-by-channel
                    k = 0
                    for (c in 0 until inputC) {
                        for (j in 0 until plane) {
                            val qv = clamp(round(tmp[k++] / scale + zp).toInt(), isUint8)
                            bb.put(qv.toByte())
                        }
                    }
                }
                bb.rewind()
                bb
            }
            else -> inputNHWC // fallback
        }

        // 3) Allocate output buffers to exactly match model outputs
        val outCount = interpreter.outputTensorCount
        val outputsMap = HashMap<Int, Any>(outCount)
        for (i in 0 until outCount) {
            val t = interpreter.getOutputTensor(i)
            val s = t.shape()
            val dt = t.dataType()
            val arr: Any = when (dt) {
                DataType.FLOAT32 -> when (s.size) {
                    1 -> FloatArray(s[0])
                    2 -> Array(s[0]) { FloatArray(s[1]) }
                    3 -> Array(s[0]) { Array(s[1]) { FloatArray(s[2]) } }
                    4 -> Array(s[0]) { Array(s[1]) { Array(s[2]) { FloatArray(s[3]) } } }
                    else -> throw IllegalStateException("Unsupported FLOAT32 output rank ${s.size} at $i: ${s.contentToString()}")
                }
                DataType.INT8, DataType.UINT8 -> when (s.size) {
                    1 -> ByteArray(s[0])
                    2 -> Array(s[0]) { ByteArray(s[1]) }
                    3 -> Array(s[0]) { Array(s[1]) { ByteArray(s[2]) } }
                    4 -> Array(s[0]) { Array(s[1]) { Array(s[2]) { ByteArray(s[3]) } } }
                    else -> throw IllegalStateException("Unsupported INT8/UINT8 output rank ${s.size} at $i: ${s.contentToString()}")
                }
                else -> when (s.size) {
                    1 -> FloatArray(s[0])
                    2 -> Array(s[0]) { FloatArray(s[1]) }
                    3 -> Array(s[0]) { Array(s[1]) { FloatArray(s[2]) } }
                    4 -> Array(s[0]) { Array(s[1]) { Array(s[2]) { FloatArray(s[3]) } } }
                    else -> throw IllegalStateException("Unsupported output rank ${s.size} at $i")
                }
            }
            outputsMap[i] = arr
        }

        // 4) Run inference (explicit tensors were already allocated in init)
        try {
            interpreter.runForMultipleInputsOutputs(arrayOf(runtimeInput), outputsMap)
        } catch (t: Throwable) {
            Log.e(TAG, "infer() failed: ${t.message}", t)
            lastMaxDets = 0
            return emptyList()
        }

        // 5) Pick the first output as the detection head and parse in a YOLO-ish way.
        //    (If your model really uses multi-head, we can adjust once we see the shapes in logs.)
        val headTensor = interpreter.getOutputTensor(0)
        val headShape = headTensor.shape() // e.g., [1, C, N] or [1, N, C]
        val headType = headTensor.dataType()

        // Helper to read as unified float with dequant if needed
        val hQ = headTensor.quantizationParams()
        val hScale = hQ.scale
        val hZp = hQ.zeroPoint
        val isFloatHead = (headType == DataType.FLOAT32)
        val isUint8Head = (headType == DataType.UINT8)

        // Determine layout of the head (channels-first vs last)
        var channelsFirst = true
        var N: Int
        var C: Int
        if (headShape.size == 3 && headShape[0] == 1) {
            val d1 = headShape[1]; val d2 = headShape[2]
            if (d1 <= 64 && d2 >= 1000) { // [1, C, N]
                channelsFirst = true; C = d1; N = d2
            } else {                       // [1, N, C]
                channelsFirst = false; N = d1; C = d2
            }
        } else {
            // Fallback default
            channelsFirst = true; C = 5; N = 8400
        }

        // Pull the array back from outputsMap[0] with the right type signature
        val cfF = outputsMap[0] as? Array<Array<FloatArray>>
        val clF = outputsMap[0] as? Array<Array<FloatArray>>
        val cfB = outputsMap[0] as? Array<Array<ByteArray>>
        val clB = outputsMap[0] as? Array<Array<ByteArray>>

        fun read(i: Int, c: Int): Float {
            return if (channelsFirst) {
                if (isFloatHead) {
                    (outputsMap[0] as Array<Array<FloatArray>>)[0][c][i]
                } else {
                    val b = (outputsMap[0] as Array<Array<ByteArray>>)[0][c][i]
                    val q = if (isUint8Head) (b.toInt() and 0xFF) else b.toInt()
                    (q - hZp) * hScale
                }
            } else {
                if (isFloatHead) {
                    (outputsMap[0] as Array<Array<FloatArray>>)[0][i][c]
                } else {
                    val b = (outputsMap[0] as Array<Array<ByteArray>>)[0][i][c]
                    val q = if (isUint8Head) (b.toInt() and 0xFF) else b.toInt()
                    (q - hZp) * hScale
                }
            }
        }

        // If the head emits logits, squash; else identity.
        fun toProb(x: Float): Float = if (x in 0f..1f) x else (1f / (1f + kotlin.math.exp(-x)))

        // 6) Parse predictions → candidates (YOLO-ish: [cx,cy,w,h,obj,cls...])
        val minBoxSizePx = 90f
        val maxAreaRef = 0.75f * min(inputW, inputH) * min(inputW, inputH)
        val cand = ArrayList<Det>(MAX_CANDIDATES)
        val scoreIdx = if (C >= 5) 4 else (C - 1).coerceAtLeast(0)
        val nClassCh = (C - 5).coerceAtLeast(0)

        for (i in 0 until N) {
            val obj = toProb(read(i, scoreIdx))
            val clsProb = if (nClassCh >= 2) {
                var best = 0f; var cc = 5
                while (cc < C) { val p = toProb(read(i, cc)); if (p > best) best = p; cc++ }
                best
            } else 1f
            val score = obj * clsProb
            if (score < scoreThresh) continue

            // Model-plane coordinates are 0..inputW/H (we map to a square ref for rotate step)
            val cx = read(i, 0) * inputW
            val cy = read(i, 1) * inputH
            val w  = read(i, 2) * inputW
            val h  = read(i, 3) * inputH

            if (w < minBoxSizePx || h < minBoxSizePx) continue
            val ar = if (h <= 0f) Float.POSITIVE_INFINITY else (w / h)
            if (ar < 0.85f || ar > 1.18f) continue
            if (w * h > maxAreaRef) continue

            // Rotate +90° CW in model plane then unletterbox (use min(inputW,inputH) as plane)
            //val plane = min(inputW, inputH).toFloat()
            //val rot = rotateModelBox90CW(cx, cy, w, h, plane)
            //val boxBmp = unletterboxRect(rot[0], rot[1], rot[2], rot[3], lb, bmp.width, bmp.height)

            //cand += Det(
            //    box = boxBmp,
            //    score = score,
            //    cls = 0,
            //    cx640 = rot[0], cy640 = rot[1], w640 = rot[2], h640 = rot[3]
            //)

            // No rotation needed: map model-plane box directly back to bitmap space
            val boxBmp = unletterboxRect(cx, cy, w, h, lb, bmp.width, bmp.height)

            cand += Det(
                box = boxBmp,
                score = score,
                cls = 0,
                cx640 = cx, cy640 = cy, w640 = w, h640 = h
            )

            if (cand.size >= MAX_CANDIDATES) break
        }

        // 7) Cooldown reuse
        if (cand.isEmpty()) {
            lastMaxDets = 0
            if (coolDown > 0) { coolDown--; return lastKept }
            return emptyList()
        }

        // 8) NMS + cap + bookkeeping
        val kept = nms(cand, 0.50f)
        val keptCapped = if (kept.size > maxDets) kept.sortedByDescending { it.score }.take(maxDets) else kept
        lastMaxDets = keptCapped.size
        Log.i(TAG, "infer() kept=${keptCapped.size} (score≥$scoreThresh)")

        lastKept = keptCapped
        coolDown = COOLDOWN_FRAMES
        return keptCapped
    }
}