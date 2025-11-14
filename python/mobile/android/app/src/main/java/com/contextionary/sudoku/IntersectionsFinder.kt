package com.contextionary.sudoku

import android.content.Context
import android.graphics.*
import android.util.Log
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

data class IntersectionsResult(
    /** Points in SOURCE/bitmap coords (x,y), sized ~100 (10×10). Ordered row-major if gridized succeeds. */
    val points: List<PointF>,
    /** Heatmap-space peak scores (0..1) for each returned point, same order as 'points'. */
    val scores: FloatArray,
    /** Expanded ROI we fed the model (source/bitmap coords). Useful for overlays/jitter mapping. */
    val expandedRoiSrc: Rect,
)

/**
 * Runs a 128×128 single-channel TFLite model that outputs a 1×1×128×128 heatmap.
 * Post-process:
 *  - local-max NMS (3×3), threshold thrPred
 *  - cap peaks to topK
 *  - optional "gridize": coarsely cluster into 10 rows/10 cols; sort row-major; keep 100.
 *
 * This version also writes a small "intersections parity pack" for each call to infer(...),
 * under filesDir/rectify_debug/intersections_YYYYMMDD_HHMMSS[_TAG]/.
 */
class IntersectionsFinder(
    private val ctx: Context,
    modelAsset: String = "models/intersections_fp16.tflite",
    numThreads: Int = 2,
    enableNnapi: Boolean = true,
) {
    private val interpreter: Interpreter
    private var delegate: Delegate? = null
    private val nnapiEnabled: Boolean

    private val inputH = 128
    private val inputW = 128

    init {
        val opts = Interpreter.Options().apply {
            setNumThreads(numThreads)
        }
        var nnapi: NnApiDelegate? = null
        if (enableNnapi) {
            try {
                nnapi = NnApiDelegate()
                opts.addDelegate(nnapi)
            } catch (_: Throwable) {
                nnapi = null
            }
        }
        delegate = nnapi
        nnapiEnabled = (nnapi != null)

        val buf = FileUtil.loadMappedFile(ctx, modelAsset)
        interpreter = Interpreter(buf, opts)

        Log.i("IntersectionsFinder", "Loaded $modelAsset  NNAPI=$nnapiEnabled  threads=$numThreads")
    }

    fun close() {
        try { interpreter.close() } catch (_: Throwable) {}
        try { (delegate as? NnApiDelegate)?.close() } catch (_: Throwable) {}
    }

    // ---------- dump helpers ----------
    private fun debugRoot(): File = File(ctx.filesDir, "rectify_debug").apply { mkdirs() }

    /** Safe wrapper that runs [block] only when outDir != null, and creates parent folders. */
    private inline fun withDump(outDir: File?, name: String, block: (File) -> Unit) {
        if (outDir == null) return
        try {
            outDir.mkdirs()
            val dst = File(outDir, name)
            dst.parentFile?.mkdirs()
            block(dst)
        } catch (t: Throwable) {
            Log.w("IntersectionsFinder", "Dump failed for $name in ${outDir.absolutePath}", t)
        }
    }

    private fun savePng(dst: File, bmp: Bitmap) {
        FileOutputStream(dst).use { fos -> bmp.compress(Bitmap.CompressFormat.PNG, 100, fos) }
    }
    private fun writeText(dst: File, s: String) {
        dst.writeText(s)
    }
    private fun dumpFloatBin(dst: File, data: FloatArray) {
        val bb = ByteBuffer.allocate(data.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        data.forEach { bb.putFloat(it) }
        FileOutputStream(dst).use { it.write(bb.array()) }
    }
    private fun bitmapFromFloatsGray01(f: FloatArray, w: Int, h: Int): Bitmap {
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val px = IntArray(w * h)
        for (i in 0 until w * h) {
            val v = (255f * f[i].coerceIn(0f, 1f)).toInt()
            px[i] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
        }
        out.setPixels(px, 0, w, 0, 0, w, h)
        return out
    }

    /**
     * @param src     RGBA bitmap (camera frame).
     * @param roiSrc  detection RectF in source coords (red box).
     * @param padFrac symmetric padding to square-expand the ROI (e.g., 0.08f).
     *
     * Returns IntersectionsResult or null if too few peaks / gridization fails.
     */
    fun infer(
        src: Bitmap,
        roiSrc: RectF,
        padFrac: Float = 0.08f,
        thrPred: Float = 0.80f,
        topK: Int = 140,      // allow overshoot; we’ll gridize to 100
        requireGridize: Boolean = true,
        dumpDebug: Boolean = false,
        dumpTag: String? = null
    ): IntersectionsResult? {

        // Create dump dir for this call if requested
        val ts = java.text.SimpleDateFormat("yyyyMMdd_HHmmss", java.util.Locale.US)
            .format(java.util.Date())
        val tagSuffix = dumpTag?.takeIf { it.isNotBlank() }?.let { "_$it" } ?: ""
        val dumpDir: File? = if (dumpDebug) {
            File(debugRoot(), "intersections_${ts}${tagSuffix}").also { it.mkdirs() }
        } else null

        // ---- 1) Expanded square ROI (like current flow) ----
        val cx = roiSrc.centerX()
        val cy = roiSrc.centerY()
        val side = max(roiSrc.width(), roiSrc.height()) * (1f + 2f * padFrac)
        val half = side / 2f
        val l = (cx - half).coerceIn(0f, (src.width - 1).toFloat()).toInt()
        val t = (cy - half).coerceIn(0f, (src.height - 1).toFloat()).toInt()
        val r = (cx + half).coerceIn((l + 1).toFloat(), src.width.toFloat()).toInt()
        val b = (cy + half).coerceIn((t + 1).toFloat(), src.height.toFloat()).toInt()
        val roi = Rect(l, t, r, b)
        val crop = Bitmap.createBitmap(src, roi.left, roi.top, roi.width(), roi.height())
        withDump(dumpDir, "roi_src.png") { dst -> savePng(dst, crop) }

        // ---- 2) Preprocess to 128×128 gray in [0,1] ----
        val inFloats = FloatArray(inputW * inputH)
        val resized = Bitmap.createScaledBitmap(crop, inputW, inputH, true)
        val px = IntArray(inputW * inputH)
        resized.getPixels(px, 0, inputW, 0, 0, inputW, inputH)
        var idx = 0
        for (y in 0 until inputH) {
            for (x in 0 until inputW) {
                val c = px[idx]
                val r8 = (c shr 16) and 255
                val g8 = (c shr 8) and 255
                val b8 = (c) and 255
                // BT.601 luma
                val gray = (0.299f * r8 + 0.587f * g8 + 0.114f * b8) / 255f
                inFloats[idx] = gray
                idx++
            }
        }
        withDump(dumpDir, "roi_model_in_128.png") { dst ->
            savePng(dst, bitmapFromFloatsGray01(inFloats, inputW, inputH))
        }

        // Build TFLite input [1,1,H,W]
        val inBuf = Array(1) { Array(1) { Array(inputH) { FloatArray(inputW) } } }
        idx = 0
        for (y in 0 until inputH) for (x in 0 until inputW) inBuf[0][0][y][x] = inFloats[idx++]

        // ---- 3) Inference -> heatmap (1×1×128×128) ----
        val out = Array(1) { Array(1) { Array(inputH) { FloatArray(inputW) } } }
        interpreter.run(inBuf, out)
        // sigmoid
        val hm = FloatArray(inputW * inputH)
        idx = 0
        for (y in 0 until inputH) for (x in 0 until inputW) {
            val z = out[0][0][y][x]
            hm[idx++] = 1f / (1f + kotlin.math.exp(-z))
        }
        withDump(dumpDir, "heatmap.bin") { dst -> dumpFloatBin(dst, hm) }
        withDump(dumpDir, "heatmap.png") { dst ->
            savePng(dst, bitmapFromFloatsGray01(hm, inputW, inputH))
        }

        // ---- 4) Local-max NMS (3×3), threshold, topK ----
        val peaks = mutableListOf<Triple<Int, Int, Float>>() // (x,y,score) in 128-space
        fun hmAt(xx: Int, yy: Int) = hm[yy * inputW + xx]
        for (yy in 1 until inputH - 1) {
            for (xx in 1 until inputW - 1) {
                val v = hmAt(xx, yy)
                if (v < thrPred) continue
                var isMax = true
                loop@ for (dy in -1..1) for (dx in -1..1) {
                    if (dx == 0 && dy == 0) continue
                    if (hmAt(xx + dx, yy + dy) > v) { isMax = false; break@loop }
                }
                if (isMax) peaks.add(Triple(xx, yy, v))
            }
        }
        peaks.sortByDescending { it.third }
        if (peaks.size > topK) peaks.subList(topK, peaks.size).clear()

        // Save peaks before gridize (safe)
        withDump(dumpDir, "peaks_128.csv") { dst ->
            writeText(dst, peaks.joinToString("\n") { "${it.first},${it.second},${"%.6f".format(it.third)}" })
        }

        if (!requireGridize && peaks.size < 90) {
            // meta + return null
            val meta = JSONObject(
                mapOf(
                    "padFrac" to padFrac,
                    "thrPred" to thrPred,
                    "topK" to topK,
                    "inputH" to inputH,
                    "inputW" to inputW,
                    "roi_src" to listOf(roi.left, roi.top, roi.right, roi.bottom),
                    "n_peaks" to peaks.size,
                    "delegate" to if (nnapiEnabled) "NNAPI" else "CPU/XNNPACK",
                    "dumpTag" to (dumpTag ?: ""),
                    "note" to "too_few_peaks_and_requireGridize=false"
                )
            )
            withDump(dumpDir, "meta.json") { dst -> writeText(dst, meta.toString()) }
            Log.i("IntersectionsFinder", "dump (few peaks): ${dumpDir?.absolutePath ?: "n/a"}")
            return null
        }

        // ---- 5) Gridize to 10×10 (row-major) ----
        val pts128 = peaks.map { PointF(it.first.toFloat(), it.second.toFloat()) }
        val scores = peaks.map { it.third }.toFloatArray()
        val grid = gridize10x10(pts128)
        if (grid == null) {
            val meta = JSONObject(
                mapOf(
                    "padFrac" to padFrac,
                    "thrPred" to thrPred,
                    "topK" to topK,
                    "inputH" to inputH,
                    "inputW" to inputW,
                    "roi_src" to listOf(roi.left, roi.top, roi.right, roi.bottom),
                    "n_peaks" to peaks.size,
                    "delegate" to if (nnapiEnabled) "NNAPI" else "CPU/XNNPACK",
                    "dumpTag" to (dumpTag ?: ""),
                    "note" to "gridize_failed"
                )
            )
            withDump(dumpDir, "meta.json") { dst -> writeText(dst, meta.toString()) }
            Log.i("IntersectionsFinder", "dump (gridize fail): ${dumpDir?.absolutePath ?: "n/a"}")
            return null
        }

        // order row-major
        val pts128RM = mutableListOf<PointF>()
        val scrRM = mutableListOf<Float>()
        for (row in 0 until 10) {
            val rowPts = grid[row].sortedBy { it.x }
            for (p in rowPts) {
                pts128RM.add(p)
                // best-effort score lookup by nearest
                var best = 0f; var bestD = Float.POSITIVE_INFINITY
                for (i in pts128.indices) {
                    val dx = pts128[i].x - p.x
                    val dy = pts128[i].y - p.y
                    val d2 = dx*dx + dy*dy
                    if (d2 < bestD) { bestD = d2; best = scores[i] }
                }
                scrRM.add(best)
            }
        }

        // ---- 6) Map 128-space back to SOURCE (expand + scale) ----
        val sx = roi.width().toFloat() / inputW
        val sy = roi.height().toFloat() / inputH
        val ptsSrc = pts128RM.map { p -> PointF(roi.left + p.x * sx, roi.top + p.y * sy) }

        // ---- 7) Dump success pack (safe) ----
        withDump(dumpDir, "grid_128.csv") { dst ->
            writeText(dst,
                pts128RM.zip(scrRM).joinToString("\n") { (p, s) ->
                    "${"%.3f".format(p.x)},${"%.3f".format(p.y)},${"%.6f".format(s)}"
                }
            )
        }
        withDump(dumpDir, "grid_src.csv") { dst ->
            writeText(dst,
                ptsSrc.zip(scrRM).joinToString("\n") { (p, s) ->
                    "${"%.1f".format(p.x)},${"%.1f".format(p.y)},${"%.6f".format(s)}"
                }
            )
        }
        val meta = JSONObject(
            mapOf(
                "padFrac" to padFrac,
                "thrPred" to thrPred,
                "topK" to topK,
                "inputH" to inputH,
                "inputW" to inputW,
                "roi_src" to listOf(roi.left, roi.top, roi.right, roi.bottom),
                "n_peaks" to peaks.size,
                "delegate" to if (nnapiEnabled) "NNAPI" else "CPU/XNNPACK",
                "dumpTag" to (dumpTag ?: ""),
                "note" to "success"
            )
        )
        withDump(dumpDir, "meta.json") { dst -> writeText(dst, meta.toString()) }
        Log.i("IntersectionsFinder", "Intersections dump: ${dumpDir?.absolutePath ?: "n/a"}")

        return IntersectionsResult(points = ptsSrc, scores = scrRM.toFloatArray(), expandedRoiSrc = roi)
    }

    /**
     * Coarse 1-D clustering along Y to get 10 rows, then within each row
     * coarse 1-D clustering along X to get ~10 cols; then keep 10 per row.
     * Robust to small over/undercounts; returns null if it can’t form 10 rows.
     */
    private fun gridize10x10(pts: List<PointF>): Array<MutableList<PointF>>? {
        if (pts.size < 90) return null
        val byY = pts.sortedBy { it.y }
        // split into 10 contiguous bands by Y quantiles
        val rows = Array(10) { mutableListOf<PointF>() }
        for ((i, p) in byY.withIndex()) {
            val r = min(9, (i * 10) / max(1, byY.size)) // 0..9
            rows[r].add(p)
        }
        // guard against empty rows → try simple K-means fallback on Y
        if (rows.any { it.isEmpty() }) {
            val rowsKm = kmeans1D(pts.map { it.y }, 10)
            if (rowsKm == null) return null
            val out = Array(10) { mutableListOf<PointF>() }
            for ((pi, r) in rowsKm.assignments.withIndex()) out[r].add(pts[pi])
            for (r in 0 until 10) out[r].sortBy { it.y }
            return out
        }
        // within each row, keep 10 best-spaced by X
        val outRows = Array(10) { mutableListOf<PointF>() }
        for (r in 0 until 10) {
            val rowPts = rows[r].sortedBy { it.x }
            if (rowPts.size <= 10) {
                outRows[r].addAll(rowPts)
            } else {
                // keep 10 evenly sampled across X
                val n = rowPts.size
                for (k in 0 until 10) {
                    val idx = ((k + 0.5f) * n / 10f).toInt().coerceIn(0, n - 1)
                    outRows[r].add(rowPts[idx])
                }
            }
            if (outRows[r].size < 8) return null
            // trim or pad to 10
            while (outRows[r].size > 10) outRows[r].removeLast()
            while (outRows[r].size < 10) outRows[r].add(outRows[r].last())
        }
        return outRows
    }

    private data class KMeans1D(val centers: FloatArray, val assignments: IntArray)
    private fun kmeans1D(vals: List<Float>, k: Int, iters: Int = 20): KMeans1D? {
        if (vals.isEmpty() || vals.size < k) return null
        val v = vals.sorted()
        val centers = FloatArray(k) { i -> v[((i + 0.5f) * v.size / k).toInt()] }
        val assign = IntArray(vals.size)
        repeat(iters) {
            // assign
            for (i in vals.indices) {
                var bi = 0; var bd = Float.POSITIVE_INFINITY
                for (c in 0 until k) {
                    val d = kotlin.math.abs(vals[i] - centers[c])
                    if (d < bd) { bd = d; bi = c }
                }
                assign[i] = bi
            }
            // update
            val sums = FloatArray(k); val cnts = IntArray(k)
            for (i in vals.indices) { val a = assign[i]; sums[a] += vals[i]; cnts[a]++ }
            for (c in 0 until k) if (cnts[c] > 0) centers[c] = sums[c] / cnts[c]
        }
        return KMeans1D(centers, assign)
    }
}