package com.contextionary.sudoku

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class MainActivity : ComponentActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var detector: Detector
    private lateinit var analyzerExecutor: ExecutorService
    private lateinit var cornerRefiner: CornerRefiner
    private lateinit var shutter: android.media.MediaActionSound

    // === Timing / throttling ===
    private var frameIndex = 0
    private var lastInferMs = 0L
    private val minInferIntervalMs = 120L
    private val skipEvery = 1

    // === HUD thresholds (detector and corners) ===
    private val HUD_DET_THRESH = 0.55f
    private val HUD_MAX_DETS = 6

    // === Corner gating params ===
    private val CORNER_PEAK_THR = 0.90f      // all four must be >= this
    private val AREA_RATIO_MIN = 0.90f       // quadArea >= 90% of detector box area
    private val AREA_RATIO_MAX = 1.20f       // and <= 120% of detector box area
    private val SIDE_RATIO_MAX  = 1.8f       // side length max/min bound
    private val ASPECT_TOL      = 1.30f      // aspect similarity (±30% in ratio)

    // Jitter gates in model (128×128) space
    private val JITTER_WINDOW = 3            // N frames to average over
    private val MAX_JITTER_PX128 = 3f        // max allowed avg per-corner motion per frame

    // === Best-of-N locking ===
    // How many consecutive passing frames we buffer before locking the single best.
    // Change to 2, 4, etc.
    companion object {
        private const val STREAK_N = 2

        // Show/hide the green 128×128 crop rectangle on the HUD
        private const val SHOW_CROP_OVERLAY = false
    }

    // === Lock / stability state ===
    private var locked = false

    // Deque of last N passing frames; we’ll pick the best one to lock.
    private data class PassingFrame(
        val corners: CornerRefiner.Corners,
        val roi: RectF,
        val peaks: FloatArray,       // size 4
        val minPeak: Float,
        val tsMs: Long
    )
    private val passing = ArrayDeque<PassingFrame>()

    // For jitter check we only compare to previous frame in 128-space
    private data class Quad128(
        val tlX: Float, val tlY: Float,
        val trX: Float, val trY: Float,
        val brX: Float, val brY: Float,
        val blX: Float, val blY: Float
    )
    private val jitterHistory = ArrayDeque<Quad128>()

    private val askCameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) startCamera() else finish() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.preview)
        overlay = findViewById(R.id.overlay)

        // Camera shutter sound
        shutter = android.media.MediaActionSound()
        shutter.load(android.media.MediaActionSound.SHUTTER_CLICK)

        previewView.scaleType = PreviewView.ScaleType.FIT_CENTER
        overlay.setUseFillCenter(previewView.scaleType == PreviewView.ScaleType.FILL_CENTER)
        overlay.setCornerPeakThreshold(CORNER_PEAK_THR)

        // Detector
        try {
            detector = Detector(
                this,
                modelAsset = "models/grid_detector_int8_dynamic.tflite",
                labelsAsset = "labels.txt",
                enableNnapi = true,
                numThreads = 4
            )
            Log.i("MainActivity", "Detector created OK with dynamic model.")
        } catch (t: Throwable) {
            Log.e("MainActivity", "Detector init FAILED", t)
            overlay.setSourceSize(previewView.width, previewView.height)
            overlay.updateBoxes(emptyList(), HUD_DET_THRESH, 0)
            overlay.updateCorners(null, null)
            overlay.updateCornerCropRect(null)
        }

        // Corner refiner
        cornerRefiner = CornerRefiner(
            this,
            modelAsset = "models/corner_heatmaps3_fp32.tflite",
            numThreads = 2
        )

        cornerRefiner.setDebugConfig(
            verbose = true,     // per-corner peaks & soft-argmax, timings
            timing = true,
            sanity = true,
            fileDumps = false   // set true if you want PNG dumps to /cache
        )

        analyzerExecutor = Executors.newSingleThreadExecutor()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            askCameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::analyzerExecutor.isInitialized) analyzerExecutor.shutdown()
        if (::shutter.isInitialized) {
            try { shutter.release() } catch (_: Throwable) {}
        }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()

            val preview = Preview.Builder()
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(960, 720))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setOutputImageRotationEnabled(true)
                .setTargetRotation(previewView.display.rotation)
                .build()

            analysis.setAnalyzer(analyzerExecutor) { proxy ->
                try {
                    if (locked) { proxy.close(); return@setAnalyzer } // freeze after lock

                    frameIndex++
                    if (frameIndex % (skipEvery + 1) != 0) { proxy.close(); return@setAnalyzer }

                    val now = SystemClock.elapsedRealtime()
                    if (now - lastInferMs < minInferIntervalMs) { proxy.close(); return@setAnalyzer }
                    lastInferMs = now

                    val bmp = proxy.toBitmapRGBA() ?: run {
                        Log.w("MainActivity", "toBitmapRGBA() returned null (fmt=${proxy.format})")
                        proxy.close(); return@setAnalyzer
                    }

                    // 1) Grid detection
                    val dets = detector.infer(bmp, scoreThresh = HUD_DET_THRESH, maxDets = HUD_MAX_DETS)

                    // Pick the most centered grid
                    val cxImg = bmp.width / 2f
                    val cyImg = bmp.height / 2f
                    val picked = dets.minByOrNull { det ->
                        val dx = det.box.centerX() - cxImg
                        val dy = det.box.centerY() - cyImg
                        dx * dx + dy * dy
                    }
                    val toShow = if (picked != null) listOf(picked) else emptyList()

                    // 2) Corner refine on the chosen ROI
                    val (corners, peaks) = if (picked != null) {
                        val c = try { cornerRefiner.refine(bmp, picked.box) } catch (t: Throwable) {
                            Log.w("MainActivity", "Corner refine failed", t); null
                        }
                        val p: FloatArray? = cornerRefiner.getLastPeaks()
                        Pair(c, p)
                    } else Pair(null, null)

                    // Expanded ROI (for green crop overlay) — hidden unless toggled on
                    val cropRect = if (SHOW_CROP_OVERLAY) cornerRefiner.getLastExpandedRoi() else null

                    // 3) Update HUD every frame
                    runOnUiThread {
                        overlay.setSourceSize(bmp.width, bmp.height)
                        overlay.updateBoxes(toShow, HUD_DET_THRESH, HUD_MAX_DETS)
                        overlay.updateCorners(corners, peaks)
                        overlay.updateCornerCropRect(cropRect) // this will be null when hidden
                    }

                    // 4) Full gating: confidence, geometry, area, aspect, jitter, cyan-guard
                    if (picked != null && corners != null && peaks != null && peaks.size >= 4) {
                        val roi = picked.box
                        val minPeak = peaks.minOrNull() ?: 0f
                        val allFourHigh = peaks.all { it >= CORNER_PEAK_THR }

                        // Geometry sanity
                        val geomOk = isConvexAndPositiveTLTRBRBL(corners.tl, corners.tr, corners.br, corners.bl) &&
                                sideLenRatio(corners.tl, corners.tr, corners.br, corners.bl) <= SIDE_RATIO_MAX

                        // Area agreement lower & upper bound
                        val areaQuad = quadArea(corners)
                        val areaRed  = (roi.width() * roi.height()).coerceAtLeast(1f)
                        val areaRatio = areaQuad / areaRed
                        val areaOk = areaRatio >= AREA_RATIO_MIN && areaRatio <= AREA_RATIO_MAX

                        // Aspect similarity (within ±30%) — use ln on doubles, guard div/0
                        val aspectQuad = quadAspectApprox(corners.tl, corners.tr, corners.br, corners.bl)
                        val aspectRed  = aspect(roi)
                        val aspectRatio = (aspectQuad / aspectRed).coerceAtLeast(1e-6f)
                        val aspectOk = kotlin.math.abs(kotlin.math.ln(aspectRatio.toDouble())) <= kotlin.math.ln(ASPECT_TOL.toDouble())

                        // Jitter in 128-space (compare to previous)
                        val (tlx, tly) = bmpTo128(corners.tl, roi)
                        val (trx, try_) = bmpTo128(corners.tr, roi)
                        val (brx, bry) = bmpTo128(corners.br, roi)
                        val (blx, bly) = bmpTo128(corners.bl, roi)
                        val curr128 = Quad128(tlx, tly, trx, try_, brx, bry, blx, bly)
                        val jitterOk = avgJitterPx128(curr128) <= MAX_JITTER_PX128

                        // --- NEW: cyan guard (in VIEW space) ---

                        // --- Cyan guard (in SOURCE/bitmap space via overlay) ---
                        val guardSrc = overlay.getGuardRectInSource()
                        val roiSrc = RectF(roi) // picked box is already in source/bitmap coords

                        // tolerance in source px ~0.8% of min dim, min 2px
                        val tolSrc = (min(bmp.width, bmp.height) / 120f).coerceAtLeast(2f)

                        val guardOk = if (guardSrc != null) {
                            !touchesBorder(roiSrc, guardSrc, tolSrc)
                        } else {
                            true // no guard yet (first draw); don't block
                        }

                        // Map bitmap to view just like the overlay
                        //val useFillCenter = (previewView.scaleType == PreviewView.ScaleType.FILL_CENTER)
                        //val viewW = previewView.width
                        //val viewH = previewView.height
                        //val (s, dw, dh, offX, offY) = computeViewMapping(viewW, viewH, bmp.width, bmp.height, useFillCenter)

                        //val mappedBmp = RectF(offX, offY, offX + dw, offY + dh)
                        //val guide = computeCyanGuide(mappedBmp)
                        //val roiView = mapRectToView(roi, s, offX, offY)

                        // Tolerance roughly matches overlay stroke scaling (≈ 0.8% of min dimension)
                        //val tolPx = (min(viewW, viewH) / 120f).coerceAtLeast(2f)
                        //val guardTouch = touchesBorder(roiView, guide, tolPx)
                        //val guardOk = !guardTouch

                        // Decide
                        val good = allFourHigh && geomOk && areaOk && aspectOk && jitterOk && guardOk

                        // Maintain jitter window always
                        jitterHistory.addLast(curr128)
                        while (jitterHistory.size > JITTER_WINDOW) jitterHistory.removeFirst()

                        if (good) {
                            // Collect passing frame; keep only last N
                            passing.addLast(PassingFrame(corners, roi, peaks, minPeak, now))
                            while (passing.size > STREAK_N) passing.removeFirst()

                            Log.d("Gate",
                                "PASS f=$frameIndex minPeak=${"%.3f".format(minPeak)} " +
                                        "areaR=${"%.2f".format(areaRatio)} aspectOk=$aspectOk geomOk=$geomOk jitterOk=$jitterOk guardOk=$guardOk " +
                                        "(buf=${passing.size}/$STREAK_N)"
                            )

                            // If we have N good in the buffer -> pick best and lock
                            if (passing.size == STREAK_N) {
                                val best = passing.maxByOrNull { it.minPeak }!!
                                locked = true
                                runOnUiThread {
                                    overlay.setLocked(true)
                                    // fire shutter UI/audio you already wired up
                                    overlay.playShutter(previewView)
                                }
                                Log.i(
                                    "Lock",
                                    "Locked with best-of-$STREAK_N (minPeak=${"%.3f".format(best.minPeak)}) tsΔ=${now - best.tsMs}ms"
                                )
                                onGridLocked(best.corners, best.roi, bmp)
                                passing.clear()
                            }
                        } else {
                            // Log the first failing reason to help tuning
                            when {
                                !allFourHigh -> Log.d("Gate", "FAIL f=$frameIndex reason=peaks min=${"%.3f".format(minPeak)} < $CORNER_PEAK_THR")
                                !geomOk      -> Log.d("Gate", "FAIL f=$frameIndex reason=geom")
                                !areaOk      -> Log.d("Gate", "FAIL f=$frameIndex reason=area ratio=${"%.2f".format(areaRatio)} not in [$AREA_RATIO_MIN,$AREA_RATIO_MAX]")
                                !aspectOk    -> Log.d("Gate", "FAIL f=$frameIndex reason=aspect")
                                !jitterOk    -> Log.d("Gate", "FAIL f=$frameIndex reason=jitter")
                                !guardOk     -> Log.d("Gate", "FAIL f=$frameIndex reason=cyan-guard touch tol=${"%.1f".format(tolSrc)}px")
                            }
                            // Reset the best-of-N buffer when a frame fails
                            passing.clear()
                        }
                    } else {
                        // Not enough info: reset buffers
                        passing.clear()
                        jitterHistory.clear()
                    }

                } catch (t: Throwable) {
                    Log.e("Detector", "Analyzer error on frame $frameIndex", t)
                    runOnUiThread {
                        overlay.setSourceSize(previewView.width, previewView.height)
                        overlay.updateBoxes(emptyList(), HUD_DET_THRESH, 0)
                        overlay.updateCorners(null, null)
                        overlay.updateCornerCropRect(null)
                        overlay.setLocked(false)
                    }
                    passing.clear()
                    jitterHistory.clear()
                } finally {
                    proxy.close()
                }
            }

            provider.unbindAll()
            provider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    // Called once when we decide the grid is good & stable.
    private fun onGridLocked(c: CornerRefiner.Corners, roi: RectF, bmp: Bitmap) {
        Log.i("MainActivity", "onGridLocked: roi=$roi TL=${c.tl} TR=${c.tr} BR=${c.br} BL=${c.bl}")
        // TODO: hand off to rectifier pipeline here.
    }

    // ===== Geometry helpers =====

    // Shoelace polygon area for TL→TR→BR→BL
    private fun quadArea(c: CornerRefiner.Corners): Float {
        fun cross(ax: Float, ay: Float, bx: Float, by: Float) = ax * by - ay * bx
        val x1 = c.tl.x; val y1 = c.tl.y
        val x2 = c.tr.x; val y2 = c.tr.y
        val x3 = c.br.x; val y3 = c.br.y
        val x4 = c.bl.x; val y4 = c.bl.y
        val sum = cross(x1, y1, x2, y2) + cross(x2, y2, x3, y3) +
                cross(x3, y3, x4, y4) + cross(x4, y4, x1, y1)
        return kotlin.math.abs(sum) * 0.5f
    }

    private fun isConvexAndPositiveTLTRBRBL(
        tl: CornerRefiner.Pt, tr: CornerRefiner.Pt,
        br: CornerRefiner.Pt, bl: CornerRefiner.Pt
    ): Boolean {
        fun cross(ax: Float, ay: Float, bx: Float, by: Float) = ax * by - ay * bx
        val z1 = cross(tr.x - tl.x, tr.y - tl.y, br.x - tl.x, br.y - tl.y)
        val z2 = cross(br.x - tr.x, br.y - tr.y, bl.x - tr.x, bl.y - tr.y)
        val z3 = cross(bl.x - br.x, bl.y - br.y, tl.x - br.x, tl.y - br.y)
        val z4 = cross(tl.x - bl.x, tl.y - bl.y, tr.x - bl.x, tr.y - bl.y)
        val allPos = z1 > 0 && z2 > 0 && z3 > 0 && z4 > 0
        val allNeg = z1 < 0 && z2 < 0 && z3 < 0 && z4 < 0
        val area = quadArea(CornerRefiner.Corners(tl, tr, br, bl))
        return (allPos || allNeg) && area > 1e-3f
    }

    // Return true if roi intersects/touches the cyan guard (in source coords).
    private fun roiTouchesGuard(roi: RectF, guard: RectF, epsilon: Float = 1.0f): Boolean {
        // Make the guard very slightly bigger so "touching" counts as intersecting.
        val g = RectF(guard.left - epsilon, guard.top - epsilon, guard.right + epsilon, guard.bottom + epsilon)
        return RectF.intersects(roi, g)
    }

    private fun sideLenRatio(
        tl: CornerRefiner.Pt, tr: CornerRefiner.Pt,
        br: CornerRefiner.Pt, bl: CornerRefiner.Pt
    ): Float {
        fun d(a: CornerRefiner.Pt, b: CornerRefiner.Pt): Float {
            val dx = a.x - b.x; val dy = a.y - b.y
            return sqrt(dx*dx + dy*dy)
        }
        val s1 = d(tl, tr); val s2 = d(tr, br); val s3 = d(br, bl); val s4 = d(bl, tl)
        val mx = maxOf(s1, s2, s3, s4)
        val mn = minOf(s1, s2, s3, s4)
        return if (mn <= 1e-6f) Float.POSITIVE_INFINITY else mx / mn
    }

    private fun aspect(r: RectF): Float {
        val w = r.width().coerceAtLeast(1f)
        val h = r.height().coerceAtLeast(1f)
        return w / h
    }

    // Returns true if a and b touch or overlap (edges included).
    private fun rectsTouchOrOverlap(a: RectF, b: RectF): Boolean {
        return !(a.right < b.left || a.left > b.right || a.bottom < b.top || a.top > b.bottom)
    }

    private fun quadAspectApprox(
        tl: CornerRefiner.Pt, tr: CornerRefiner.Pt,
        br: CornerRefiner.Pt, bl: CornerRefiner.Pt
    ): Float {
        fun d(a: CornerRefiner.Pt, b: CornerRefiner.Pt): Float {
            val dx = a.x - b.x; val dy = a.y - b.y
            return sqrt(dx*dx + dy*dy)
        }
        val top = d(tl, tr); val bottom = d(bl, br)
        val left = d(tl, bl); val right = d(tr, br)
        val w = (top + bottom) * 0.5f
        val h = (left + right) * 0.5f
        return if (h <= 1e-6f) Float.POSITIVE_INFINITY else w / h
    }

    // Map bitmap-space back to 128-space to measure jitter there.
    private fun bmpTo128(p: CornerRefiner.Pt, roi: RectF): Pair<Float, Float> {
        val nx = ((p.x - roi.left) / roi.width()).coerceIn(0f, 1f)
        val ny = ((p.y - roi.top)  / roi.height()).coerceIn(0f, 1f)
        // use 127 so 0..127 matches the model grid indexing
        return Pair(nx * 127f, ny * 127f)
    }

    private fun avgJitterPx128(curr: Quad128): Float {
        if (jitterHistory.isEmpty()) return 0f
        val prev = jitterHistory.last()
        fun d(ax: Float, ay: Float, bx: Float, by: Float): Float {
            val dx = ax - bx; val dy = ay - by
            return sqrt(dx*dx + dy*dy)
        }
        val dTL = d(curr.tlX, curr.tlY, prev.tlX, prev.tlY)
        val dTR = d(curr.trX, curr.trY, prev.trX, prev.trY)
        val dBR = d(curr.brX, curr.brY, prev.brX, prev.brY)
        val dBL = d(curr.blX, curr.blY, prev.blX, prev.blY)
        return (dTL + dTR + dBR + dBL) * 0.25f
    }


    // ==== View-space mapping & cyan guard helpers ====

    private fun computeViewMapping(
        viewW: Int,
        viewH: Int,
        srcW: Int,
        srcH: Int,
        useFillCenter: Boolean
    ): Quadruple /* (s, dw, dh, offX, offY) */ {
        val vw = viewW.toFloat()
        val vh = viewH.toFloat()
        val sx = vw / srcW
        val sy = vh / srcH
        val s: Float
        val dw: Float
        val dh: Float
        val offX: Float
        val offY: Float
        if (useFillCenter) {
            s = max(sx, sy)
        } else {
            s = min(sx, sy)
        }
        dw = srcW * s
        dh = srcH * s
        offX = (vw - dw) / 2f
        offY = (vh - dh) / 2f
        return Quadruple(s, dw, dh, offX, offY)
    }

    private data class Quadruple(val s: Float, val dw: Float, val dh: Float, val offX: Float, val offY: Float)

    private fun mapRectToView(r: RectF, s: Float, offX: Float, offY: Float): RectF {
        return RectF(
            offX + r.left * s,
            offY + r.top * s,
            offX + r.right * s,
            offY + r.bottom * s
        )
    }

    private fun computeCyanGuide(mappedBmp: RectF): RectF {
        val side = min(mappedBmp.width(), mappedBmp.height())
        return if (mappedBmp.width() <= mappedBmp.height()) {
            val top = mappedBmp.centerY() - side / 2f
            RectF(mappedBmp.left, top, mappedBmp.left + side, top + side)
        } else {
            val left = mappedBmp.centerX() - side / 2f
            RectF(left, mappedBmp.top, left + side, mappedBmp.top + side)
        }
    }

    // Check if a rect "touches" the border (stroke) of a square, within tolerance.
// We define touching as any edge of 'r' being within tolPx of any side of 'border'
// and overlapping along the orthogonal axis.
    private fun touchesBorder(r: RectF, border: RectF, tolPx: Float): Boolean {
        fun overlap1D(a1: Float, a2: Float, b1: Float, b2: Float): Boolean {
            val lo = max(a1, b1)
            val hi = min(a2, b2)
            return hi >= lo
        }
        // Left side
        val touchLeft = (kotlin.math.abs(r.right - border.left) <= tolPx ||
                kotlin.math.abs(r.left - border.left) <= tolPx) &&
                overlap1D(r.top, r.bottom, border.top, border.bottom)

        // Right side
        val touchRight = (kotlin.math.abs(r.left - border.right) <= tolPx ||
                kotlin.math.abs(r.right - border.right) <= tolPx) &&
                overlap1D(r.top, r.bottom, border.top, border.bottom)

        // Top side
        val touchTop = (kotlin.math.abs(r.bottom - border.top) <= tolPx ||
                kotlin.math.abs(r.top - border.top) <= tolPx) &&
                overlap1D(r.left, r.right, border.left, border.right)

        // Bottom side
        val touchBottom = (kotlin.math.abs(r.top - border.bottom) <= tolPx ||
                kotlin.math.abs(r.bottom - border.bottom) <= tolPx) &&
                overlap1D(r.left, r.right, border.left, border.right)

        return touchLeft || touchRight || touchTop || touchBottom
    }
}

// RGBA8888 -> Bitmap (unchanged)
private fun ImageProxy.toBitmapRGBA(): Bitmap? {
    val plane = planes.firstOrNull() ?: return null
    val w = width
    val h = height
    val rowStride = plane.rowStride
    val pixelStride = plane.pixelStride
    if (pixelStride != 4) {
        Log.w("MainActivity", "Unexpected pixelStride=$pixelStride for RGBA_8888")
        return null
    }

    val needed = w * h * 4
    val contiguous = (rowStride == w * 4)

    val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
    val buffer: java.nio.ByteBuffer = plane.buffer

    return try {
        if (contiguous) {
            buffer.rewind()
            val slice = buffer.duplicate()
            slice.rewind()
            val safeLimit = min(slice.capacity(), needed)
            slice.limit(safeLimit)
            bmp.copyPixelsFromBuffer(slice)
        } else {
            val row = ByteArray(w * 4)
            val dst = java.nio.ByteBuffer.allocateDirect(needed)
            for (y in 0 until h) {
                buffer.position(y * rowStride)
                buffer.get(row, 0, row.size)
                dst.put(row)
            }
            dst.rewind()
            bmp.copyPixelsFromBuffer(dst)
        }
        bmp
    } catch (t: Throwable) {
        Log.e("MainActivity", "toBitmapRGBA() failed", t)
        null
    }
}