// ==============================================================================
// MainActivity — CameraX pipeline, gating, rectification handoff, and capture UX
// ==============================================================================
// WHY
// ------------------------------------------------------------------------------
// We need a smooth capture loop that finds a Sudoku grid in real time, validates it with corner heatmaps, and only locks/captures when everything is clearly good. MainActivity coordinates that flow and keeps the UI responsive.
//
// WHAT
// ------------------------------------------------------------------------------
// Sets up CameraX (Preview + ImageAnalysis), throttles frames, runs Detector and CornerRefiner, draws the HUD via OverlayView, enforces multiple gates (peaks, geometry, area, aspect, jitter, cyan-guard), and only then rectifies + classifies. Success triggers a shutter effect and navigates to results.
//
// HOW (high-level)
// ------------------------------------------------------------------------------
// 1) Request CAMERA permission; start CameraX with a Preview and an RGBA ImageAnalysis stream.
// 2) For every analyzed frame (throttled), convert to Bitmap.
// 3) Detector.infer() -> pick the most-centered box.
// 4) CornerRefiner.refine() on that ROI -> peaks and TL/TR/BR/BL.
// 5) OverlayView updates HUD (boxes, peaks, cyan guard, optional green ROI).
// 6) If N consecutive frames pass all gates, try rectification + classification; upon success, lock & celebrate.
//
// FILE ORGANIZATION
// ------------------------------------------------------------------------------
// • Fields: camera views, ML components (Detector, CornerRefiner, DigitClassifier), gates, histories.
// • Lifecycle: onCreate() -> startCamera(); onDestroy() cleanup.
// • Analyzer: throttling logic, detection + refine + HUD update + gating.
// • Handoff: attemptRectifyAndClassify() (best-of-N capture) and onLockedGridCaptured().
// • Geometry helpers: area/aspect/convexity/jitter/guard mapping.
//
// RUNTIME FLOW / HOW THIS FILE IS USED
// ------------------------------------------------------------------------------
// User points camera; cyan guard guides aim. Detector finds grids; we choose the most centered. CornerRefiner validates corners. Passing frames accumulate; on success we rectify + classify, play shutter, and show results. Otherwise we keep scanning — no premature locks.
//
// NOTES
// ------------------------------------------------------------------------------
// Comments are ASCII-only. Original code untouched; just comment lines added. Look for 'gates' in analyzer to see the criteria used before capture.
//
// ==============================================================================
package com.contextionary.sudoku

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
//import android.util.Size
import android.util.Size as UiSize
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

// === OpenCV ===
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
//import org.opencv.core.Rect
import android.graphics.Rect
import android.graphics.PointF
import org.opencv.imgproc.Imgproc
import org.opencv.core.Size as CvSize


import android.view.View
import android.view.ViewGroup
import android.view.Gravity
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ImageView
import android.widget.TextView
import android.widget.Button
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Color


import com.google.android.material.button.MaterialButton
import android.app.Activity



import androidx.appcompat.view.ContextThemeWrapper


// The central orchestrator for camera preview, analysis, HUD, detection, corner refinement, gating, and the final capture/rectify/classify path.
class MainActivity : ComponentActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var detector: Detector
    private lateinit var analyzerExecutor: ExecutorService
    private lateinit var shutter: android.media.MediaActionSound

    // Removed the duplicate lateinit version; keep this one:
    private var digitClassifier: DigitClassifier? = null
    private var handoffInProgress = false

    private var resultsSudokuView: SudokuResultView? = null

    // === Timing / throttling ===
    private var frameIndex = 0
    private var lastInferMs = 0L
    private val minInferIntervalMs = 80L
    private val skipEvery = 0

    // Prevents double-processing while a capture is being rectified/classified.
    @Volatile
    private var isProcessingLockedGrid: Boolean = false


    private var gateState: GateState = GateState.NONE


    // === MM4: results overlay state ===
    private var captureLocked = false


    // Back-compat alias so the rest of the code compiles
    private var locked: Boolean
        get() = captureLocked
        set(value) { captureLocked = value }



    // Gate change helper (updates state + OverlayView HUD)
    private fun changeGate(state: GateState, why: String? = null) {
        if (gateState != state) {
            gateState = state
            runOnUiThread { overlay.setGateState(state) }
            Logx.d("Gate", "to" to state.name, "why" to (why ?: ""))
        }
    }

    // Intersections jitter helper (average per-point motion in 128×128 model space)
    private fun avgJitterPx128(curr: Grid128): Float {
        if (jitterHistory.isEmpty()) return 0f
        val prev = jitterHistory.last()
        val n = kotlin.math.min(curr.xs.size, prev.xs.size)
        var sum = 0f
        for (i in 0 until n) {
            val dx = curr.xs[i] - prev.xs[i]
            val dy = curr.ys[i] - prev.ys[i]
            sum += kotlin.math.sqrt(dx*dx + dy*dy)
        }
        return if (n <= 0) 0f else sum / n
    }


    private var resultsRoot: FrameLayout? = null
    private var resultsImage: ImageView? = null
    private var lastBoardBitmap: Bitmap? = null
    private var lastDigits81: IntArray? = null


    private var resultsDigits: IntArray? = null

    private var resultsConfidences: FloatArray? = null

    // === HUD thresholds (detector and corners) ===
    private val HUD_DET_THRESH = 0.55f
    private val HUD_MAX_DETS = 6

    // === Corner gating params ===
    private val CORNER_PEAK_THR = 0.90f      // all four must be >= this
    private val AREA_RATIO_MIN = 0.90f       // quadArea >= 90% of detector box area
    private val AREA_RATIO_MAX = 1.20f       // and <= 120% of detector box area
    private val SIDE_RATIO_MAX  = 1.8f       // side length max/min bound
    private val ASPECT_TOL      = 1.30f      // aspect similarity (±30% in ratio)




    private lateinit var intersections: IntersectionsFinder

    // New gates
    private val INT_PEAK_THR = 0.25f
    private val JITTER_WINDOW = 3
    private val MAX_JITTER_PX128 = 7f

    // For jitter history of 100 pts (in 128×128 model space)
    private data class Grid128(val xs: FloatArray, val ys: FloatArray) // size=100
    private val jitterHistory = ArrayDeque<Grid128>()

    // Keep last passing frame for best-of-N
    private data class PassingFrame(
        val ptsSrc: List<PointF>,   // 100 src-space points (row-major)
        val roi: RectF,
        val minPeak: Float,
        val tsMs: Long,
        val expandedRoi: Rect       // for overlays / mapping
    )

    private val passing = ArrayDeque<PassingFrame>()


    private val resultsLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        // Either result: retake (RESULT_CANCELED) or keep (RESULT_OK),
        // we want to return to a fresh scanning state.
        resetCaptureForFreshScan()
    }

    // === Best-of-N locking ===
    companion object {
        private const val STREAK_N = 2
        private const val SHOW_CROP_OVERLAY = false
        private const val ROI_PAD_FRAC = 0.08f   // 8% on each side (tweak: 0.06–0.12)

        // === L3 (rectify/classify) tunables ===
        private const val GRID_SIZE = 576           // 9 * 64, square warp target
        private const val CELL_SIZE = 64
        private const val MIN_RECT_PASS_AVG_CONF = 0.75f
        private const val MAX_LOWCONF = 6           // how many cells may be low-confidence
        private const val LOWCONF_THR = 0.60f       // what "low" means, per cell
        //private const val GRID_SIZE = 450  // square pixels for our “rough” board render
    }


    private val askCameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) startCamera() else finish() }

    //private fun dp(v: Int): Int =
    //    (v * resources.displayMetrics.density).toInt()

    // ---- dp helper (keep exactly ONE of these in this class) ----
    private fun Int.dp(): Int = (this * resources.displayMetrics.density).toInt()


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.preview)
        overlay = findViewById(R.id.overlay)

        // OpenCV init (debug loader is fine for dev builds)
        if (!OpenCVLoader.initDebug()) {
            Log.e("OpenCV", "OpenCV init failed")
        }

        // Camera shutter sound
        shutter = android.media.MediaActionSound()
        shutter.load(android.media.MediaActionSound.SHUTTER_CLICK)

        previewView.scaleType = PreviewView.ScaleType.FIT_CENTER
        overlay.setUseFillCenter(previewView.scaleType == PreviewView.ScaleType.FILL_CENTER)
        overlay.setCornerPeakThreshold(CORNER_PEAK_THR)

        overlay.showCornerDots = true
        overlay.showBoxLabels = false
        overlay.showHudText   = false

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



        intersections = IntersectionsFinder(
            this,
            modelAsset = "models/intersections_fp16.tflite",
            numThreads = 2,
            enableNnapi = true
        )

        // HUD: show intersections, hide corner UI
        overlay.showCornerDots = false
        overlay.showIntersections = true


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

    private fun resetCaptureForFreshScan() {
        // Let the analyzer run again
        locked = false
        handoffInProgress = false

        // Unlock and clear the HUD
        overlay.setLocked(false)
        overlay.updateBoxes(emptyList(), HUD_DET_THRESH, 0)
        overlay.updateCorners(null, null)
        overlay.updateCornerCropRect(null)

        // Optional: if you track frames, you can reset it
        // frameIndex = 0

        // Go back to initial gate visuals
        changeGate(GateState.L1, "retake")
    }


    // Create the results Overlay (board + buttons) and size/align them precisely.
    private fun ensureResultsOverlay() {
        if (resultsRoot != null) return

        resultsRoot = FrameLayout(this).apply {
            setBackgroundColor(Color.BLACK)
            alpha = 0f
            isClickable = true
            isFocusable = true
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        }

        // NOTE: gravity CENTER to center the whole column vertically
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setPadding(24.dp(), 24.dp(), 24.dp(), 24.dp())
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        }

        // 1) Board view (hard-size it in post{} once we know the screen size)
        val boardView = SudokuResultView(this).apply {
            id = View.generateViewId()
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                0, // we’ll override height/width later
                1f
            ).apply {
                setMargins(0, 0, 0, 24.dp()) // provisional gap above buttons
            }
        }
        resultsSudokuView = boardView

        // 2) Button row
        val buttonsRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }

        // Styled buttons (your styles.xml themes)
        val retakeBtnCtx = ContextThemeWrapper(this, R.style.Sudoku_Button_Outline)
        val keepBtnCtx   = ContextThemeWrapper(this, R.style.Sudoku_Button_Primary)

        val btnRetake = MaterialButton(
            retakeBtnCtx,
            null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle
        ).apply {
            text = "Retake"
            isAllCaps = false
            layoutParams = LinearLayout.LayoutParams(0, 52.dp(), 1f).apply {
                setMargins(0, 0, 12.dp(), 0)
            }
            setOnClickListener {
                dismissResults(resumePreview = true)
                resumeAnalyzer()
            }
        }

        val btnKeep = MaterialButton(keepBtnCtx).apply {
            text = "Keep"
            isAllCaps = false
            layoutParams = LinearLayout.LayoutParams(0, 52.dp(), 1f).apply {
                setMargins(12.dp(), 0, 0, 0)
            }
            setOnClickListener { onKeepResults() }
        }

        buttonsRow.addView(btnRetake)
        buttonsRow.addView(btnKeep)

        container.addView(boardView)
        container.addView(buttonsRow)
        resultsRoot!!.addView(container)
        (findViewById<ViewGroup>(android.R.id.content)).addView(resultsRoot)

        // ---- After layout: compute a centered square, align buttons to grid ----
        resultsRoot!!.post {
            val rootW = resultsRoot!!.width
            val rootH = resultsRoot!!.height

            val screenMargin = 24.dp()   // outer margin around composition
            val btnHeight    = 52.dp()   // 48–56dp target
            val gapAboveBtns = 24.dp()   // space between board & buttons

            val availW = rootW - 2 * screenMargin
            val availH = rootH - 2 * screenMargin - btnHeight - gapAboveBtns

            val boardSize = minOf(availW, availH)

            // Fix the board to a centered square
            (resultsSudokuView?.layoutParams as LinearLayout.LayoutParams).apply {
                width = boardSize
                height = boardSize
                weight = 0f
                setMargins(0, 0, 0, gapAboveBtns)
                gravity = Gravity.CENTER_HORIZONTAL
            }
            resultsSudokuView?.requestLayout()

            // Make the button row the same width as the board
            (buttonsRow.layoutParams as LinearLayout.LayoutParams).apply {
                width = boardSize
                height = LinearLayout.LayoutParams.WRAP_CONTENT
                gravity = Gravity.CENTER_HORIZONTAL
            }

            // Align button row’s left/right edges with the board’s OUTER GRID BORDER.
            // SudokuResultView uses: pad = max(16dp, 4% of view width). Recompute that here:
            val gridPad = kotlin.math.max((boardSize * 0.04f).toInt(), 16.dp())
            buttonsRow.setPadding(gridPad, 0, gridPad, 0)
            buttonsRow.requestLayout()
        }
    }




    // Show / dismiss results overlay

    /*
    private fun showResults(boardBitmap: Bitmap?, digits81: IntArray?) {
        ensureResultsOverlay()

        resultsDigits = digits81
        resultsSudokuView?.setDigits(digits81 ?: IntArray(81))

        resultsRoot?.apply {
            visibility = View.VISIBLE
            alpha = 0f
            animate().alpha(1f).setDuration(180).start()
        }
        overlay.alpha = 1f
        previewView.alpha = 1f
        overlay.animate().alpha(0f).setDuration(120).start()
        previewView.animate().alpha(0f).setDuration(120).start()
    }

     */

    private fun showResults(boardBitmap: Bitmap?, digits81: IntArray?, confs81: FloatArray?) {
        ensureResultsOverlay()

        resultsDigits = digits81
        resultsConfidences = confs81

        // Feed the view
        if (digits81 != null && confs81 != null) {
            resultsSudokuView?.setDigitsAndConfidences(digits81, confs81)
        } else {
            resultsSudokuView?.setDigits(digits81 ?: IntArray(81))
        }

        // Make sure overlay is actually visible before animating
        resultsRoot?.apply {
            visibility = View.VISIBLE
            alpha = 0f
            animate().alpha(1f).setDuration(180).start()
        }

        // Fade out camera + HUD underneath
        overlay.animate().alpha(0f).setDuration(120).start()
        previewView.animate().alpha(0f).setDuration(120).start()
    }


    private fun onKeepResults() {
        val digits = resultsDigits ?: return
        val intent = Intent(this, ResultActivity::class.java).apply {
            putExtra("digits", digits)
        }
        startActivity(intent)
        //overridePendingTransition(0, 0)
    }

    private fun dismissResults(resumePreview: Boolean = true) {
        resultsRoot?.visibility = View.GONE
        lastBoardBitmap = null
        lastDigits81 = null

        if (resumePreview) {
            captureLocked = false
            resumeAnalyzer() // your existing method to un-pause and continue frames
        } else {
            // Keep it locked. You can add a “Save / Export” flow later.
        }
    }

    private fun resumeAnalyzer() {
        // Fully reset all scan gating
        locked = false
        handoffInProgress = false

        overlay.setLocked(false)

        // Put the HUD back into a neutral/ready state
        // (Going to L1 tends to be clearer than NONE after a green L3)
        changeGate(GateState.L1, "retake_resume")

        // If you keep any last-corners / last-ROI cached for drawing,
        // clear them here (only if you actually have these methods/fields):
        // overlay.setCorners(null)
        // overlay.setWarp(null)

        previewView.alpha = 1f
        overlay.alpha = 1f

        overlay.postInvalidateOnAnimation()
    }

    // Compose digits onto the board image
    private fun composeBoardWithDigits(board: Bitmap, digits81: IntArray?): Bitmap {
        val out = board.copy(Bitmap.Config.ARGB_8888, true)
        val c = Canvas(out)
        val n = 9
        val cell = out.width / n.toFloat()
        val textSize = cell * 0.6f

        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            this.textSize = textSize
            textAlign = Paint.Align.CENTER
            setShadowLayer(4f, 0f, 0f, Color.BLACK)
        }

        val fm = paint.fontMetrics
        val textOffset = (-(fm.ascent + fm.descent) / 2f)

        if (digits81 != null && digits81.size == 81) {
            for (r in 0 until n) {
                for (cIdx in 0 until n) {
                    val d = digits81[r * n + cIdx]
                    if (d in 1..9) {
                        val cx = cIdx * cell + cell / 2f
                        val cy = r * cell + cell / 2f + textOffset
                        c.drawText(d.toString(), cx, cy, paint)
                    }
                }
            }
        }
        return out
    }

    // Optional rough warp from ROI to square board
    // Uses the 4 outer corners. If the homography fails, we fall back to the raw ROI
    private fun roughWarpBoard(roiBmp: Bitmap,
                               tlx: Float, tly: Float, trx: Float, try_: Float,
                               brx: Float, bry: Float, blx: Float, bly: Float): Bitmap {
        return try {
            val src = Mat()
            Utils.bitmapToMat(roiBmp, src)  // RGBA
            val dst = Mat(GRID_SIZE, GRID_SIZE, CvType.CV_8UC4)

            val srcPts = MatOfPoint2f(
                Point(tlx.toDouble(), tly.toDouble()),
                Point(trx.toDouble(), try_.toDouble()),
                Point(brx.toDouble(), bry.toDouble()),
                Point(blx.toDouble(), bly.toDouble())
            )
            val dstPts = MatOfPoint2f(
                Point(0.0, 0.0),
                Point((GRID_SIZE - 1).toDouble(), 0.0),
                Point((GRID_SIZE - 1).toDouble(), (GRID_SIZE - 1).toDouble()),
                Point(0.0, (GRID_SIZE - 1).toDouble())
            )

            val H = Imgproc.getPerspectiveTransform(srcPts, dstPts)
            Imgproc.warpPerspective(src, dst, H, CvSize(GRID_SIZE.toDouble(), GRID_SIZE.toDouble()), Imgproc.INTER_LINEAR)

            val out = Bitmap.createBitmap(GRID_SIZE, GRID_SIZE, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(dst, out)

            src.release(); dst.release(); srcPts.release(); dstPts.release()
            out
        } catch (t: Throwable) {
            roiBmp
        }
    }

    private fun toFlat(grid: Array<IntArray>): IntArray =
        IntArray(81) { i -> grid[i / 9][i % 9] }

    private fun toFlat(grid: Array<FloatArray>): FloatArray =
        FloatArray(81) { i -> grid[i / 9][i % 9] }

    private fun toGrid(flat: IntArray): Array<IntArray> =
        Array(9) { r -> IntArray(9) { c -> flat[r * 9 + c] } }

    private fun toGrid(flat: FloatArray): Array<FloatArray> =
        Array(9) { r -> FloatArray(9) { c -> flat[r * 9 + c] } }

    // Build and bind CameraX Preview and ImageAnalysis. Analyzer pipeline:
    //  - Skip/throttle frames for consistent cadence.
    //  - Convert ImageProxy -> RGBA Bitmap.
    //  - Run Detector; choose the most-centered detection (if any).
    //  - Run CornerRefiner on that ROI; obtain corners + peaks.
    //  - Update OverlayView (source size, boxes, corners, crop ROI).
    //  - Enforce all gates (peaks≥thr, convexity, side ratios, area vs red box, aspect tolerance, jitter in 128-space, cyan-guard border). If a frame passes, add to the passing deque; once N frames pass, attempt rectification + classification.


    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()

            val preview = Preview.Builder()
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(UiSize(960, 720))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setOutputImageRotationEnabled(true)
                .setTargetRotation(previewView.display.rotation)
                .build()

            analysis.setAnalyzer(analyzerExecutor) { proxy ->
                try {
                    // Hold analyzer while rectifying/classifying or when locked.
                    if (locked || handoffInProgress) { proxy.close(); return@setAnalyzer }

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

                    // Choose the most centered detection (if any)
                    val cxImg = bmp.width / 2f
                    val cyImg = bmp.height / 2f
                    val picked = dets.minByOrNull { det ->
                        val dx = det.box.centerX() - cxImg
                        val dy = det.box.centerY() - cyImg
                        dx * dx + dy * dy
                    }
                    val toShow = if (picked != null) listOf(picked) else emptyList()

                    changeGate(if (picked != null) GateState.L1 else GateState.NONE,
                        if (picked != null) "detected" else "no_detection")

                    // 2) Intersections on chosen ROI (NO DUMP on normal frames)
                    val result = if (picked != null) {
                        try {
                            intersections.infer(
                                src = bmp,
                                roiSrc = picked.box,
                                padFrac = ROI_PAD_FRAC,
                                thrPred = INT_PEAK_THR,
                                topK = 140,
                                requireGridize = true,
                                dumpDebug = false,
                                dumpTag = null
                            )
                        } catch (t: Throwable) {
                            Log.w("MainActivity", "Intersections infer failed", t); null
                        }
                    } else null

                    // HUD update
                    runOnUiThread {
                        overlay.setSourceSize(bmp.width, bmp.height)
                        overlay.updateBoxes(toShow, HUD_DET_THRESH, HUD_MAX_DETS)
                        overlay.updateCornerCropRect(result?.expandedRoiSrc)
                        overlay.updateIntersections(result?.points)
                    }

                    // 3) Gating on intersections
                    if (picked != null && result != null && result.points.size >= 91) {
                        val roi = picked.box
                        val ptsSrc = result.points

                        // Jitter in 128-space using expanded ROI from intersections
                        val ex = result.expandedRoiSrc
                        fun to128(p: PointF): Pair<Float, Float> {
                            val x128 = (p.x - ex.left) * 128f / max(1, ex.width()).toFloat()
                            val y128 = (p.y - ex.top)  * 128f / max(1, ex.height()).toFloat()
                            return x128 to y128
                        }
                        val xs = FloatArray(ptsSrc.size) { i -> to128(ptsSrc[i]).first }
                        val ys = FloatArray(ptsSrc.size) { i -> to128(ptsSrc[i]).second }
                        val grid128 = Grid128(xs, ys)
                        val jitterOk = avgJitterPx128(grid128) <= MAX_JITTER_PX128

                        // Geometry from outer intersections
                        val tl = ptsSrc.minByOrNull { it.x + it.y }!!
                        val br = ptsSrc.maxByOrNull { it.x + it.y }!!
                        val tr = ptsSrc.minByOrNull { (bmp.width - it.x) + it.y }!!
                        val bl = ptsSrc.minByOrNull { it.x + (bmp.height - it.y) }!!

                        val geomOk = isConvexAndPositiveTLTRBRBL(tl, tr, br, bl) &&
                                sideLenRatio(tl, tr, br, bl) <= SIDE_RATIO_MAX

                        // Area/aspect vs detector box
                        val areaQuad = quadArea(tl, tr, br, bl)
                        val areaRed  = (roi.width() * roi.height()).coerceAtLeast(1f)
                        val areaRatio = areaQuad / areaRed
                        val areaOk = areaRatio >= AREA_RATIO_MIN && areaRatio <= AREA_RATIO_MAX

                        val aspectQuad = quadAspectApprox(tl, tr, br, bl)
                        val aspectRed  = aspect(roi)
                        val aspectOk = kotlin.math.abs(ln((aspectQuad / aspectRed).toDouble())) <= ln(ASPECT_TOL.toDouble())

                        // Cyan guard
                        val guardSrc = overlay.getGuardRectInSource()
                        val roiSrc = RectF(roi)
                        val tolSrc = (min(bmp.width, bmp.height) / 120f).coerceAtLeast(2f)
                        val guardOk = if (guardSrc != null) !touchesBorder(roiSrc, guardSrc, tolSrc) else true

                        val minPeak = result.scores.minOrNull() ?: 0f
                        val allHigh = minPeak >= INT_PEAK_THR

                        // Maintain jitter window
                        jitterHistory.addLast(grid128)
                        while (jitterHistory.size > JITTER_WINDOW) jitterHistory.removeFirst()

                        val good = allHigh && geomOk && areaOk && aspectOk && jitterOk && guardOk
                        if (good) {
                            passing.addLast(PassingFrame(ptsSrc, roi, minPeak, now, ex))
                            while (passing.size > STREAK_N) passing.removeFirst()
                            changeGate(GateState.L2, "100pts_good")

                            if (passing.size == STREAK_N) {
                                // *** LOCK: we will use THIS frame. ***
                                handoffInProgress = true
                                runOnUiThread { overlay.setLocked(true) }

                                // Re-run intersections WITH DUMP on the very same frame,
                                // so roi_model_in_128/heatmap/peaks/grid are from the locked frame.
                                val dumpRes = try {
                                    intersections.infer(
                                        src = bmp,
                                        roiSrc = picked.box,
                                        padFrac = ROI_PAD_FRAC,
                                        thrPred = INT_PEAK_THR,
                                        topK = 140,
                                        requireGridize = true,
                                        dumpDebug = true,
                                        dumpTag = "locked_${SystemClock.uptimeMillis()}"
                                    )
                                } catch (t: Throwable) {
                                    Log.w("MainActivity", "Intersections dump infer failed, falling back to non-dump result", t)
                                    result
                                } ?: result

                                // Clear streak buffer now that we’re proceeding
                                passing.clear()

                                // Handoff to rectification/classification using the dump result
                                attemptRectifyAndClassify(
                                    ptsSrc = dumpRes.points,
                                    detectorRoi = roi,
                                    srcBmp = bmp,
                                    expandedRoiSrc = dumpRes.expandedRoiSrc
                                )
                            }
                        } else {
                            passing.clear()
                            changeGate(if (picked != null) GateState.L1 else GateState.NONE, "intersections_not_good")
                        }
                    } else {
                        passing.clear()
                        jitterHistory.clear()
                        changeGate(if (picked != null) GateState.L1 else GateState.NONE, "insufficient_info")
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



    // Runs on a worker thread: expand+crop ROI; call Rectifier.process() to get tiles; push tiles into DigitClassifier; flatten to 81 digits and probabilities. If successful, set 'locked', flash/shutter, and navigate to results; otherwise keep scanning.
    // Runs on a worker thread: use the 10×10 intersection lattice to cut each cell
    // from its own quadrilateral, robust to bowed or bent grid lines.
    // Use the exact expanded ROI (expandedRoiSrc) that the intersections model used.
// This keeps point->ROI-local mapping aligned and fixes the cyan points offset.
    private fun attemptRectifyAndClassify(
        ptsSrc: List<PointF>,
        detectorRoi: RectF,     // kept for logging if you want
        srcBmp: Bitmap,
        expandedRoiSrc: Rect    // <-- the rectangle used by intersections.infer(...)
    ) {
        Thread {
            var err: String? = null
            var digitsFlat: IntArray? = null
            var probsFlat: FloatArray? = null
            var boardBmpOut: Bitmap? = null

            try {
                // --- Prepare debug folder ---
                val debugDir = getRectifyDebugDir()
                if (!debugDir.exists()) debugDir.mkdirs()
                clearDirectory(debugDir)

                // ---- Crop the exact expanded ROI used by the model ----
                val leftI   = expandedRoiSrc.left.coerceIn(0, srcBmp.width - 1)
                val topI    = expandedRoiSrc.top.coerceIn(0, srcBmp.height - 1)
                val rightI  = expandedRoiSrc.right.coerceIn(leftI + 1, srcBmp.width)
                val bottomI = expandedRoiSrc.bottom.coerceIn(topI + 1, srcBmp.height)

                val roiBitmap = Bitmap.createBitmap(srcBmp, leftI, topI, rightI - leftI, bottomI - topI)
                saveBitmapPng(java.io.File(debugDir, "roi.png"), roiBitmap)

                // Map detected points into THIS ROI’s local coordinates
                fun toLocal(p: PointF) = PointF(p.x - leftI, p.y - topI)
                val ptsLocal = ptsSrc.map { toLocal(it) }

                // Order/repair into a strict 10×10 lattice
                val grid10 = orderPointsInto10x10(ptsLocal)
                    ?: throw IllegalStateException("Could not form 10×10 grid (points=${ptsLocal.size})")

                // Save ROI with overlaid (local) points for sanity check
                saveBitmapPng(
                    java.io.File(debugDir, "roi_points.png"),
                    drawPointsOverlayOnBitmap(roiBitmap, grid10.flatten())
                )

                // Cut each cell by its own quad
                val tiles = Array(9) { rIdx ->
                    Array(9) { cIdx ->
                        val tl = grid10[rIdx][cIdx]
                        val tr = grid10[rIdx][cIdx + 1]
                        val br = grid10[rIdx + 1][cIdx + 1]
                        val bl = grid10[rIdx + 1][cIdx]
                        warpQuadToSquare(roiBitmap, tl, tr, br, bl, CELL_SIZE, CELL_SIZE)
                    }
                }

                // Debug: save cell tiles r1c1..r9c9
                for (rr in 0 until 9) for (cc in 0 until 9) {
                    saveBitmapPng(java.io.File(debugDir, "cell_r${rr + 1}c${cc + 1}.png"), tiles[rr][cc])
                }

                // Build a mosaic preview
                val mosaic = Bitmap.createBitmap(GRID_SIZE, GRID_SIZE, Bitmap.Config.ARGB_8888)
                val can = Canvas(mosaic)
                for (rr in 0 until 9) for (cc in 0 until 9) {
                    can.drawBitmap(tiles[rr][cc], (cc * CELL_SIZE).toFloat(), (rr * CELL_SIZE).toFloat(), null)
                }
                boardBmpOut = mosaic

                // Classify
                ensureDigitClassifier()
                val (digits, confs) = digitClassifier!!.classifyTiles(tiles)
                digitsFlat = IntArray(81) { i -> digits[i / 9][i % 9] }
                probsFlat  = FloatArray(81) { i -> confs[i / 9][i % 9] }

                // Lock & show
                locked = true
                runOnUiThread {
                    overlay.playShutter(null)
                    overlay.setLocked(true)
                    changeGate(GateState.L3, "locked")
                }
            } catch (t: Throwable) {
                err = t.message ?: "$t"
                Log.e("MainActivity", "attemptRectifyAndClassify (aligned ROI) failed", t)
            } finally {
                handoffInProgress = false
                if (locked && boardBmpOut != null) {
                    runOnUiThread {
                        showResults(
                            boardBmpOut!!,
                            digitsFlat ?: IntArray(81) { 0 },
                            probsFlat  ?: FloatArray(81) { 1f }
                        )
                    }
                } else {
                    runOnUiThread {
                        overlay.setLocked(false)
                        changeGate(GateState.L1, "resume_after_fail")
                        if (err != null) {
                            Toast.makeText(this, "Rectify failed: $err", Toast.LENGTH_SHORT).show()
                        }
                    }
                }
            }
        }.start()
    }

    // Lazy-init the TFLite digit classifier; avoids upfront load on app start.
    private fun ensureDigitClassifier() {
        if (digitClassifier == null) {
            val dc = DigitClassifier(this, "models/digit_cnn_fp32.tflite", 2).apply {
                dumpParity = false
                dumpSessionTag = "crop_test"

                // Parity capture settings
                dumpWhitelist = emptySet()     // capture all 81 cells
                innerCrop = 0.92f              // ← ensures same crop as Python
            }
            digitClassifier = dc
        } else {
            // Update existing classifier flags in case of reuse
            digitClassifier?.apply {
                dumpParity = false // set it to true when you want to DEBUG with parity pack
                dumpSessionTag = "crop_test"
                dumpWhitelist = emptySet()
                innerCrop = 0.92f
            }
        }
    }

    // Called once when we decide the grid is good & stable.
    // Deprecated/alternate capture path that does rectification/classification after 'locked'. Shows shutter and results on success; otherwise toasts an error.
    private fun onLockedGridCaptured(lockedRoiBitmap: Bitmap) {
        // MM1 TEMP: Layer 3 (rectify/classify/navigate) disabled.
        Logx.d("Gate", "skip" to "L3_disabled_MM1", "note" to "stubbing onLockedGridCaptured")
    }


    // --- Debug I/O helpers ------------------------------------------------------


    // Where to write debug images (VISIBLE in Device File Explorer for a debuggable build)
    private fun getDebugDir(): java.io.File {
        val dir = java.io.File(filesDir, "rectify_debug")
        if (!dir.exists()) dir.mkdirs()
        return dir
    }

    private fun saveBitmapDebug(bmp: Bitmap, name: String) {
        try {
            val dir = getDebugDir()
            val file = java.io.File(dir, name)
            java.io.FileOutputStream(file).use { out ->
                bmp.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
            Log.i("RectifyDebug", "Saved: ${file.absolutePath}")
        } catch (t: Throwable) {
            Log.e("RectifyDebug", "saveBitmapDebug failed for $name", t)
        }
    }


    // Use INTERNAL app storage so it’s visible in Device File Explorer:
// Device File Explorer → data → data → com.contextionary.sudoku → files → rectify_debug
    private fun getRectifyDebugDir(): java.io.File {
        val dir = java.io.File(filesDir, "rectify_debug")
        if (!dir.exists()) dir.mkdirs()
        return dir
    }

    private fun clearDirectory(dir: java.io.File) {
        if (!dir.exists()) return
        dir.listFiles()?.forEach { f ->
            try {
                if (f.isDirectory) {
                    // KEEP intersections parity packs
                    if (!f.name.startsWith("intersections_")) {
                        f.deleteRecursively()
                    }
                } else {
                    // Remove loose files (roi.png, cell_*.png, etc.)
                    f.delete()
                }
            } catch (_: Throwable) { /* ignore */ }
        }
    }

    private fun saveBitmapPng(file: java.io.File, bmp: Bitmap) {
        try {
            file.parentFile?.mkdirs()
            java.io.FileOutputStream(file).use { out ->
                bmp.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
        } catch (t: Throwable) {
            Log.w("MainActivity", "saveBitmapPng failed: ${file.absolutePath}", t)
        }
    }

    private fun drawPointsOverlayOnBitmap(base: Bitmap, pointsLocal: List<PointF>): Bitmap {
        val out = base.copy(Bitmap.Config.ARGB_8888, true)
        val c = Canvas(out)
        val p = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            color = Color.CYAN
        }
        val r = (2.0f * resources.displayMetrics.density).coerceAtLeast(2f) // ~2dp
        pointsLocal.forEach { pt ->
            c.drawCircle(pt.x, pt.y, r, p)
        }
        return out
    }


    // ===== Rectify debug saving =====


    private fun wipeDir(dir: java.io.File) {
        if (!dir.exists()) return
        dir.listFiles()?.forEach {
            if (it.isDirectory) wipeDir(it) else runCatching { it.delete() }
        }
    }


    /** Draws the given points onto a copy of 'base' and returns the annotated bitmap. */
    private fun drawPointsOverlay(
        base: android.graphics.Bitmap,
        points: List<android.graphics.PointF>,
        radiusPx: Float = 3f
    ): android.graphics.Bitmap {
        val out = base.copy(android.graphics.Bitmap.Config.ARGB_8888, true)
        val c = android.graphics.Canvas(out)
        val p = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
            style = android.graphics.Paint.Style.FILL
            color = android.graphics.Color.CYAN
        }
        for (pt in points) c.drawCircle(pt.x, pt.y, radiusPx, p)
        return out
    }

    // ===== Geometry helpers (PointF versions) =====

    private fun quadArea(tl: PointF, tr: PointF, br: PointF, bl: PointF): Float {
        fun cross(ax: Float, ay: Float, bx: Float, by: Float) = ax * by - ay * bx
        val sum =
            cross(tl.x, tl.y, tr.x, tr.y) +
                    cross(tr.x, tr.y, br.x, br.y) +
                    cross(br.x, br.y, bl.x, bl.y) +
                    cross(bl.x, bl.y, tl.x, tl.y)
        return kotlin.math.abs(sum) * 0.5f
    }

    private fun isConvexAndPositiveTLTRBRBL(
        tl: PointF, tr: PointF, br: PointF, bl: PointF
    ): Boolean {
        fun cross(ax: Float, ay: Float, bx: Float, by: Float) = ax * by - ay * bx
        val z1 = cross(tr.x - tl.x, tr.y - tl.y, br.x - tl.x, br.y - tl.y)
        val z2 = cross(br.x - tr.x, br.y - tr.y, bl.x - tr.x, bl.y - tr.y)
        val z3 = cross(bl.x - br.x, bl.y - br.y, tl.x - br.x, tl.y - br.y)
        val z4 = cross(tl.x - bl.x, tl.y - bl.y, tr.x - bl.x, tr.y - bl.y)
        val allPos = z1 > 0 && z2 > 0 && z3 > 0 && z4 > 0
        val allNeg = z1 < 0 && z2 < 0 && z3 < 0 && z4 < 0
        val area = quadArea(tl, tr, br, bl)
        return (allPos || allNeg) && area > 1e-3f
    }

    private fun sideLenRatio(tl: PointF, tr: PointF, br: PointF, bl: PointF): Float {
        fun d(a: PointF, b: PointF): Float {
            val dx = a.x - b.x; val dy = a.y - b.y
            return kotlin.math.sqrt(dx*dx + dy*dy)
        }
        val s1 = d(tl, tr); val s2 = d(tr, br); val s3 = d(br, bl); val s4 = d(bl, tl)
        val mx = maxOf(s1, s2, s3, s4)
        val mn = minOf(s1, s2, s3, s4)
        return if (mn <= 1e-6f) Float.POSITIVE_INFINITY else mx / mn
    }

    private fun quadAspectApprox(tl: PointF, tr: PointF, br: PointF, bl: PointF): Float {
        fun d(a: PointF, b: PointF): Float {
            val dx = a.x - b.x; val dy = a.y - b.y
            return kotlin.math.sqrt(dx*dx + dy*dy)
        }
        val top = d(tl, tr); val bottom = d(bl, br)
        val left = d(tl, bl); val right = d(tr, br)
        val w = (top + bottom) * 0.5f
        val h = (left + right) * 0.5f
        return if (h <= 1e-6f) Float.POSITIVE_INFINITY else w / h
    }

    private fun aspect(r: RectF): Float {
        val w = r.width().coerceAtLeast(1f)
        val h = r.height().coerceAtLeast(1f)
        return w / h
    }


    /**
     * Convert the raw intersection detections into a strict 10×10 grid [row][col].
     * Strategy:
     *  - Cluster into 10 horizontal bands by Y using the 9 largest gaps.
     *  - Within each band, sort by X; if the band has != 10 points, linearly
     *    interpolate to produce 10 evenly spaced points between the band’s min/max.
     *  - Finally, column-align by averaging X across rows to reduce jitter.
     *
     * Returns null if we really can’t form a proper grid.
     */
    private fun orderPointsInto10x10(points: List<PointF>): Array<Array<PointF>>? {
        if (points.size < 90) return null

        // 1) Sort by Y
        val byY = points.sortedBy { it.y }

        // 2) Split into 10 rows by cutting at the 9 largest Y gaps
        val gaps = mutableListOf<Pair<Int, Float>>() // (index, gapValue) between byY[i] and byY[i+1]
        for (i in 0 until byY.size - 1) {
            gaps += i to (byY[i + 1].y - byY[i].y)
        }
        val cutIndices = gaps.sortedByDescending { it.second }.take(9).map { it.first }.sorted()
        val rows = ArrayList<List<PointF>>(10)
        var start = 0
        for (cut in cutIndices) {
            rows += byY.subList(start, cut + 1)
            start = cut + 1
        }
        rows += byY.subList(start, byY.size)
        if (rows.size != 10) return null

        // 3) Within each row, sort by X and expand/shrink to exactly 10 points via interpolation
        val row10 = Array(10) { Array(10) { PointF() } }
        for (r in 0 until 10) {
            val row = rows[r].sortedBy { it.x }
            row10[r] = interpolateRowToTen(row)
        }

        // 4) Column alignment pass: average X per column & nudge points slightly toward that mean
        for (c in 0 until 10) {
            var sumX = 0f
            for (r in 0 until 10) sumX += row10[r][c].x
            val meanX = sumX / 10f
            for (r in 0 until 10) {
                val p = row10[r][c]
                // Gentle pull to column mean to reduce jitter while keeping row order
                row10[r][c] = PointF((p.x * 0.8f + meanX * 0.2f), p.y)
            }
        }
        return row10
    }

    /** Ensure a row contains exactly 10 points by linear interpolation along X. */
    private fun interpolateRowToTen(sortedRow: List<PointF>): Array<PointF> {
        val out = Array(10) { PointF() }
        if (sortedRow.isEmpty()) {
            // Fallback: make a dummy row—caller will likely fail geometry later.
            for (i in 0 until 10) out[i] = PointF(i.toFloat(), 0f)
            return out
        }
        val left = sortedRow.first().x
        val right = sortedRow.last().x
        if (right <= left + 1e-3f) {
            // Degenerate; collapse to left
            for (i in 0 until 10) out[i] = PointF(left, sortedRow.first().y)
            return out
        }

        // Build a piecewise-linear map of X→Y using the existing points,
        // then sample it at 10 evenly spaced Xs between [left, right].
        val xs = sortedRow.map { it.x }.toFloatArray()
        val ys = sortedRow.map { it.y }.toFloatArray()

        fun yAt(xq: Float): Float {
            // find bracketing segment
            var i = xs.indexOfLast { it <= xq }
            if (i < 0) return ys.first()
            if (i >= xs.size - 1) return ys.last()
            val x0 = xs[i]; val x1 = xs[i + 1]
            val y0 = ys[i]; val y1 = ys[i + 1]
            val t = ((xq - x0) / (x1 - x0)).coerceIn(0f, 1f)
            return y0 + t * (y1 - y0)
        }

        for (i in 0 until 10) {
            val xq = left + (right - left) * (i / 9f)
            out[i] = PointF(xq, yAt(xq))
        }
        return out
    }

    /** Warp a source-space quadrilateral to a WxH square bitmap (OpenCV). */
    private fun warpQuadToSquare(
        src: Bitmap,
        tl: PointF, tr: PointF, br: PointF, bl: PointF,
        w: Int, h: Int
    ): Bitmap {
        val srcMat = Mat()
        Utils.bitmapToMat(src, srcMat) // RGBA

        val dstMat = Mat(h, w, CvType.CV_8UC4)

        val srcPts = MatOfPoint2f(
            org.opencv.core.Point(tl.x.toDouble(), tl.y.toDouble()),
            org.opencv.core.Point(tr.x.toDouble(), tr.y.toDouble()),
            org.opencv.core.Point(br.x.toDouble(), br.y.toDouble()),
            org.opencv.core.Point(bl.x.toDouble(), bl.y.toDouble())
        )
        val dstPts = MatOfPoint2f(
            org.opencv.core.Point(0.0, 0.0),
            org.opencv.core.Point((w - 1).toDouble(), 0.0),
            org.opencv.core.Point((w - 1).toDouble(), (h - 1).toDouble()),
            org.opencv.core.Point(0.0, (h - 1).toDouble())
        )

        val H = Imgproc.getPerspectiveTransform(srcPts, dstPts)
        Imgproc.warpPerspective(srcMat, dstMat, H, CvSize(w.toDouble(), h.toDouble()), Imgproc.INTER_LINEAR)

        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(dstMat, out)

        srcMat.release(); dstMat.release(); srcPts.release(); dstPts.release()
        return out
    }

    // ==== View-space mapping & cyan guard helpers ====

    // Compute view mapping used if you ever need to reproduce guard mapping math here.
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

    // Map bitmap-space RectF to view-space RectF given scale and offsets.
    private fun mapRectToView(r: RectF, s: Float, offX: Float, offY: Float): RectF {
        return RectF(
            offX + r.left * s,
            offY + r.top * s,
            offX + r.right * s,
            offY + r.bottom * s
        )
    }

    // Compute the centered cyan square guide for a mapped bitmap rect.
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
    // Determine if the detection box touches the guard border within a tolerance (source space).
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

        return touchLeft || touchRight || touchBottom || touchTop
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