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

import java.util.concurrent.atomic.AtomicBoolean


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

    private val gate = GateController()


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
        val ptsSrc: List<PointF>,
        val roi: RectF,
        val minPeak: Float,
        val tsMs: Long,
        val expandedRoi: Rect,
        val roiBmp: Bitmap,            // cropped ROI bitmap for this frame (used for debug overlay)
        val jitterPx128: Float,        // <-- non-default must come before defaulted params
        var score: ScoreBreakdown? = null   // optional cache of the computed breakdown
    )

    // Detailed per-frame scoring used for CSV debug.
    private data class ScoreBreakdown(
        val total: Float,
        val confPct: Float,
        val lineStraightness: Float,
        val orthogonality: Float,
        val cellUniformity: Float,
        val clearance: Float,
        val jitterScore: Float
    )

    private data class GateSnapshot(
        val hasDetectorLock: Boolean,
        val gridizedOk: Boolean,
        val validPoints: Int,
        val jitterPx128: Float,
        val rectifyOk: Boolean,
        val avgConf: Float,
        val lowConfCells: Int
    )




    // -------------------------------------------------------------------------
    // Drop-in replacement: GateController
    //  - Amber only after "dots appeared" dwell (RED_TO_AMBER_MS)
    //  - Gentle amber-loss grace (AMBER_LOSS_GRACE_MS)
    //  - L3 (green) is set externally at lock time (provisional green)
    //  - Demote from L3 if post checks fail (GREEN_FAIL_GRACE_MS)
    // -------------------------------------------------------------------------
    private class GateController {
        var state: GateState = GateState.NONE; private set
        private var enteredAt = System.currentTimeMillis()

        // dwell for "dots visible" before we can enter Amber
        private var firstSeenGridizedAt: Long? = null
        // grace before dropping Amber when dots disappear
        private var amberLossSince: Long? = null
        // grace before dropping Green if post checks fail
        private var greenFailSince: Long? = null

        // local tunables (keep decoupled from companion constants)
        private val RED_TO_AMBER_MS       = 300L   // dwell after dots appear
        private val AMBER_LOSS_GRACE_MS   = 180L   // avoid flicker
        private val GREEN_FAIL_GRACE_MS   = 120L   // gentle demotion

        private fun now() = System.currentTimeMillis()

        fun update(s: GateSnapshot): GateState {
            val t = now()
            val prev = state

            // NONE: idle until we have any detector lock
            if (state == GateState.NONE) {
                if (s.hasDetectorLock) {
                    state = GateState.L1; enteredAt = t
                }
                return state
            }

            // keep internal timers in sync with gridized visibility
            if (s.gridizedOk) {
                if (firstSeenGridizedAt == null) firstSeenGridizedAt = t
                amberLossSince = null
            } else {
                firstSeenGridizedAt = null
                if (state == GateState.L2) {
                    if (amberLossSince == null) amberLossSince = t
                } else {
                    amberLossSince = null
                }
            }

            when (state) {
                GateState.L1 -> {
                    // promote only if dots have been visible long enough
                    val dwell = firstSeenGridizedAt?.let { t - it } ?: 0L
                    if (dwell >= RED_TO_AMBER_MS && s.hasDetectorLock && s.validPoints >= 90) {
                        state = GateState.L2; enteredAt = t
                    }
                    // lose detector completely → back to NONE
                    if (!s.hasDetectorLock) {
                        state = GateState.NONE; enteredAt = t
                        firstSeenGridizedAt = null
                        amberLossSince = null
                        greenFailSince = null
                    }
                }
                GateState.L2 -> {
                    // demote if dots vanished and grace window elapsed
                    if (!s.gridizedOk) {
                        val loss = amberLossSince?.let { t - it } ?: 0L
                        if (loss >= AMBER_LOSS_GRACE_MS) {
                            state = if (s.hasDetectorLock) GateState.L1 else GateState.NONE
                            enteredAt = t
                            firstSeenGridizedAt = null
                            amberLossSince = null
                        }
                    }
                    // promotion to L3 is driven externally at lock time
                }
                GateState.L3 -> {
                    // in green, if post checks fail, wait a bit then drop to Amber/Red
                    val postOk = (s.rectifyOk && s.avgConf >= 0.75f && s.lowConfCells <= 6)
                    if (!postOk) {
                        if (greenFailSince == null) greenFailSince = t
                        if ((t - greenFailSince!!) >= GREEN_FAIL_GRACE_MS) {
                            state = if (s.gridizedOk && s.hasDetectorLock) GateState.L2 else if (s.hasDetectorLock) GateState.L1 else GateState.NONE
                            enteredAt = t
                            greenFailSince = null
                        }
                    } else {
                        greenFailSince = null
                    }
                }
                else -> { /* NONE handled above */ }
            }
            return state
        }
    }




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
        private const val STREAK_N = 4
        private const val SHOW_CROP_OVERLAY = false
        private const val ROI_PAD_FRAC = 0.08f   // 8% on each side (tweak: 0.06–0.12)

        // === L3 (rectify/classify) tunables ===
        private const val GRID_SIZE = 576           // 9 * 64, square warp target
        private const val CELL_SIZE = 64
        private const val MIN_RECT_PASS_AVG_CONF = 0.75f
        private const val MAX_LOWCONF = 6           // how many cells may be low-confidence
        private const val LOWCONF_THR = 0.60f       // what "low" means, per cell
        //private const val GRID_SIZE = 450  // square pixels for our “rough” board render

        private const val DUMP_LOCKED_INTERSECTIONS = false

        // TRAFFIC-LIGHT SIGNALING

        private const val RED_TO_AMBER_MS   = 150L
        private const val AMBER_TO_GREEN_MS = 250L
        private const val DEMOTE_GRACE_MS   = 200L

        private const val MIN_VALID_PTS         = 90         // intersections ≥90/100
        // private const val MAX_JITTER_PX128      = 7f         // already used in your flow
        private const val MIN_AVG_CELL_CONF     = 0.75f
        private const val LOWCONF_CELL_THR      = 0.60f
        private const val MAX_LOWCONF_CELLS     = 6
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

        overlay.showCornerDots = false
        overlay.showBoxLabels = false
        overlay.showHudText   = false
        overlay.showCropRect = false

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
        overlay.showIntersections = false


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







    // -------------------------------------------------------------------------
    // Drop-in replacement: startCamera()
    //  - Amber after "dots appear" dwell (delegated to GateController via gridizedOk)
    //  - Best-of-N: score the last 4 passing frames; lock the best (uses current bmp for work)
    //  - Provisional Green at lock; heavy work happens in attemptRectifyAndClassify()
    //  - Demotions: NONE when no pick; L1 when intersections/gates break
    // -------------------------------------------------------------------------
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

                    // 1) Detection
                    val dets = detector.infer(bmp, scoreThresh = HUD_DET_THRESH, maxDets = HUD_MAX_DETS)
                    val cxImg = bmp.width / 2f
                    val cyImg = bmp.height / 2f
                    val picked = dets.minByOrNull { det ->
                        val dx = det.box.centerX() - cxImg
                        val dy = det.box.centerY() - cyImg
                        dx * dx + dy * dy
                    }

                    // HUD: boxes
                    runOnUiThread {
                        overlay.setSourceSize(bmp.width, bmp.height)
                        overlay.updateBoxes(if (picked != null) listOf(picked) else emptyList(), HUD_DET_THRESH, HUD_MAX_DETS)
                    }

                    if (picked == null) {
                        // No detection → NONE, clear histories
                        passing.clear()
                        jitterHistory.clear()
                        runOnUiThread {
                            overlay.updateCornerCropRect(null)
                            overlay.updateIntersections(null)
                        }
                        changeGate(GateState.NONE, "no_detection")
                        proxy.close(); return@setAnalyzer
                    }

                    // 2) Intersections (no dump on live frames)
                    val inter = try {
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

                    // HUD: crop overlay + dots
                    runOnUiThread {
                        overlay.updateCornerCropRect(inter?.expandedRoiSrc)
                        overlay.updateIntersections(inter?.points)
                    }

                    // 3) Gating snapshot for SM (pre-rectify)
                    val haveGrid = (inter != null && inter.points.size >= 90)
                    val ptsSrc = inter?.points ?: emptyList()

                    // Jitter in 128-space
                    val ex = inter?.expandedRoiSrc
                    val jitterPx = if (haveGrid && ex != null) {
                        val xs = FloatArray(ptsSrc.size) { i -> ((ptsSrc[i].x - ex.left) * 128f / max(1, ex.width())) }
                        val ys = FloatArray(ptsSrc.size) { i -> ((ptsSrc[i].y - ex.top)  * 128f / max(1, ex.height())) }
                        val grid128 = Grid128(xs, ys)
                        val v = avgJitterPx128(grid128)
                        jitterHistory.addLast(grid128)
                        while (jitterHistory.size > JITTER_WINDOW) jitterHistory.removeFirst()
                        v
                    } else {
                        jitterHistory.clear(); 999f
                    }

                    val jitterOk = jitterPx <= 7f

                    // Geometry from outer intersections if we have them
                    val geomOk = if (haveGrid) run {
                        val tl = ptsSrc.minByOrNull { it.x + it.y }!!
                        val br = ptsSrc.maxByOrNull { it.x + it.y }!!
                        val tr = ptsSrc.minByOrNull { (bmp.width - it.x) + it.y }!!
                        val bl = ptsSrc.minByOrNull { it.x + (bmp.height - it.y) }!!
                        val roi = picked.box
                        val areaQuad = quadArea(tl, tr, br, bl)
                        val areaRed  = (roi.width() * roi.height()).coerceAtLeast(1f)
                        val areaRatio = areaQuad / areaRed
                        val areaOk = areaRatio >= AREA_RATIO_MIN && areaRatio <= AREA_RATIO_MAX
                        val aspectQuad = quadAspectApprox(tl, tr, br, bl)
                        val aspectRed  = aspect(RectF(roi))
                        val aspectOk = kotlin.math.abs(ln((aspectQuad / aspectRed).toDouble())) <= ln(ASPECT_TOL.toDouble())
                        val shapeOk = isConvexAndPositiveTLTRBRBL(tl, tr, br, bl) && sideLenRatio(tl, tr, br, bl) <= SIDE_RATIO_MAX
                        areaOk && aspectOk && shapeOk
                    } else false

                    // Cyan guard
                    val guardSrc = overlay.getGuardRectInSource()
                    val roiSrc = RectF(picked.box)
                    val tolSrc = (min(bmp.width, bmp.height) / 120f).coerceAtLeast(2f)
                    val guardOk = if (guardSrc != null) !touchesBorder(roiSrc, guardSrc, tolSrc) else true

                    val minPeak = inter?.scores?.minOrNull() ?: 0f
                    val allHigh = haveGrid && (minPeak >= INT_PEAK_THR)

                    // Feed SM with pre-rectify snapshot (rectifyOk=false)
                    val pre = GateSnapshot(
                        hasDetectorLock = true,
                        gridizedOk      = haveGrid,
                        validPoints     = inter?.points?.size ?: 0,
                        jitterPx128     = jitterPx,
                        rectifyOk       = false,
                        avgConf         = 0f,
                        lowConfCells    = Int.MAX_VALUE
                    )
                    val sm = gate.update(pre)
                    if (sm != gateState) changeGate(sm, "preRectify")

                    // Build "good" predicate (for streak buffer)
                    val good = haveGrid && allHigh && geomOk && guardOk && jitterOk

                    if (good && inter != null) {
                        // Capture the exact ROI bitmap now (for per-frame overlay saving later)
                        val leftI   = inter.expandedRoiSrc.left.coerceIn(0, bmp.width - 1)
                        val topI    = inter.expandedRoiSrc.top.coerceIn(0, bmp.height - 1)
                        val rightI  = inter.expandedRoiSrc.right.coerceIn(leftI + 1, bmp.width)
                        val bottomI = inter.expandedRoiSrc.bottom.coerceIn(topI + 1, bmp.height)
                        val roiBitmap = Bitmap.createBitmap(bmp, leftI, topI, rightI - leftI, bottomI - topI)

                        // Maintain a buffer of last N good frames
                        passing.addLast(
                            PassingFrame(
                                ptsSrc      = inter.points,
                                roi         = RectF(picked.box),
                                minPeak     = minPeak,
                                tsMs        = now,
                                expandedRoi = inter.expandedRoiSrc,
                                roiBmp      = roiBitmap,
                                jitterPx128 = jitterPx         // <-- store per-frame jitter
                            )
                        )
                        while (passing.size > STREAK_N) passing.removeFirst()

                        // Stay visually in Amber while collecting
                        if (gateState != GateState.L2) changeGate(GateState.L2, "good_frame")

                        if (passing.size >= STREAK_N) {
                            // --- Score all N; save images + CSV for audit; pick best ---
                            val guard = overlay.getGuardRectInSource()
                            val frames = passing.toList()

                            var bestIdx = -1
                            var bestScore = Float.NEGATIVE_INFINITY

                            for (i in frames.indices) {
                                val pf = frames[i]
                                val bd = computeFrameScore(
                                    ptsSrc       = pf.ptsSrc,
                                    ex           = pf.expandedRoi,
                                    roi          = pf.roi,
                                    jitterPx128  = pf.jitterPx128,   // <-- use the frame’s stored jitter
                                    minPeak      = pf.minPeak,
                                    guardRect    = guard,
                                    imgW         = bmp.width,
                                    imgH         = bmp.height
                                )
                                pf.score = bd
                                if (bd.total > bestScore) { bestScore = bd.total; bestIdx = i }
                            }

                            // Save debug pack before clearing
                            val debugRoot = getRectifyDebugDir()
                            saveBestOfNDebugPack(debugRoot, frames, bestIdx.coerceAtLeast(0))

                            // Recycle bitmaps & clear
                            frames.forEach { f -> try { f.roiBmp.recycle() } catch (_: Throwable) {} }
                            passing.clear()

                            // Provisional green immediately
                            handoffInProgress = true
                            changeGate(GateState.L3, "lock_provisional_green")
                            runOnUiThread { overlay.setLocked(true) }

                            // Re-run intersections with dump on the same live frame
                            val dumpRes = try {
                                intersections.infer(
                                    src = bmp,
                                    roiSrc = picked.box,
                                    padFrac = ROI_PAD_FRAC,
                                    thrPred = INT_PEAK_THR,
                                    topK = 140,
                                    requireGridize = true,
                                    dumpDebug = DUMP_LOCKED_INTERSECTIONS,
                                    dumpTag = if (DUMP_LOCKED_INTERSECTIONS) "locked_${SystemClock.uptimeMillis()}" else null
                                )
                            } catch (t: Throwable) {
                                Log.w("MainActivity", "Intersections dump infer failed; using current inter", t)
                                inter
                            } ?: inter

                            attemptRectifyAndClassify(
                                ptsSrc = dumpRes.points,
                                detectorRoi = RectF(picked.box),
                                srcBmp = bmp,
                                expandedRoiSrc = dumpRes.expandedRoiSrc
                            )
                        }
                    } else {
                        // bad frame → drop to L1 (or stay None if no lock)
                        passing.clear()
                        if (picked != null) {
                            changeGate(GateState.L1, "not_good")
                        }
                    }
                } catch (t: Throwable) {
                    Log.e("Detector", "Analyzer error on frame $frameIndex", t)
                    runOnUiThread {
                        overlay.setSourceSize(previewView.width, previewView.height)
                        overlay.updateBoxes(emptyList(), HUD_DET_THRESH, 0)
                        overlay.updateCornerCropRect(null)
                        overlay.updateIntersections(null)
                        overlay.setLocked(false)
                    }
                    passing.clear()
                    jitterHistory.clear()
                    changeGate(GateState.NONE, "exception")
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



    // -------------------------------------------------------------------------
// Save Best-of-N debug pack:
//  - frame_1_points.png ... frame_N_points.png (points drawn on ROI)
//  - bestofN.csv with per-frame scoring breakdown and metadata
// -------------------------------------------------------------------------
    private fun saveBestOfNDebugPack(
        parentDir: java.io.File,
        frames: List<PassingFrame>,
        chosenIdx: Int
    ) {
        runCatching {
            val ts = java.text.SimpleDateFormat("yyyyMMdd_HHmmss", java.util.Locale.US)
                .format(java.util.Date())

            val dir = java.io.File(
                parentDir,
                "bestofN${STREAK_N}_${ts}_${android.os.SystemClock.uptimeMillis()}"
            )
            if (!dir.exists()) dir.mkdirs()

            // locale-stable formatters
            fun f6(x: Float) = String.format(java.util.Locale.US, "%.6f", x)
            fun f2(x: Float) = String.format(java.util.Locale.US, "%.2f", x)

            // --- helpers (local to this method) --------------------------------

            // Order 100 points into a 10×10 grid (row-major) by banding on Y then sorting by X.
            fun orderIntoGrid10(ptsLocal: List<PointF>): Array<Array<PointF>> {
                require(ptsLocal.size >= 100) { "Need at least 100 points" }

                // sort by Y asc, then partition into 10 consecutive bands of ~10
                val sorted = ptsLocal.sortedBy { it.y }
                val rows = Array(10) { ArrayList<PointF>(10) }
                for (r in 0 until 10) {
                    val start = (r * sorted.size) / 10
                    val end = ((r + 1) * sorted.size) / 10
                    val band = sorted.subList(start, end).sortedBy { it.x }
                    // keep exactly 10 by trimming or padding with nearest endpoints
                    val row = when {
                        band.size == 10 -> band
                        band.size > 10  -> band.take(10)
                        else -> {
                            val need = 10 - band.size
                            val padLeft  = generateSequence { band.first() }.take(need / 2).toList()
                            val padRight = generateSequence { band.last()  }.take(need - need / 2).toList()
                            (padLeft + band + padRight)
                        }
                    }
                    rows[r].addAll(row)
                }
                return Array(10) { r -> Array(10) { c -> rows[r][c] } }
            }

            // Per row, resample to 10 evenly spaced X positions from minX..maxX.
            // Y is linearly interpolated between the row endpoints.
            fun resampleRowsEvenX(grid: Array<Array<PointF>>): Array<Array<PointF>> {
                val out = Array(10) { Array(10) { PointF() } }
                for (r in 0 until 10) {
                    val row = grid[r]
                    val left  = row.first()
                    val right = row.last()
                    val minX = left.x
                    val maxX = right.x
                    val dx = (maxX - minX).coerceAtLeast(1e-3f)

                    for (c in 0 until 10) {
                        val t = c / 9f  // 0..1
                        val x = minX + t * dx
                        val y = left.y + t * (right.y - left.y)
                        out[r][c] = PointF(x, y)
                    }
                }
                return out
            }

            // Column-mean nudge: pull each point's X a fraction toward its column mean.
            fun nudgeColumnsToMeans(grid: Array<Array<PointF>>, frac: Float = 0.20f): Array<Array<PointF>> {
                val out = Array(10) { r -> Array(10) { c -> PointF(grid[r][c].x, grid[r][c].y) } }
                for (c in 0 until 10) {
                    var sumX = 0f
                    for (r in 0 until 10) sumX += grid[r][c].x
                    val meanX = sumX / 10f
                    for (r in 0 until 10) {
                        val p = out[r][c]
                        p.x = p.x + frac * (meanX - p.x)
                    }
                }
                return out
            }

            // Render the processed lattice onto a copy of roiBmp.
            fun drawProcessedGridOverlay(base: Bitmap, grid: Array<Array<PointF>>): Bitmap {
                val bmp = base.copy(Bitmap.Config.ARGB_8888, true)
                val canvas = Canvas(bmp)

                val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    color = Color.WHITE
                    style = Paint.Style.STROKE
                    strokeWidth = (bmp.width.coerceAtMost(bmp.height) / 360f).coerceAtLeast(1.5f)
                }
                val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    color = Color.CYAN
                    style = Paint.Style.FILL
                }
                val dotR = (bmp.width.coerceAtMost(bmp.height) / 140f).coerceAtLeast(2.5f)

                // rows
                for (r in 0 until 10) {
                    for (c in 0 until 9) {
                        val a = grid[r][c]; val b = grid[r][c + 1]
                        canvas.drawLine(a.x, a.y, b.x, b.y, linePaint)
                    }
                }
                // columns
                for (c in 0 until 10) {
                    for (r in 0 until 9) {
                        val a = grid[r][c]; val b = grid[r + 1][c]
                        canvas.drawLine(a.x, a.y, b.x, b.y, linePaint)
                    }
                }
                // dots
                for (r in 0 until 10) {
                    for (c in 0 until 10) {
                        val p = grid[r][c]
                        canvas.drawCircle(p.x, p.y, dotR, dotPaint)
                    }
                }
                return bmp
            }

            // --------------------------------------------------------------------

            // CSV header
            val csv = StringBuilder().apply {
                appendLine(
                    "index,timestamp_ms,total,confPct,lineStraightness,orthogonality,cellUniformity,clearance,jitterScore,minPeak,validPoints,roi_left,roi_top,roi_right,roi_bottom,chosen"
                )
            }

            frames.forEachIndexed { idx, pf ->
                // Localize points to ROI
                val localPts = pf.ptsSrc.map { p ->
                    PointF(p.x - pf.expandedRoi.left, p.y - pf.expandedRoi.top)
                }

                // 1) Save annotated ROI (raw intersections)
                val annotated = drawPointsOverlayOnBitmap(pf.roiBmp, localPts)
                saveBitmapPng(java.io.File(dir, "frame_${idx + 1}_points.png"), annotated)

                // 2) Save raw ROI
                saveBitmapPng(java.io.File(dir, "frame_${idx + 1}_raw.png"), pf.roiBmp)

                // 3) CSV row
                val bd = pf.score ?: ScoreBreakdown(
                    total = 0f, confPct = 0f, lineStraightness = 0f, orthogonality = 0f,
                    cellUniformity = 0f, clearance = 0f, jitterScore = 0f
                )
                val r = pf.roi
                csv.appendLine(
                    listOf(
                        (idx + 1).toString(),
                        pf.tsMs.toString(),
                        f6(bd.total),
                        f6(bd.confPct),
                        f6(bd.lineStraightness),
                        f6(bd.orthogonality),
                        f6(bd.cellUniformity),
                        f6(bd.clearance),
                        f6(bd.jitterScore),
                        f6(pf.minPeak),
                        pf.ptsSrc.size.toString(),
                        f2(r.left), f2(r.top), f2(r.right), f2(r.bottom),
                        if (idx == chosenIdx) "1" else "0"
                    ).joinToString(",")
                )
            }

            // Write CSV + chosen index
            java.io.File(dir, "scores.csv").writeText(csv.toString())
            java.io.File(dir, "chosen_idx.txt").writeText(chosenIdx.toString())

            // --- NEW: write processed lattice overlay for the chosen frame only ---
            if (chosenIdx in frames.indices) {
                val pf = frames[chosenIdx]

                // ROI-local points for the chosen frame
                val localPts = pf.ptsSrc.map { p ->
                    PointF(p.x - pf.expandedRoi.left, p.y - pf.expandedRoi.top)
                }

                // Rebuild the lattice exactly like rectification (ordering → resample → nudge)
                val g0 = orderIntoGrid10(localPts)
                val g1 = resampleRowsEvenX(g0)
                val g2 = nudgeColumnsToMeans(g1, frac = 0.20f)   // use same factor as rectifier

                val processedOverlay = drawProcessedGridOverlay(pf.roiBmp, g2)
                saveBitmapPng(java.io.File(dir, "frame_${chosenIdx + 1}_points_resampled.png"), processedOverlay)
            }

            android.util.Log.i("BestOfN", "Saved Best-of-N debug to ${dir.absolutePath}")
        }.onFailure {
            android.util.Log.e("BestOfN", "Failed saving Best-of-N pack", it)
        }
    }





    // -------------------------------------------------------------------------
    // New helper: score a frame's "grid-likeness" (0..1-ish, higher is better)
    // Uses only data we already have (intersections, ROI, jitter, guard clearance).
    // -------------------------------------------------------------------------
    private fun computeFrameScore(
        ptsSrc: List<PointF>,
        ex: Rect,
        roi: RectF,
        jitterPx128: Float,
        minPeak: Float,
        guardRect: RectF?,
        imgW: Int,
        imgH: Int
    ): ScoreBreakdown {
        if (ptsSrc.size < 90) {
            return ScoreBreakdown(
                total = -1f, confPct = 0f, lineStraightness = 0f,
                orthogonality = 0f, cellUniformity = 0f, clearance = 0f, jitterScore = 0f
            )
        }

        fun to128(p: PointF): PointF {
            val x = ((p.x - ex.left) * 128f / max(1, ex.width()))
            val y = ((p.y - ex.top)  * 128f / max(1, ex.height()))
            return PointF(x, y)
        }

        val g = Array(10) { r -> Array(10) { c -> to128(ptsSrc[r * 10 + c]) } }

        fun lineResidual(points: Array<PointF>): Float {
            val p0 = points.first(); val p1 = points.last()
            val vx = p1.x - p0.x; val vy = p1.y - p0.y
            val vlen = kotlin.math.sqrt(vx*vx + vy*vy).coerceAtLeast(1e-3f)
            val nx = -vy / vlen; val ny =  vx / vlen
            var sum = 0f
            for (p in points) sum += kotlin.math.abs((p.x - p0.x) * nx + (p.y - p0.y) * ny)
            return (sum / points.size)
        }

        var straight = 0f
        for (r in 0 until 10) straight += lineResidual(g[r])
        for (c in 0 until 10) {
            val col = Array(10) { r -> g[r][c] }
            straight += lineResidual(col)
        }
        val lineStraightness = (1f - (straight / (20f * 2.0f))).coerceIn(0f, 1f)

        fun dir(p0: PointF, p1: PointF): PointF {
            val vx = p1.x - p0.x; val vy = p1.y - p0.y
            val l = kotlin.math.sqrt(vx*vx + vy*vy).coerceAtLeast(1e-3f)
            return PointF(vx / l, vy / l)
        }
        val rowDir = dir(g[0].first(), g[0].last())
        val colDir = dir(g.first()[0], g.last()[0])
        val dot = kotlin.math.abs(rowDir.x * colDir.x + rowDir.y * colDir.y)
        val orthogonality = (1f - dot).coerceIn(0f, 1f)

        fun meanStd(values: FloatArray): Pair<Float, Float> {
            val m = values.average().toFloat()
            var v = 0f; for (v0 in values) { val d = v0 - m; v += d * d }
            v /= values.size.coerceAtLeast(1)
            return m to kotlin.math.sqrt(v)
        }
        val widths = FloatArray(9 * 10) { i ->
            val r = i / 9; val c = i % 9
            val a = g[r][c]; val b = g[r][c+1]
            kotlin.math.sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y))
        }
        val heights = FloatArray(10 * 9) { i ->
            val r = i / 10; val c = i % 10
            val a = g[r][c]; val b = g[r+1][c]
            kotlin.math.sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y))
        }
        val wStd = meanStd(widths).second
        val hStd = meanStd(heights).second
        val cellUniformity = (1f - ((wStd + hStd) / 10f)).coerceIn(0f, 1f)

        val clearance = if (guardRect != null) {
            val dL = kotlin.math.abs(roi.left   - guardRect.left)
            val dT = kotlin.math.abs(roi.top    - guardRect.top)
            val dR = kotlin.math.abs(guardRect.right - roi.right)
            val dB = kotlin.math.abs(guardRect.bottom - roi.bottom)
            val dMin = min(min(dL, dR), min(dT, dB))
            (dMin / (min(imgW, imgH) * 0.10f)).coerceIn(0f, 1f)
        } else 1f

        val confPct = minPeak.coerceIn(0f, 1f)
        val jitterScore = (1f - (jitterPx128 / 7f)).coerceIn(0f, 1f)

        val w1=0.25f; val w2=0.20f; val w3=0.15f; val w4=0.15f; val w5=0.10f; val w6=0.15f
        val total = w1*confPct + w2*lineStraightness + w3*orthogonality + w4*cellUniformity + w5*clearance + w6*jitterScore

        return ScoreBreakdown(
            total = total,
            confPct = confPct,
            lineStraightness = lineStraightness,
            orthogonality = orthogonality,
            cellUniformity = cellUniformity,
            clearance = clearance,
            jitterScore = jitterScore
        )
    }





    // Center-crop a square bitmap to an inner region, then resize back to outSize×outSize.
// innerFrac is the fraction to trim from EACH side (e.g. 0.10f = 10% per side).
    private fun centerCropAndResize(
        src: Bitmap,
        innerFrac: Float,
        outSize: Int
    ): Bitmap {
        val w = src.width
        val h = src.height

        // We expect square tiles, but be defensive
        val marginX = ((w * innerFrac).toInt()).coerceIn(0, w / 4)
        val marginY = ((h * innerFrac).toInt()).coerceIn(0, h / 4)

        val cropX = marginX
        val cropY = marginY
        val cropW = (w - 2 * marginX).coerceAtLeast(1)
        val cropH = (h - 2 * marginY).coerceAtLeast(1)

        val cropped = Bitmap.createBitmap(src, cropX, cropY, cropW, cropH)
        val scaled  = Bitmap.createScaledBitmap(cropped, outSize, outSize, true)

        // We don’t need the intermediate cropped bitmap after scaling
        if (cropped != src) {
            try { cropped.recycle() } catch (_: Throwable) {}
        }

        return scaled
    }




    // Runs on a worker thread: expand+crop ROI; call Rectifier.process() to get tiles; push tiles into DigitClassifier; flatten to 81 digits and probabilities. If successful, set 'locked', flash/shutter, and navigate to results; otherwise keep scanning.
    // Runs on a worker thread: use the 10×10 intersection lattice to cut each cell
    // from its own quadrilateral, robust to bowed or bent grid lines.
    // Use the exact expanded ROI (expandedRoiSrc) that the intersections model used.
// This keeps point->ROI-local mapping aligned and fixes the cyan points offset.


    private fun attemptRectifyAndClassify(
        ptsSrc: List<PointF>,
        detectorRoi: RectF,
        srcBmp: Bitmap,
        expandedRoiSrc: Rect
    ) {
        val GREEN_TO_SHUTTER_MS = 150L
        val shutterCanceled = AtomicBoolean(false)

        // schedule shutter; runnable won't do anything if canceled
        val shutterRunnable = Runnable {
            if (!shutterCanceled.get()) {
                overlay.playShutter(null)
            }
        }
        overlay.postDelayed(shutterRunnable, GREEN_TO_SHUTTER_MS)

        Thread {
            var err: String? = null
            var digitsFlat: IntArray? = null
            var probsFlat: FloatArray? = null
            var boardBmpOut: Bitmap? = null

            try {
                val debugDir = getRectifyDebugDir()
                if (!debugDir.exists()) debugDir.mkdirs()
                clearDirectory(debugDir)

                // exact crop used by intersections
                val leftI   = expandedRoiSrc.left.coerceIn(0, srcBmp.width - 1)
                val topI    = expandedRoiSrc.top.coerceIn(0, srcBmp.height - 1)
                val rightI  = expandedRoiSrc.right.coerceIn(leftI + 1, srcBmp.width)
                val bottomI = expandedRoiSrc.bottom.coerceIn(topI + 1, srcBmp.height)

                val roiBitmap = Bitmap.createBitmap(
                    srcBmp,
                    leftI,
                    topI,
                    rightI - leftI,
                    bottomI - topI
                )
                saveBitmapPng(java.io.File(debugDir, "roi.png"), roiBitmap)

                fun toLocal(p: PointF) = PointF(p.x - leftI, p.y - topI)
                val ptsLocal = ptsSrc.map { toLocal(it) }

                val grid10 = orderPointsInto10x10(ptsLocal)
                    ?: throw IllegalStateException("Could not form 10×10 grid (points=${ptsLocal.size})")

                // Visualize ordered grid points on ROI
                saveBitmapPng(
                    java.io.File(debugDir, "roi_points.png"),
                    drawPointsOverlayOnBitmap(roiBitmap, grid10.flatten())
                )

                // --- TILE WARP + CENTER CROP (Fix B) ---------------------------------
                val CELL_SIZE = 64
                val GRID_SIZE = 576
                val INNER_FRAC = 0.07f   // trim 10% from each side of the warped tile

                val tiles = Array(9) { rIdx ->
                    Array(9) { cIdx ->
                        val tl = grid10[rIdx][cIdx]
                        val tr = grid10[rIdx][cIdx + 1]
                        val br = grid10[rIdx + 1][cIdx + 1]
                        val bl = grid10[rIdx + 1][cIdx]

                        // 1) Warp full quad to square
                        val rawTile = warpQuadToSquare(
                            roiBitmap,
                            tl, tr, br, bl,
                            CELL_SIZE, CELL_SIZE
                        )

                        // 2) Center-crop inner region and resize back to 64×64
                        val croppedTile = centerCropAndResize(
                            rawTile,
                            INNER_FRAC,
                            CELL_SIZE
                        )

                        // We no longer need the raw tile bitmap
                        try { rawTile.recycle() } catch (_: Throwable) {}

                        croppedTile
                    }
                }

                // Save final tiles for inspection (these are *post* center-crop tiles)
                for (rr in 0 until 9) {
                    for (cc in 0 until 9) {
                        saveBitmapPng(
                            java.io.File(debugDir, "cell_r${rr + 1}c${cc + 1}.png"),
                            tiles[rr][cc]
                        )
                    }
                }

                // Build mosaic board (also using post-crop tiles)
                val mosaic = Bitmap.createBitmap(GRID_SIZE, GRID_SIZE, Bitmap.Config.ARGB_8888)
                val can = Canvas(mosaic)
                for (rr in 0 until 9) {
                    for (cc in 0 until 9) {
                        can.drawBitmap(
                            tiles[rr][cc],
                            (cc * CELL_SIZE).toFloat(),
                            (rr * CELL_SIZE).toFloat(),
                            null
                        )
                    }
                }
                boardBmpOut = mosaic

                // --- Digit classification -------------------------------------------
                ensureDigitClassifier()
                val (digits, confs) = digitClassifier!!.classifyTiles(tiles)
                digitsFlat = IntArray(81) { i -> digits[i / 9][i % 9] }
                probsFlat  = FloatArray(81) { i -> confs[i / 9][i % 9] }

                // post-rectify gates
                val avgConf = probsFlat!!.average().toFloat()
                val lowConfCells = probsFlat!!.count { it < 0.60f }

                val postSnap = GateSnapshot(
                    hasDetectorLock = true,
                    gridizedOk      = true,
                    validPoints     = ptsSrc.size,
                    jitterPx128     = 0f,
                    rectifyOk       = true,
                    avgConf         = avgConf,
                    lowConfCells    = lowConfCells
                )
                val sm = gate.update(postSnap)
                runOnUiThread { changeGate(sm, "postRectify") }

                // success path → keep provisional green, shutter will fire (already scheduled)
                locked = true

            } catch (t: Throwable) {
                err = t.message ?: "$t"
                Log.e("MainActivity", "attemptRectifyAndClassify failed", t)
            } finally {
                handoffInProgress = false
                if (locked && boardBmpOut != null && digitsFlat != null && probsFlat != null) {
                    // OK → go to results; shutter already scheduled/fired (or will fire)
                    runOnUiThread {
                        showResults(boardBmpOut!!, digitsFlat!!, probsFlat!!)
                    }
                } else {
                    // FAIL → cancel shutter and softly demote to Amber
                    shutterCanceled.set(true)
                    runOnUiThread {
                        overlay.removeCallbacks(shutterRunnable)
                        overlay.setLocked(false)
                        changeGate(GateState.L2, "demote_after_fail")
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
                    // Keep intersections parity packs AND Best-of-N audit packs
                    val keep = f.name.startsWith("intersections_") || f.name.startsWith("bestofN")
                    if (!keep) f.deleteRecursively()
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