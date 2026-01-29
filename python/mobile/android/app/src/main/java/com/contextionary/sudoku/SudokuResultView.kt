package com.contextionary.sudoku

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import android.view.animation.LinearInterpolator
import androidx.core.content.res.ResourcesCompat
import kotlin.math.min

/**
 * Sudoku board renderer with confidence-aware digit styling.
 *
 * Extended (v0.5.0+):
 * - Cyan border for cells auto-corrected by the Sudoku logic engine
 * - Red border for cells that remain unresolved / suspicious
 *
 * New (v0.6.0):
 * - Distinguish printed givens vs handwritten solutions
 * - Optional mini-candidates in empty cells
 * - Two data entry paths:
 *     • setFromCellGrid(readouts)
 *     • setUiData(...)
 *
 * New (v0.?.?):
 * - Third rendering provenance: AUTOCORRECTED
 *   Rendered as PRINTED but BOLD (monospace bold).
 */
class SudokuResultView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    // ---------------- Raw cell data (flattened 81) ----------------
    private val digits: IntArray = IntArray(81) { 0 }
    private val confs: FloatArray = FloatArray(81) { 1f }

    // Provenance flags (how to render the chosen digit)
    private val isGivenShown = BooleanArray(81) { false }       // printed style
    private val isSolutionShown = BooleanArray(81) { false }    // handwritten style
    private val isAutoShown = BooleanArray(81) { false }        // printed bold style

    // Optional candidates (bitmask for 1..9)
    private val candMask = IntArray(81) { 0 }


    private val COLOR_CORRECTED = Color.CYAN

    // Annotations from Sudoku logic layer
    private val changedFlags = BooleanArray(81)      // cyan border
    private val unresolvedFlags = BooleanArray(81)   // red border

    // Cell currently highlighted for user confirmation (LLM-driven)
    private var confirmationIndex: Int? = null
    private var confirmationPulsePhase: Float = 0f
    private var confirmationAnimator: ValueAnimator? = null

    // Paint for the pulsing confirmation border
    private val paintConfirmationBorder = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = Color.YELLOW
        strokeWidth = 4f
    }

    /** Listener for cell taps. */
    interface OnCellClickListener {
        fun onCellClicked(row: Int, col: Int)
    }

    private var cellClickListener: OnCellClickListener? = null
    fun setOnCellClickListener(l: OnCellClickListener?) { cellClickListener = l }

    // -------- Public setters (legacy) --------
    fun setDigits(ints: IntArray?) {
        if (ints != null && ints.size == 81) {
            System.arraycopy(ints, 0, digits, 0, 81)
            // When only digits are set, default to handwritten for nonzero
            for (i in 0 until 81) {
                isAutoShown[i] = false
                isGivenShown[i] = false
                isSolutionShown[i] = digits[i] != 0
            }
            invalidate()
        }
    }

    fun setConfidences(float81: FloatArray?) {
        if (float81 != null && float81.size == 81) {
            System.arraycopy(float81, 0, confs, 0, 81)
            invalidate()
        }
    }

    fun setDigitsAndConfidences(ints: IntArray?, float81: FloatArray?) {
        var changed = false
        if (ints != null && ints.size == 81) {
            System.arraycopy(ints, 0, digits, 0, 81)
            for (i in 0 until 81) {
                isAutoShown[i] = false
                isGivenShown[i] = false
                isSolutionShown[i] = digits[i] != 0
            }
            changed = true
        }
        if (float81 != null && float81.size == 81) {
            System.arraycopy(float81, 0, confs, 0, 81)
            changed = true
        }
        if (changed) invalidate()
    }

    // -------- New: canonical data paths --------

    /**
     * Use this if the controller already fused heads and decided what to show.
     *
     * @param displayDigits           final chosen digit per cell (0..9)
     * @param displayConfs            conf per cell [0..1]
     * @param shownIsGiven            true → render printed style
     * @param shownIsSolution         true → render handwritten style
     * @param candidatesMask          per-cell bitmask (bit d-1 set ⇒ candidate d present)
     * @param shownIsAutoCorrected    true → render printed BOLD style (takes precedence)
     */
    fun setUiData(
        displayDigits: IntArray,
        displayConfs: FloatArray,
        shownIsGiven: BooleanArray,
        shownIsSolution: BooleanArray,
        candidatesMask: IntArray?,
        shownIsAutoCorrected: BooleanArray? = null
    ) {
        require(displayDigits.size == 81 && displayConfs.size == 81)
        require(shownIsGiven.size == 81 && shownIsSolution.size == 81)
        if (shownIsAutoCorrected != null) require(shownIsAutoCorrected.size == 81)

        System.arraycopy(displayDigits, 0, digits, 0, 81)
        System.arraycopy(displayConfs, 0, confs, 0, 81)

        for (i in 0 until 81) {
            isAutoShown[i] = shownIsAutoCorrected?.getOrNull(i) ?: false
            isGivenShown[i] = shownIsGiven[i]
            isSolutionShown[i] = shownIsSolution[i]
            candMask[i] = candidatesMask?.getOrNull(i) ?: 0
        }

        invalidate()
    }

    /**
     * Use this to pass the raw 9×9 readouts straight from CellInterpreter.
     * Applies the S-first rule currently in chooseDisplay().
     *
     * NOTE: This path does NOT know which cells were autocorrected.
     * If you want AUTOCORRECTED styling, use setUiData(...) with shownIsAutoCorrected.
     */
    fun setFromCellGrid(grid: CellGridReadout) {
        require(grid.rows == 9 && grid.cols == 9)
        var k = 0
        for (r in 0 until 9) {
            for (c in 0 until 9) {
                val rd = grid.cells[r][c]
                val (d, p, asGiven, asSol) = chooseDisplay(rd)
                digits[k] = d
                confs[k] = p
                isAutoShown[k] = false
                isGivenShown[k] = asGiven
                isSolutionShown[k] = asSol
                candMask[k] = rd.candidateMask
                k++
            }
        }
        invalidate()

        if (BuildConfig.DEBUG) {
            var g = 0; var s = 0; var a = 0; var z = 0
            for (i in 0 until 81) {
                when {
                    isAutoShown[i] -> a++
                    isGivenShown[i] -> g++
                    isSolutionShown[i] -> s++
                    else -> z++
                }
            }
            android.util.Log.i("SudokuView", "setFromCellGrid: auto=$a printed=$g handwritten=$s blanks=$z")
        }
    }

    // Fusion thresholds (kept for legacy chooseDisplay path)
    private val THR_GIVEN_HI = 0.85f
    private val THR_SOL_HI   = 0.70f
    private val THR_SOL_MID  = 0.45f

    private data class Pick(val digit: Int, val conf: Float, val asGiven: Boolean, val asSol: Boolean)

    private fun chooseDisplay(rd: CellReadout): Pick {
        // Selection by S-first policy:
        // - If S==0 → show GIVEN
        // - Else if G==0 → show SOLUTION
        // - Else → show GIVEN
        val s = rd.solutionDigit
        val g = rd.givenDigit

        return if (s == 0) {
            Pick(g, rd.givenConf, asGiven = (g != 0), asSol = false)
        } else {
            if (g == 0) {
                Pick(s, rd.solutionConf, asGiven = false, asSol = (s != 0))
            } else {
                Pick(g, rd.givenConf, asGiven = (g != 0), asSol = false)
            }
        }
    }

    fun setChangedCells(indices: List<Int>?) {
        java.util.Arrays.fill(changedFlags, false)
        indices?.forEach { idx -> if (idx in 0..80) changedFlags[idx] = true }
        invalidate()
    }

    fun setUnresolvedCells(indices: List<Int>?) {
        java.util.Arrays.fill(unresolvedFlags, false)
        indices?.forEach { idx -> if (idx in 0..80) unresolvedFlags[idx] = true }
        invalidate()
    }

    fun setLogicAnnotations(changed: List<Int>?, unresolved: List<Int>?) {
        java.util.Arrays.fill(changedFlags, false)
        java.util.Arrays.fill(unresolvedFlags, false)
        changed?.forEach { idx -> if (idx in 0..80) changedFlags[idx] = true }
        unresolved?.forEach { idx -> if (idx in 0..80) unresolvedFlags[idx] = true }
        invalidate()
    }

    /** Highlight pulsing */
    fun startConfirmationPulse(index: Int?) {
        val validIndex = index?.takeIf { it in 0..80 }
        confirmationIndex = validIndex
        if (validIndex == null) { stopConfirmationPulse(); return }
        if (confirmationAnimator == null) {
            confirmationAnimator = ValueAnimator.ofFloat(0f, 1f).apply {
                duration = 800L
                repeatCount = ValueAnimator.INFINITE
                repeatMode = ValueAnimator.RESTART
                interpolator = LinearInterpolator()
                addUpdateListener {
                    confirmationPulsePhase = it.animatedValue as Float
                    invalidate()
                }
            }
        }
        if (confirmationAnimator?.isStarted != true) confirmationAnimator?.start()
    }

    fun stopConfirmationPulse() {
        confirmationAnimator?.cancel()
        confirmationAnimator = null
        confirmationIndex = null
        confirmationPulsePhase = 0f
        invalidate()
    }

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        confirmationAnimator?.cancel()
        confirmationAnimator = null
    }

    // ------------- Palette & thresholds -------------
    private val COLOR_HIGH = Color.WHITE
    private val COLOR_MED  = Color.argb(0xE0, 0xFF, 0xD5, 0x4F)
    private val COLOR_LOW  = Color.argb(0xE6, 0xEF, 0x53, 0x50)

    private val THR_HIGH = SudokuConfidence.THRESH_HIGH
    private val THR_LOW  = SudokuConfidence.THRESH_LOW

    // ------------- Paints & geometry -------------
    private val gridPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = Color.argb(0xD9, 0xFF, 0xFF, 0xFF)
        strokeCap = Paint.Cap.SQUARE
        isDither = true
    }
    private val outerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = Color.WHITE
        strokeCap = Paint.Cap.SQUARE
        isDither = true
    }

    // Fonts:
    // - printed (givens): system MONOSPACE
    // - printed bold (autocorrected): system MONOSPACE bold
    // - handwritten (solutions & candidates): Segoe Print Bold
    private val tfPrinted: Typeface = Typeface.create(Typeface.MONOSPACE, Typeface.NORMAL)
    private val tfPrintedBold: Typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
    private val tfHandwritten: Typeface = ResourcesCompat.getFont(context, R.font.segoeprb)
        ?: Typeface.create(Typeface.SANS_SERIF, Typeface.BOLD)

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = COLOR_HIGH
        style = Paint.Style.FILL
        textAlign = Paint.Align.CENTER
        typeface = tfPrinted
        setShadowLayer(dp(1.5f), 0f, 0f, Color.argb(120, 0, 0, 0))
        isSubpixelText = true
    }
    private val underlinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = COLOR_LOW
        strokeCap = Paint.Cap.SQUARE
        isDither = true
    }
    private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = COLOR_MED
        isDither = true
    }
    private val paintChangedBorder = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = Color.CYAN
        strokeCap = Paint.Cap.SQUARE
        isDither = true
        strokeWidth = 3f
    }
    private val paintUnresolvedBorder = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = Color.RED
        strokeCap = Paint.Cap.SQUARE
        isDither = true
        strokeWidth = 3f
    }

    // Candidate paint
    private val candPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(0xCC, 0xFF, 0xFF, 0xFF)
        style = Paint.Style.FILL
        textAlign = Paint.Align.CENTER
        typeface = tfHandwritten
        isSubpixelText = true
    }

    private val boardRect = RectF()
    private var cell = 0f
    private var baselineAdjust = 0f

    private var strokeCell = 1f
    private var strokeBlock = 3f
    private var strokeOuter = 3f

    private var cellInset = 0f
    private var underlineYInset = 0f
    private var underlineThickness = 0f
    private var dotRadius = 0f
    private var dotInset = 0f

    // ---------------- Layout / measure ----------------
    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val w = MeasureSpec.getSize(widthMeasureSpec)
        val h = MeasureSpec.getSize(heightMeasureSpec)
        val size = min(w, h)
        setMeasuredDimension(size, size)
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        val pad = maxOf(dp(16f), w * 0.04f)
        boardRect.set(pad, pad, w - pad, h - pad)
        cell = boardRect.width() / 9f

        strokeCell  = clamp(dp(1f), cell * 0.018f, dp(1.5f))
        strokeBlock = dp(3f)
        strokeOuter = dp(3f)

        cellInset = cell * 0.12f
        textPaint.textSize = (cell - 2f * cellInset) * 0.80f
        val fm = textPaint.fontMetrics
        baselineAdjust = -(fm.ascent + fm.descent) / 2f

        underlineThickness = dp(1f)
        underlinePaint.strokeWidth = underlineThickness
        underlineYInset = cellInset * 0.55f

        dotRadius = dp(2f)
        dotInset  = cellInset * 0.45f

        val borderStroke = clamp(dp(1.5f), cell * 0.06f, dp(3f))
        paintChangedBorder.strokeWidth = borderStroke
        paintUnresolvedBorder.strokeWidth = borderStroke

        // Candidate text size (smaller)
        candPaint.textSize = (cell - 2f * cellInset) * 0.32f
    }

    // ---------------- Drawing ----------------
    override fun onDraw(canvas: Canvas) {
        // 1) Outer border
        outerPaint.strokeWidth = strokeOuter
        drawAlignedRect(canvas, boardRect, outerPaint)

        // 2) 3x3 block lines
        gridPaint.strokeWidth = strokeBlock
        for (i in 0..9 step 3) {
            val x = boardRect.left + i * cell
            val y = boardRect.top + i * cell
            drawAlignedV(canvas, x, boardRect.top, boardRect.bottom, gridPaint)
            drawAlignedH(canvas, y, boardRect.left, boardRect.right, gridPaint)
        }

        // 3) Cell lines
        gridPaint.strokeWidth = strokeCell
        for (i in 0..9) {
            val x = boardRect.left + i * cell
            val y = boardRect.top + i * cell
            drawAlignedV(canvas, x, boardRect.top, boardRect.bottom, gridPaint)
            drawAlignedH(canvas, y, boardRect.left, boardRect.right, gridPaint)
        }

        // 4) Digits + markers + annotations
        for (r in 0 until 9) for (c in 0 until 9) {
            val idx = r * 9 + c

            // Cell rect
            val cellLeft   = boardRect.left + c * cell
            val cellTop    = boardRect.top  + r * cell
            val cellRight  = cellLeft + cell
            val cellBottom = cellTop + cell
            val cellRect = RectF(cellLeft, cellTop, cellRight, cellBottom)

            // Annotations and confirmation
            drawAnnotationsForCell(canvas, idx, cellRect)
            drawConfirmationForCell(canvas, idx, cellRect)

            val v = digits[idx]
            if (v == 0) {
                // Empty → draw mini-candidates if any
                val mask = candMask[idx]
                if (mask != 0) drawMiniCandidates(canvas, cellRect, mask)
                continue
            }

            val conf = confs[idx]

            // ✅ "Corrected" means auto OR manual (you will pass union into isAutoShown)
            val isCorrectedShown = isAutoShown[idx]

            // Confidence color (default)
            val baseColor = when {
                conf >= THR_HIGH -> COLOR_HIGH
                conf >= THR_LOW  -> COLOR_MED
                else             -> COLOR_LOW
            }

            // ✅ New rule: corrected digits are CYAN
            textPaint.color = if (isCorrectedShown) COLOR_CORRECTED else baseColor


            // Provenance precedence:
            // AUTO (printed bold) > GIVEN (printed) > SOLUTION (handwritten)
            textPaint.typeface = when {
                isAutoShown[idx] -> tfPrintedBold
                isGivenShown[idx] -> tfPrinted
                else -> tfHandwritten
            }

            val left   = cellLeft + cellInset
            val top    = cellTop  + cellInset
            val right  = left + (cell - 2f * cellInset)
            val bottom = top  + (cell - 2f * cellInset)

            val cx = (left + right) * 0.5f
            val cy = (top + bottom) * 0.5f + baselineAdjust
            canvas.drawText(v.toString(), cx, cy, textPaint)

            // ✅ Optional cleanup: don't show uncertainty markers on corrected digits
            if (!isCorrectedShown) {
                if (conf < THR_HIGH && conf >= THR_LOW) {
                    val dx = right - dotInset
                    val dy = top + dotInset
                    canvas.drawCircle(dx, dy, dotRadius, dotPaint)
                }
                if (conf < THR_LOW) {
                    val uy = bottom - underlineYInset
                    val ux1 = left + dp(2f)
                    val ux2 = right - dp(2f)
                    drawAlignedH(canvas, uy, ux1, ux2, underlinePaint)
                }
            }
        }
    }

    // Draw 3×3 candidate notes inside a cell.
    private fun drawMiniCandidates(canvas: Canvas, rect: RectF, mask: Int) {
        val cols = 3
        val rows = 3
        val innerPad = cellInset * 0.55f
        val w = (rect.width() - 2 * innerPad) / cols
        val h = (rect.height() - 2 * innerPad) / rows
        val fm = candPaint.fontMetrics
        val base = -(fm.ascent + fm.descent) / 2f

        for (d in 1..9) {
            // ✅ FIX: mask is encoded with (1 shl (d - 1))
            if ((mask and (1 shl (d - 1))) == 0) continue
            val i = (d - 1) % cols
            val j = (d - 1) / cols
            val cx = rect.left + innerPad + i * w + w / 2f
            val cy = rect.top  + innerPad + j * h + h / 2f + base
            canvas.drawText(d.toString(), cx, cy, candPaint)
        }
    }

    private fun drawAnnotationsForCell(canvas: Canvas, idx: Int, cellRect: RectF) {
        val hasChanged = changedFlags[idx]
        val isUnresolved = unresolvedFlags[idx]
        if (!hasChanged && !isUnresolved) return

        val inset = cell * 0.10f
        val r = RectF(
            cellRect.left + inset,
            cellRect.top + inset,
            cellRect.right - inset,
            cellRect.bottom - inset
        )
        if (hasChanged) drawAlignedRect(canvas, r, paintChangedBorder)
        if (isUnresolved) drawAlignedRect(canvas, r, paintUnresolvedBorder)
    }

    private fun drawConfirmationForCell(canvas: Canvas, idx: Int, cellRect: RectF) {
        val target = confirmationIndex ?: return
        if (idx != target) return
        val inset = cell * 0.12f
        val r = RectF(
            cellRect.left + inset,
            cellRect.top + inset,
            cellRect.right - inset,
            cellRect.bottom - inset
        )
        val phase = confirmationPulsePhase
        val pulse = 0.5f + 0.5f * kotlin.math.sin(phase * 2f * Math.PI).toFloat()
        val alpha = (128 + pulse * 127).toInt().coerceIn(80, 255)
        paintConfirmationBorder.alpha = alpha
        paintConfirmationBorder.strokeWidth = strokeCell * (1.5f + pulse)
        drawAlignedRect(canvas, r, paintConfirmationBorder)
    }

    // ---------------- Touch handling ----------------
    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (cellClickListener == null) return super.onTouchEvent(event)
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> return true
            MotionEvent.ACTION_UP -> {
                val x = event.x
                val y = event.y
                if (!boardRect.contains(x, y)) {
                    performClick()
                    return super.onTouchEvent(event)
                }
                if (cell > 0f) {
                    val col = ((x - boardRect.left) / cell).toInt().coerceIn(0, 8)
                    val row = ((y - boardRect.top) / cell).toInt().coerceIn(0, 8)
                    cellClickListener?.onCellClicked(row, col)
                }
                performClick()
                return true
            }
        }
        return super.onTouchEvent(event)
    }

    override fun performClick(): Boolean = super.performClick()

    // ---------------- Helpers ----------------
    private fun dp(v: Float): Float = v * resources.displayMetrics.density
    private fun clamp(min: Float, value: Float, max: Float): Float =
        when { value < min -> min; value > max -> max; else -> value }

    /** Pixel-align strokes so thin lines stay crisp. */
    private fun aligned(pos: Float, stroke: Float): Float =
        if ((stroke % 2f) == 1f) pos.toInt() + 0.5f else pos.toInt().toFloat()

    private fun drawAlignedV(c: Canvas, x: Float, y1: Float, y2: Float, p: Paint) {
        val px = aligned(x, p.strokeWidth); c.drawLine(px, y1, px, y2, p)
    }
    private fun drawAlignedH(c: Canvas, y: Float, x1: Float, x2: Float, p: Paint) {
        val py = aligned(y, p.strokeWidth); c.drawLine(x1, py, x2, py, p)
    }
    private fun drawAlignedRect(c: Canvas, r: RectF, p: Paint) {
        val l = aligned(r.left,   p.strokeWidth)
        val t = aligned(r.top,    p.strokeWidth)
        val rr = aligned(r.right, p.strokeWidth)
        val b = aligned(r.bottom, p.strokeWidth)
        c.drawRect(l, t, rr, b, p)
    }

    /** Export exactly what you see to a Bitmap (ARGB_8888). */
    fun renderToBitmap(): Bitmap {
        val w = width.takeIf { it > 0 } ?: 1080
        val h = height.takeIf { it > 0 } ?: 1080
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmp)
        canvas.drawColor(Color.BLACK)
        draw(canvas)
        return bmp
    }
}