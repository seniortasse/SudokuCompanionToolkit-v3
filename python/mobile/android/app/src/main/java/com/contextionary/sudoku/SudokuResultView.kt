package com.contextionary.sudoku

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

/**
 * Sudoku board renderer with confidence-aware digit styling.
 * - High: white
 * - Medium: amber + small corner dot
 * - Very low: red + thin underline
 * - Monochrome grid on black; crisp pixel-aligned strokes
 */
class SudokuResultView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    // ---------------- Data ----------------
    private val digits: IntArray = IntArray(81) { 0 }
    private val confs: FloatArray = FloatArray(81) { 1f }

    /** Set just the digits (confidences remain as-is). */
    fun setDigits(ints: IntArray?) {
        if (ints != null && ints.size == 81) {
            System.arraycopy(ints, 0, digits, 0, 81)
            invalidate()
        }
    }

    /** Set just the confidences (digits remain as-is). */
    fun setConfidences(float81: FloatArray?) {
        if (float81 != null && float81.size == 81) {
            System.arraycopy(float81, 0, confs, 0, 81)
            invalidate()
        }
    }

    /** Convenience: set digits and confidences together. */
    fun setDigitsAndConfidences(ints: IntArray?, float81: FloatArray?) {
        var changed = false
        if (ints != null && ints.size == 81) {
            System.arraycopy(ints, 0, digits, 0, 81)
            changed = true
        }
        if (float81 != null && float81.size == 81) {
            System.arraycopy(float81, 0, confs, 0, 81)
            changed = true
        }
        if (changed) invalidate()
    }

    // ------------- Palette & thresholds -------------
    private val COLOR_HIGH = Color.WHITE                         // #FFFFFFFF
    private val COLOR_MED  = Color.argb(0xE0, 0xFF, 0xD5, 0x4F)  // #E0FFD54F (Amber 300-ish)
    private val COLOR_LOW  = Color.argb(0xE6, 0xEF, 0x53, 0x50)  // #E6EF5350 (Red 400-ish)

    private val THR_HIGH = 0.85f
    private val THR_LOW  = 0.60f

    // ------------- Paints & geometry -------------
    private val gridPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = Color.argb(0xD9, 0xFF, 0xFF, 0xFF) // slightly softened white
        strokeCap = Paint.Cap.SQUARE
        isDither = true
    }
    private val outerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = Color.WHITE
        strokeCap = Paint.Cap.SQUARE
        isDither = true
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = COLOR_HIGH
        style = Paint.Style.FILL
        textAlign = Paint.Align.CENTER
        typeface = Typeface.create(Typeface.MONOSPACE, Typeface.NORMAL)
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

    private val boardRect = RectF()
    private var cell = 0f
    private var baselineAdjust = 0f

    // Line widths
    private var strokeCell = 1f
    private var strokeBlock = 3f   // equal thickness with outer per request
    private var strokeOuter = 3f

    // Insets and markers
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
        // Inner padding so outer stroke doesn't clip; ~16–20dp or 4% of width
        val pad = maxOf(dp(16f), w * 0.04f)
        boardRect.set(pad, pad, w - pad, h - pad)

        cell = boardRect.width() / 9f

        // Strokes (cell thin; 3x3 and outer both 3dp)
        strokeCell  = clamp(dp(1f), cell * 0.018f, dp(1.5f))
        strokeBlock = dp(3f)
        strokeOuter = dp(3f)

        // Digits: ~65% of safe cell, with ~10–12% inset
        cellInset = cell * 0.12f
        textPaint.textSize = (cell - 2f * cellInset) * 0.80f
        val fm = textPaint.fontMetrics
        baselineAdjust = -(fm.ascent + fm.descent) / 2f

        // Underline & dot metrics
        underlineThickness = dp(1f)
        underlinePaint.strokeWidth = underlineThickness
        underlineYInset = cellInset * 0.55f

        dotRadius = dp(2f)
        dotInset  = cellInset * 0.45f
    }

    // ---------------- Drawing ----------------
    override fun onDraw(canvas: Canvas) {
        // 1) Outer border
        outerPaint.strokeWidth = strokeOuter
        drawAlignedRect(canvas, boardRect, outerPaint)

        // 2) 3x3 block lines (every 3rd line)
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

        // 4) Digits + markers
        for (r in 0 until 9) {
            for (c in 0 until 9) {
                val v = digits[r * 9 + c]
                if (v == 0) continue

                val conf = confs[r * 9 + c]
                val color = when {
                    conf >= THR_HIGH -> COLOR_HIGH
                    conf >= THR_LOW  -> COLOR_MED
                    else             -> COLOR_LOW
                }
                textPaint.color = color

                // Safe content rect
                val left   = boardRect.left + c * cell + cellInset
                val top    = boardRect.top  + r * cell + cellInset
                val right  = left + (cell - 2f * cellInset)
                val bottom = top  + (cell - 2f * cellInset)

                // Digit position
                val cx = (left + right) * 0.5f
                val cy = (top + bottom) * 0.5f + baselineAdjust
                canvas.drawText(v.toString(), cx, cy, textPaint)

                // Medium: small dot (accessibility cue)
                if (conf < THR_HIGH && conf >= THR_LOW) {
                    val dx = right - dotInset
                    val dy = top + dotInset
                    canvas.drawCircle(dx, dy, dotRadius, dotPaint)
                }

                // Very low: subtle underline
                if (conf < THR_LOW) {
                    val uy = bottom - underlineYInset
                    val ux1 = left + dp(2f)
                    val ux2 = right - dp(2f)
                    drawAlignedH(canvas, uy, ux1, ux2, underlinePaint)
                }
            }
        }
    }

    // ---------------- Helpers ----------------
    private fun dp(v: Float): Float = v * resources.displayMetrics.density

    private fun clamp(min: Float, value: Float, max: Float): Float =
        when {
            value < min -> min
            value > max -> max
            else -> value
        }

    /** Pixel-align strokes so thin lines stay crisp. */
    private fun aligned(pos: Float, stroke: Float): Float =
        if ((stroke % 2f) == 1f) pos.toInt() + 0.5f else pos.toInt().toFloat()

    private fun drawAlignedV(c: Canvas, x: Float, y1: Float, y2: Float, p: Paint) {
        val px = aligned(x, p.strokeWidth)
        c.drawLine(px, y1, px, y2, p)
    }
    private fun drawAlignedH(c: Canvas, y: Float, x1: Float, x2: Float, p: Paint) {
        val py = aligned(y, p.strokeWidth)
        c.drawLine(x1, py, x2, py, p)
    }
    private fun drawAlignedRect(c: Canvas, r: RectF, p: Paint) {
        val l = aligned(r.left,   p.strokeWidth)
        val t = aligned(r.top,    p.strokeWidth)
        val rr = aligned(r.right, p.strokeWidth)
        val b = aligned(r.bottom, p.strokeWidth)
        c.drawRect(l, t, rr, b, p)
    }
}