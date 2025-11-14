package com.contextionary.sudoku

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import kotlin.math.max
import kotlin.math.min

import android.animation.ValueAnimator
import android.media.MediaActionSound
import android.util.Log

class OverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private var srcW = 0
    private var srcH = 0
    @Volatile private var boxes: List<Detector.Det> = emptyList()
    private var hudThresh: Float = 0.55f
    private var hudMaxDets: Int = 6
    private var useFillCenter = false

    // Corner confidence display gate
    private var cornerPeakThresh = 0.92f

    // LOCK state text (optional)
    private var lockedHud: Boolean = false

    // Refined corners + peaks (TL, TR, BR, BL)
    // (Legacy) Corner-based HUD is retired. Keep placeholders to avoid call-site errors.
    @Volatile private var corners: List<PointF>? = null
    @Volatile private var cornerPeaks: FloatArray? = null

    // Model input crop (expanded ROI) to overlay (in source/bitmap coords)
    @Volatile private var cornerCropRect: Rect? = null

    // === Cyan guard (square) bookkeeping ===
    // We draw the cyan guide in *view* space, but we also maintain the corresponding
    // rect in *source/bitmap* space so MainActivity can gate against it reliably.
    @Volatile private var guardRectSrc: RectF? = null

    // === Capture "shutter" pulse animation state (in source coords) ===
    @Volatile private var capturePulseRect: RectF? = null
    @Volatile private var capturePulseStartMs: Long = 0L

    // Scale-aware sizes
    private var strokePx = 4f
    private var textPx = 28f
    private var cornerRadiusPx = 8f
    private var hudPadPx = 10f
    private var cornerDotRadiusPx = 6f


    // Shutter flash state
    private var flashAlpha: Float = 0f
    private val flashPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE }
    private var shutterSound: MediaActionSound? = null

    // --- Begin: digits HUD STUB (no drawing here) ---

    @Suppress("UNUSED_VARIABLE")
    private var digits: Array<IntArray>? = null
    @Suppress("UNUSED_VARIABLE")
    private var probs: Array<FloatArray>? = null

    // === Intersections HUD ===
    @Volatile private var intersectionsSrc: List<PointF>? = null
    var showIntersections: Boolean = true
        set(v) { field = v; invalidate() }
    fun updateIntersections(pointsSrc: List<PointF>?) {
        intersectionsSrc = pointsSrc
        invalidate()
    }


    // Whether to draw refined corner dots/peaks on the HUD
    var showCornerDots: Boolean = true
        set(value) {
            field = value
            invalidate()
        }


    // ===== HUD toggles expected by MainActivity =====
    var showBoxLabels: Boolean = true
        set(value) {
            field = value
            invalidate()
        }

    var showHudText: Boolean = true
        set(value) {
            field = value
            invalidate()
        }

    // Gate state display (NONE/L1/L2/L3)
    private var gateState: GateState = GateState.NONE
    fun setGateState(state: GateState) {
        gateState = state
        invalidate()
    }


    // Paints
    private val redFill = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.argb(70, 255, 0, 0)
    }
    private val redStroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = Color.rgb(255, 64, 96)
        strokeWidth = strokePx
    }
    private val hudText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = textPx
        typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
    }
    private val hudBg = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(160, 0, 0, 0)
        style = Paint.Style.FILL
    }
    private val guideStroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = Color.CYAN
        strokeWidth = strokePx
    }
    private val cornerFill = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.rgb(0, 255, 128)
    }
    private val cornerOutline = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = Color.BLACK
        strokeWidth = 1.6f
    }

    // Green overlay for model crop (drawn only if not null)
    private val cropFill = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.argb(60, 0, 255, 0) // translucent green
    }
    private val cropStroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = Color.rgb(0, 200, 0)
        strokeWidth = 2f
        pathEffect = DashPathEffect(floatArrayOf(10f, 8f), 0f)
    }


    fun updateDigits(d: Array<IntArray>?, p: Array<FloatArray>?, lowThresh: Float = 0.85f) {
        digits = d
        probs = p
        // Intentionally not drawing digits here anymore.
    }


    fun setLocked(v: Boolean) {
        lockedHud = v
        invalidate()
    }

    fun setCornerPeakThreshold(t: Float) {
        cornerPeakThresh = t.coerceIn(0f, 1f)
        invalidate()
    }

    fun setUseFillCenter(fill: Boolean) {
        useFillCenter = fill
        invalidate()
    }

    fun setSourceSize(w: Int, h: Int) {
        if (w != srcW || h != srcH) {
            srcW = w
            srcH = h
            invalidate()
        }
    }

    /**
     * Plays the camera shutter sound and shows a brief white flash overlay.
     * Keep signature to match MainActivity call.
     */
    fun playShutter(@Suppress("UNUSED_PARAMETER") anchor: View?) {
        // Sound
        try {
            if (shutterSound == null) shutterSound = MediaActionSound()
            shutterSound?.play(MediaActionSound.SHUTTER_CLICK)
        } catch (t: Throwable) {
            Log.w("OverlayView", "MediaActionSound failed", t)
        }

        // Flash animation (0.9 -> 0)
        val anim = ValueAnimator.ofFloat(0.9f, 0f).apply {
            duration = 260L
            addUpdateListener {
                flashAlpha = it.animatedValue as Float
                invalidate()
            }
        }
        anim.start()
    }

    fun updateBoxes(d: List<Detector.Det>, scoreThresh: Float, maxDets: Int) {
        boxes = d.toList()
        hudThresh = scoreThresh
        hudMaxDets = maxDets
        invalidate()
    }

    fun updateCorners(c: List<PointF>?, peaks: FloatArray? = null) {
        corners = c
        cornerPeaks = peaks
        invalidate()
    }

    fun updateCornerCropRect(r: Rect?) {
        cornerCropRect = r
        invalidate()
    }

    /** Public accessor used by MainActivity for the cyan-guard gate (source coords). */
    fun getGuardRectInSource(): RectF? = guardRectSrc

    /** Trigger the short capture pulse animation (rect must be in source/bitmap coords). */
    fun startCapturePulse(rectInSrc: RectF) {
        capturePulseRect = RectF(rectInSrc)
        capturePulseStartMs = System.currentTimeMillis()
        postInvalidateOnAnimation()
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        val base = max(1f, min(w, h) / 240f)
        strokePx = 2.5f * base
        textPx = 12f * base
        cornerRadiusPx = 4f * base
        hudPadPx = 6f * base
        cornerDotRadiusPx = 3.2f * base

        redStroke.strokeWidth = strokePx
        guideStroke.strokeWidth = strokePx
        hudText.textSize = textPx
        cornerOutline.strokeWidth = max(1.2f, 0.8f * base)
        cropStroke.strokeWidth = max(2f, 1.2f * base)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (srcW <= 0 || srcH <= 0) return

        val vw = width.toFloat()
        val vh = height.toFloat()

        val sx = vw / srcW
        val sy = vh / srcH
        val s: Float
        val dw: Float
        val dh: Float
        val offX: Float
        val offY: Float
        if (useFillCenter) {
            s = max(sx, sy)
            dw = srcW * s; dh = srcH * s
            offX = (vw - dw) / 2f; offY = (vh - dh) / 2f
        } else {
            s = min(sx, sy)
            dw = srcW * s; dh = srcH * s
            offX = (vw - dw) / 2f; offY = (vh - dh) / 2f
        }

        fun mapX(xBmp: Float) = offX + xBmp * s
        fun mapY(yBmp: Float) = offY + yBmp * s
        fun unmapX(xView: Float) = (xView - offX) / s
        fun unmapY(yView: Float) = (yView - offY) / s

        // ===== Top-left HUD panel (honor showHudText) =====
        if (showHudText) {
            val lines = buildList {
                add("bitmap: ${srcW}×${srcH}")
                add("boxes (≥${"%.2f".format(hudThresh)}): ${boxes.size}")
                add("max dets: $hudMaxDets")
                add("Gate: $gateState")
            }
            val pad = hudPadPx
            val lineGap = 0.35f * textPx
            val hudW = lines.maxOf { hudText.measureText(it) } + pad * 2
            val hudH = lines.size * (hudText.textSize + lineGap) + pad
            var ty = 12f + textPx
            val hudRect = RectF(12f, 12f, 12f + hudW, 12f + hudH)
            canvas.drawRoundRect(hudRect, cornerRadiusPx, cornerRadiusPx, hudBg)
            lines.forEach {
                canvas.drawText(it, hudRect.left + pad, ty + 4f, hudText)
                ty += hudText.textSize + lineGap
            }
        }

        // ===== Red detection boxes (+ optional labels) =====
        val labelBg = Paint(hudBg)
        val labelPadX = 0.6f * hudPadPx
        val labelPadY = 0.4f * hudPadPx
        for (d in boxes) {
            val r = d.box
            val l = mapX(r.left)
            val t = mapY(r.top)
            val rr = mapX(r.right)
            val bb = mapY(r.bottom)

            canvas.drawRect(l, t, rr, bb, redFill)
            canvas.drawRect(l, t, rr, bb, redStroke)

            if (showBoxLabels) {
                val label = "s=${"%.2f".format(d.score)}"
                val tw = hudText.measureText(label)
                val th = hudText.textSize + labelPadY * 2
                val boxTop = (t - th - 4f).coerceAtLeast(0f)
                val bgRect = RectF(l, boxTop, l + tw + labelPadX * 2, boxTop + th)
                canvas.drawRoundRect(bgRect, cornerRadiusPx, cornerRadiusPx, labelBg)
                canvas.drawText(label, l + labelPadX, boxTop + hudText.textSize + (labelPadY / 2f), hudText)
            }
        }

        // ===== Cyan square guide (view space) + keep a source-space copy for gating =====
        val mapped = RectF(offX, offY, offX + dw, offY + dh)
        val side = min(mapped.width(), mapped.height())
        val guide = if (mapped.width() <= mapped.height()) {
            val top = mapped.centerY() - side / 2f
            RectF(mapped.left, top, mapped.left + side, top + side)
        } else {
            val left = mapped.centerX() - side / 2f
            RectF(left, mapped.top, left + side, mapped.top + side)
        }
        canvas.drawRect(guide, guideStroke)
        guardRectSrc = RectF(
            unmapX(guide.left),
            unmapY(guide.top),
            unmapX(guide.right),
            unmapY(guide.bottom)
        )

        // ===== Green overlay for model crop (optional) =====
        cornerCropRect?.let { r ->
            val l = mapX(r.left.toFloat())
            val t = mapY(r.top.toFloat())
            val rr = mapX(r.right.toFloat())
            val bb = mapY(r.bottom.toFloat())
            canvas.drawRect(l, t, rr, bb, cropFill)
            canvas.drawRect(l, t, rr, bb, cropStroke)
        }


        // ===== 100 intersection dots (no labels) =====
        intersectionsSrc?.let { pts ->
            if (showIntersections) {
                for (p in pts) {
                    val vx = mapX(p.x)
                    val vy = mapY(p.y)
                    canvas.drawCircle(vx, vy, cornerDotRadiusPx + 1.2f, cornerOutline)
                    canvas.drawCircle(vx, vy, cornerDotRadiusPx, cornerFill)
                }
            }
        }



        // ===== Optional "LOCKED" tag =====
        if (lockedHud) {
            val p = Paint().apply {
                color = Color.GREEN
                isAntiAlias = true
                textSize = 48f
                style = Paint.Style.FILL
                typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            }
            val text = "LOCKED"
            val x = 32f
            val y = 64f
            val shadow = Paint(p).apply { color = Color.BLACK; alpha = 160 }
            canvas.drawText(text, x + 2f, y + 2f, shadow)
            canvas.drawText(text, x, y, p)
        }

        // ===== Capture pulse animation =====
        capturePulseRect?.let { rSrc ->
            val elapsed = System.currentTimeMillis() - capturePulseStartMs
            val dur = 200L
            if (elapsed <= dur) {
                val t = elapsed.toFloat() / dur.toFloat()      // 0..1
                val grow = 1f + 0.10f * t                      // expand up to +10%
                val alpha = (255 * (1f - t)).toInt().coerceIn(0, 255)

                val cx = mapX(rSrc.centerX())
                val cy = mapY(rSrc.centerY())
                val hw = (rSrc.width()  * 0.5f * grow) * s
                val hh = (rSrc.height() * 0.5f * grow) * s

                val ring = RectF(cx - hw, cy - hh, cx + hw, cy + hh)

                val ringPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    style = Paint.Style.STROKE
                    strokeWidth = 6f
                    color = Color.WHITE
                    this.alpha = alpha
                }
                canvas.drawRect(ring, ringPaint)

                val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    style = Paint.Style.FILL
                    color = Color.WHITE
                    this.alpha = (32 * (1f - t)).toInt().coerceIn(0, 32)
                }
                canvas.drawRect(ring, fillPaint)

                postInvalidateOnAnimation()
            } else {
                capturePulseRect = null
            }
        }

        // ===== White flash overlay (shutter feedback) =====
        if (flashAlpha > 0f) {
            val oldAlpha = flashPaint.alpha
            flashPaint.alpha = (flashAlpha * 255f).toInt().coerceIn(0, 255)
            canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), flashPaint)
            flashPaint.alpha = oldAlpha
        }

        // (Digits HUD intentionally omitted; leave as-is or wire to your own preview space)
    }
}