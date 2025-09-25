package com.contextionary.sudoku

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import android.widget.Toast
import kotlin.math.max
import kotlin.math.min

/**
 * OverlayView
 *
 * Renders detection boxes on top of the camera preview.
 * - Matches PreviewView’s scale mode (FIT or FILL) via setUseFillCenter().
 * - Shows a small HUD (bitmap size, box count, max dets).
 * - Draws ONLY red boxes with a one-line label: "s=0.##".
 *
 * Debug toggles at the top let you temporarily enable the cyan “mapped bitmap”
 * frame if you ever need it again.
 */

// Simple package-level dev flags (inlined here so you don't need another file)
object DevFlags {
    @Volatile var showRectifierDebug: Boolean = false
}

class OverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    // -------- Debug toggles (all OFF by default) -----------------------------
    private val SHOW_GUIDE = true   // cyan frame of the mapped bitmap area

    // -------- External inputs from MainActivity/Detector ---------------------
    private var srcW = 0                       // original bitmap width
    private var srcH = 0                       // original bitmap height
    private var boxes: List<Detector.Det> = emptyList() // current detections
    private var hudThresh: Float = 0.55f       // HUD display only
    private var hudMaxDets: Int = 6            // HUD display only
    private var useFillCenter = false          // match PreviewView scale mode

    // Long-press anywhere on the overlay to toggle rectifier debug mode
    init {
        setOnLongClickListener {
            DevFlags.showRectifierDebug = !DevFlags.showRectifierDebug
            Toast.makeText(
                context,
                if (DevFlags.showRectifierDebug) "Rectifier Debug: ON" else "Rectifier Debug: OFF",
                Toast.LENGTH_SHORT
            ).show()
            true
        }
    }

    /** Call from MainActivity after you set previewView.scaleType. */
    fun setUseFillCenter(fill: Boolean) {
        useFillCenter = fill
        invalidate()
    }

    /** Inform the overlay of the analyzer bitmap size. */
    fun setSourceSize(w: Int, h: Int) {
        srcW = w
        srcH = h
        invalidate()
    }

    /** Push new detections + thresholds to render. */
    fun updateBoxes(d: List<Detector.Det>, scoreThresh: Float, maxDets: Int) {
        boxes = d
        hudThresh = scoreThresh
        hudMaxDets = maxDets
        invalidate()
    }

    // -------- Paints ---------------------------------------------------------
    private val redFill = Paint().apply {
        style = Paint.Style.FILL
        color = Color.argb(70, 255, 0, 0) // translucent red fill
        isAntiAlias = true
    }
    private val redStroke = Paint().apply {
        style = Paint.Style.STROKE
        color = Color.rgb(255, 64, 96)   // bright red stroke
        strokeWidth = 4f
        isAntiAlias = true
    }
    private val hudText = Paint().apply {
        color = Color.WHITE
        textSize = 28f
        typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
        isAntiAlias = true
    }
    private val hudBg = Paint().apply {
        color = Color.argb(160, 0, 0, 0)
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    // Optional cyan guide (off by default)
    private val guideStroke = Paint().apply {
        style = Paint.Style.STROKE
        color = Color.CYAN
        strokeWidth = 4f
        isAntiAlias = true
    }

    // -------- Rendering -------------------------------------------------------
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (srcW <= 0 || srcH <= 0) return

        val vw = width.toFloat()
        val vh = height.toFloat()

        // Compute mapping from BITMAP space → VIEW space.
        // Must match PreviewView’s content transform.
        val sx = vw / srcW
        val sy = vh / srcH
        val s: Float
        val dw: Float
        val dh: Float
        val offX: Float
        val offY: Float
        if (useFillCenter) {
            // FILL_CENTER: scale = max, center-crop in whichever axis overflows
            s = max(sx, sy)
            dw = srcW * s; dh = srcH * s
            offX = (vw - dw) / 2f; offY = (vh - dh) / 2f
        } else {
            // FIT_CENTER: scale = min, letterbox in whichever axis has room
            s = min(sx, sy)
            dw = srcW * s; dh = srcH * s
            offX = (vw - dw) / 2f; offY = (vh - dh) / 2f
        }

        fun mapX(xBmp: Float) = offX + xBmp * s
        fun mapY(yBmp: Float) = offY + yBmp * s

        // --- HUD (top-left) ---------------------------------------------------
        val lines = listOf(
            "bitmap: ${srcW}×${srcH}",
            "boxes (≥${"%.2f".format(hudThresh)}): ${boxes.size}",
            "Max detections: $hudMaxDets"
        )
        val pad = 10f
        val hudW = lines.maxOf { hudText.measureText(it) } + pad * 2
        val hudH = lines.size * (hudText.textSize + 6f) + pad
        var ty = 16f + hudText.textSize
        canvas.drawRoundRect(RectF(12f, 12f, 12f + hudW, 12f + hudH), 8f, 8f, hudBg)
        lines.forEach {
            canvas.drawText(it, 12f + pad, ty + 6f, hudText)
            ty += hudText.textSize + 6f
        }

        // --- RED boxes only ---------------------------------------------------
        val labelBg = Paint(hudBg) // reuse HUD bg style
        for (d in boxes) {
            val r = d.box
            val l = mapX(r.left)
            val t = mapY(r.top)
            val rr = mapX(r.right)
            val bb = mapY(r.bottom)

            // draw red rectangle
            canvas.drawRect(l, t, rr, bb, redFill)
            canvas.drawRect(l, t, rr, bb, redStroke)

            // one-line label: score
            val label = "s=${"%.2f".format(d.score)}"
            val tw = hudText.measureText(label)
            val th = hudText.textSize + 10f
            val boxTop = (t - th - 6f).coerceAtLeast(0f)
            val bgRect = RectF(l, boxTop, l + tw + 18f, boxTop + th + 8f)
            canvas.drawRoundRect(bgRect, 6f, 6f, labelBg)
            canvas.drawText(label, l + 9f, boxTop + hudText.textSize + 2f, hudText)
        }

        // --- Optional cyan guide (debug) -------------------------------------
        if (SHOW_GUIDE) {
            val mapped = RectF(offX, offY, offX + dw, offY + dh)
            canvas.drawRect(mapped, guideStroke)
        }
    }
}