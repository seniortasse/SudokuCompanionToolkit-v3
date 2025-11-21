package com.contextionary.sudoku

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import android.view.animation.LinearInterpolator
import kotlin.math.PI
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * SudoVoiceBarsView — 4 chunky “pill” bars
 * - Bars keep a pill shape (constant corner radius) and only change height
 * - Tight gaps, no “circle morph”
 * - Smooth sine-wave with gentle start/stop
 * - Tempo can be set at runtime via setTempoMs()
 * - Bar color via setBarColor(); breathing range via setMinMax()
 */
class SudoVoiceBarsView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    // Style
    private var barCount = 4
    private var gapDp = 8f                   // tighter gaps
    private var cornerDp = 14f               // constant rounding (no circle morph)
    private var minHeightFrac = 0.35f        // calm floor (keeps “pill”, not line)
    private var maxHeightFrac = 0.90f        // tall but still pill
    private var barColor = 0xFFFFFFFF.toInt()

    // Animation
    private var tempoMs = 600L               // default pace; runtime-adapted by caller
    private var animator: ValueAnimator? = null
    private var t: Float = 0f
    private var speaking = false
    private var globalAmp = 1f               // eased amplitude 0..1
    private var ease: ValueAnimator? = null

    // Runtime
    private val density = resources.displayMetrics.density
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = barColor
    }

    // ---------- Public controls ----------
    fun setBarColor(color: Int) { barColor = color; paint.color = color; invalidate() }
    fun setTempoMs(periodMs: Long) {
        tempoMs = min(2000L, max(250L, periodMs))
        if (animator?.isRunning == true) {
            animator?.cancel()
            animator = null
            ensureAnimator(); animator?.start()
        }
    }
    fun setMinMax(minFrac: Float, maxFrac: Float) {
        minHeightFrac = min(0.9f, max(0.1f, minFrac))
        maxHeightFrac = min(0.95f, max(minHeightFrac + 0.05f, maxFrac))
        invalidate()
    }

    // ---------- Drawing ----------
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat(); val h = height.toFloat()
        if (w <= 0f || h <= 0f) return

        val gapPx = gapDp * density
        val totalGap = gapPx * (barCount - 1)
        val barW = max(4f, (w - totalGap) / barCount)         // fill available width
        val radius = cornerDp * density                        // constant corner radius
        val usable = (maxHeightFrac - minHeightFrac) * h

        // phase-offset sine waves (no random jitter)
        val base = t
        for (i in 0 until barCount) {
            val phase = base + i * (PI.toFloat() / 2.6f)
            val s = (sin(phase) + 1f) * 0.5f                   // 0..1
            val frac = minHeightFrac + usable / h * s * globalAmp
            val barH = frac * h

            val left = i * (barW + gapPx)
            val right = left + barW
            val top = (h - barH) / 2f
            val bottom = top + barH

            canvas.drawRoundRect(left, top, right, bottom, radius, radius, paint)
        }
    }

    // ---------- Animation ----------
    private fun ensureAnimator() {
        if (animator != null) return
        animator = ValueAnimator.ofFloat(0f, (2f * PI).toFloat()).apply {
            duration = tempoMs
            repeatCount = ValueAnimator.INFINITE
            repeatMode = ValueAnimator.RESTART
            interpolator = LinearInterpolator()
            addUpdateListener {
                t = it.animatedValue as Float
                invalidate()
            }
        }
    }

    private fun easeTo(target: Float, dur: Long = 180L) {
        ease?.cancel()
        ease = ValueAnimator.ofFloat(globalAmp, target).apply {
            duration = dur
            interpolator = LinearInterpolator()
            addUpdateListener { a ->
                globalAmp = a.animatedValue as Float
                invalidate()
            }
        }
        ease?.start()
    }

    fun startSpeaking() {
        if (speaking) return
        speaking = true
        visibility = VISIBLE
        ensureAnimator()
        animator?.start()
        easeTo(1f, 220L) // quick ramp up
    }

    fun stopSpeaking() {
        speaking = false
        easeTo(0.2f, 200L) // gentle decay
        postDelayed({ if (!speaking) animator?.cancel() }, 260L)
    }

    override fun onDetachedFromWindow() {
        animator?.cancel()
        ease?.cancel()
        super.onDetachedFromWindow()
    }
}