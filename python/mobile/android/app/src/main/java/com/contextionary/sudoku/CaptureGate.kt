package com.contextionary.sudoku

import android.graphics.PointF
import android.graphics.RectF
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

data class CornerPrediction(
    val tl128: PointF, val tr128: PointF, val br128: PointF, val bl128: PointF,
    val tlPeak: Float, val trPeak: Float, val brPeak: Float, val blPeak: Float,
    // optional: peak positions (argmax) for “soft/argmax agreement”
    val tlArgmax128: PointF? = null, val trArgmax128: PointF? = null,
    val brArgmax128: PointF? = null, val blArgmax128: PointF? = null
)

data class LockedGrid(
    val roi: RectF,                 // detector ROI (bitmap coords)
    val tlBmp: PointF, val trBmp: PointF, val brBmp: PointF, val blBmp: PointF
)

class CaptureGate(
    private val w128: Float = 128f,
    private val h128: Float = 128f,
    var peakAllThresh: Float = 0.92f,
    var minAreaRatioVsDetector: Float = 0.90f,   // your rule
    var maxAreaRatioVsDetector: Float = 1.20f,   // extra guard
    var maxSideLenRatio: Float = 1.8f,
    var maxSoftVsArgmaxPx: Float = 4f,           // in 128 space
    var stableFramesNeeded: Int = 3,
    var maxJitterPx128: Float = 3f               // avg per-corner over window
) {
    private data class Hist(val pts128: List<PointF>)
    private val history = ArrayDeque<Hist>()

    private fun polyArea(p: List<PointF>): Float {
        var a = 0.0
        for (i in p.indices) {
            val j = (i + 1) % p.size
            a += p[i].x * p[j].y - p[j].x * p[i].y
        }
        return (a * 0.5f).toFloat()
    }
    private fun isConvexQuadCW(p: List<PointF>): Boolean {
        if (p.size != 4) return false
        val z1 = cross(p[0], p[1], p[2]); val z2 = cross(p[1], p[2], p[3])
        val z3 = cross(p[2], p[3], p[0]); val z4 = cross(p[3], p[0], p[1])
        val allPos = z1 > 0 && z2 > 0 && z3 > 0 && z4 > 0
        val allNeg = z1 < 0 && z2 < 0 && z3 < 0 && z4 < 0
        return allPos || allNeg
    }
    private fun cross(a: PointF, b: PointF, c: PointF) =
        (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)

    private fun dist(a: PointF, b: PointF): Float {
        val dx = a.x - b.x; val dy = a.y - b.y
        return sqrt(dx*dx + dy*dy)
    }

    private fun sideLenStats(p: List<PointF>): Pair<Float, Float> {
        val s1 = dist(p[0], p[1]); val s2 = dist(p[1], p[2])
        val s3 = dist(p[2], p[3]); val s4 = dist(p[3], p[0])
        val mx = max(max(s1, s2), max(s3, s4))
        val mn = min(min(s1, s2), min(s3, s4))
        return Pair(mx, mn)
    }

    private fun rectArea(r: RectF): Float = max(0f, r.width()) * max(0f, r.height())

    private fun avgJitterPx128(curr: List<PointF>): Float {
        if (history.isEmpty()) return 0f
        val last = history.last().pts128
        val d = listOf(
            dist(curr[0], last[0]), dist(curr[1], last[1]),
            dist(curr[2], last[2]), dist(curr[3], last[3])
        )
        return (d[0] + d[1] + d[2] + d[3]) * 0.25f
    }

    private fun softVsArgmaxOk(pred: CornerPrediction): Boolean {
        fun ok(s: PointF, a: PointF?): Boolean {
            if (a == null) return true
            val dx = abs(s.x - a.x); val dy = abs(s.y - a.y)
            return dx <= maxSoftVsArgmaxPx && dy <= maxSoftVsArgmaxPx
        }
        return ok(pred.tl128, pred.tlArgmax128) &&
                ok(pred.tr128, pred.trArgmax128) &&
                ok(pred.br128, pred.brArgmax128) &&
                ok(pred.bl128, pred.blArgmax128)
    }

    /**
     * Decide whether to lock this frame.
     * @param pred   corners & peaks in 128×128 space
     * @param roi    detector rect in bitmap coords
     * @param map128toBmp  function that maps a 128×128 point → bitmap space
     */
    fun maybeLock(
        pred: CornerPrediction,
        roi: RectF,
        map128toBmp: (PointF) -> PointF
    ): LockedGrid? {
        // 1) all-four confidence
        val minPeak = min(min(pred.tlPeak, pred.trPeak), min(pred.brPeak, pred.blPeak))
        if (minPeak < peakAllThresh) {
            history.clear(); return null
        }

        val quad = listOf(pred.tl128, pred.tr128, pred.br128, pred.bl128)

        // 2) convex + ordered + positive area
        val area128 = polyArea(quad)
        if (area128 <= 1e-3f || !isConvexQuadCW(quad)) {
            history.clear(); return null
        }

        // 3) side length sanity
        val (sMax, sMin) = sideLenStats(quad)
        if (sMin < 1e-3f || sMax / sMin > maxSideLenRatio) {
            history.clear(); return null
        }

        // 4) soft vs argmax agreement
        if (!softVsArgmaxOk(pred)) {
            history.clear(); return null
        }

        // 5) area ratio vs red detector
        val areaRed = rectArea(roi)
        if (areaRed > 0f) {
            // map quad to bitmap to compare apples-to-apples
            val qb = quad.map(map128toBmp)
            val areaBmp = polyArea(qb)
            val ratio = areaBmp / areaRed
            if (ratio < minAreaRatioVsDetector || ratio > maxAreaRatioVsDetector) {
                history.clear(); return null
            }
        }

        // 6) temporal stability (push & check window)
        history.addLast(Hist(quad))
        while (history.size > stableFramesNeeded) history.removeFirst()
        if (history.size < stableFramesNeeded) return null

        // average point drift vs previous
        val jitter = avgJitterPx128(quad)
        if (jitter > maxJitterPx128) return null

        // ✔ lock — return bitmap-space corners
        val tlB = map128toBmp(pred.tl128)
        val trB = map128toBmp(pred.tr128)
        val brB = map128toBmp(pred.br128)
        val blB = map128toBmp(pred.bl128)
        return LockedGrid(roi, tlB, trB, brB, blB)
    }

    fun reset() { history.clear() }
}