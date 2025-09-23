package com.contextionary.sudoku

import android.graphics.ImageFormat
import android.graphics.RectF
import androidx.camera.core.ImageProxy

object GridRefiner {
    init { System.loadLibrary("native-lib") }

    external fun nativeRefine(
        nv21: ByteArray,
        width: Int, height: Int,
        left: Float, top: Float, right: Float, bottom: Float
    ): FloatArray // [x0,y0, x1,y1, x2,y2, x3,y3, conf]
}

data class Corners(val pts: FloatArray, val conf: Float)

class GridFinder(private val detector: Detector) {

    fun find(proxy: ImageProxy): Corners? {
        if (proxy.format != ImageFormat.YUV_420_888) return null
        val bmp = proxy.toBitmap() ?: return null
        val dets = detector.infer(bmp)
        if (dets.isEmpty()) return null

        // Choose center-closest grid
        val cx = bmp.width / 2f; val cy = bmp.height / 2f
        val target = dets.minBy { d ->
            val bx = d.box.centerX(); val by = d.box.centerY()
            val dx = bx - cx; val dy = by - cy
            dx*dx + dy*dy
        }
        val roi = expand(target.box, 1.10f, bmp.width.toFloat(), bmp.height.toFloat())

        val nv21 = proxy.toNV21() ?: return null
        val arr = GridRefiner.nativeRefine(nv21, bmp.width, bmp.height, roi.left, roi.top, roi.right, roi.bottom)
        if (arr.size != 9) return null
        val conf = arr[8]
        return if (conf >= 0.35f) Corners(arr.copyOfRange(0,8), conf) else null
    }

    private fun expand(r: RectF, scale: Float, w: Float, h: Float): RectF {
        val cx = r.centerX(); val cy = r.centerY()
        val hw = r.width()/2 * scale; val hh = r.height()/2 * scale
        return RectF(
            (cx-hw).coerceAtLeast(0f),
            (cy-hh).coerceAtLeast(0f),
            (cx+hw).coerceAtMost(w-1f),
            (cy+hh).coerceAtMost(h-1f)
        )
    }
}