package com.contextionary.sudoku

import android.graphics.*
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import android.util.Log

private const val TAG = "ImageProxyExt"

fun ImageProxy.toBitmap(): Bitmap? {
    // Convert YUV_420_888 -> NV21 correctly (handles strides)
    val nv21 = toNV21() ?: return null

    // Encode to JPEG then decode to ARGB_8888 Bitmap (simple & widely compatible)
    return try {
        val yuv = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        ByteArrayOutputStream().use { out ->
            if (!yuv.compressToJpeg(Rect(0, 0, width, height), 90, out)) {
                Log.w(TAG, "compressToJpeg failed")
                return null
            }
            val bytes = out.toByteArray()
            BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        }
    } catch (t: Throwable) {
        Log.e(TAG, "toBitmap() failed", t)
        null
    }
}

/**
 * Build a valid NV21 buffer (Y + interleaved VU) from YUV_420_888 planes.
 * This respects rowStride and pixelStride.
 */
fun ImageProxy.toNV21(): ByteArray? {
    if (format != ImageFormat.YUV_420_888) {
        Log.w(TAG, "Unsupported format=$format")
        return null
    }

    val yPlane = planes[0]
    val uPlane = planes[1]
    val vPlane = planes[2]

    // Log once (handy for debugging)
    if (BuildConfig.DEBUG) {
        Log.d(
            TAG,
            "toNV21() size=${width}x$height | " +
                    "Y(rs=${yPlane.rowStride}, ps=${yPlane.pixelStride}) " +
                    "U(rs=${uPlane.rowStride}, ps=${uPlane.pixelStride}) " +
                    "V(rs=${vPlane.rowStride}, ps=${vPlane.pixelStride})"
        )
    }

    val ySize = width * height
    val chromaSize = ySize / 2
    val out = ByteArray(ySize + chromaSize)

    // ---- Copy Y (fast path: contiguous rows if stride == width) ----
    copyPlane(
        src = yPlane.buffer,
        rowStride = yPlane.rowStride,
        pixelStride = yPlane.pixelStride,
        width = width,
        height = height,
        out = out,
        outOffset = 0,
        packAsNV21 = false // just dump Y
    )

    // ---- Interleave V and U into NV21 ----
    // NV21 expects V then U (VU VU VU …)
    interleaveVU(
        vBuffer = vPlane.buffer,
        uBuffer = uPlane.buffer,
        vRowStride = vPlane.rowStride,
        uRowStride = uPlane.rowStride,
        vPixelStride = vPlane.pixelStride,
        uPixelStride = uPlane.pixelStride,
        width = width,
        height = height,
        out = out,
        outOffset = ySize
    )

    return out
}

private fun copyPlane(
    src: ByteBuffer,
    rowStride: Int,
    pixelStride: Int,
    width: Int,
    height: Int,
    out: ByteArray,
    outOffset: Int,
    packAsNV21: Boolean
) {
    val buffer = src.duplicate() // don’t disturb original position
    var outPos = outOffset
    val rowLength = if (pixelStride == 1) width else (width - 1) * pixelStride + 1

    for (row in 0 until height) {
        if (pixelStride == 1) {
            // Fast path: 1:1 copy
            buffer.get(out, outPos, width)
            outPos += width
            buffer.position(buffer.position() + rowStride - width)
        } else {
            // Generic path: honor pixelStride
            val rowBytes = ByteArray(rowLength)
            buffer.get(rowBytes, 0, rowLength)
            var col = 0
            while (col < width) {
                out[outPos++] = rowBytes[col * pixelStride]
                col++
            }
            buffer.position(buffer.position() + rowStride - rowLength)
        }
    }
}

/**
 * Writes interleaved VU chroma into `out` starting at outOffset.
 */
private fun interleaveVU(
    vBuffer: ByteBuffer,
    uBuffer: ByteBuffer,
    vRowStride: Int,
    uRowStride: Int,
    vPixelStride: Int,
    uPixelStride: Int,
    width: Int,
    height: Int,
    out: ByteArray,
    outOffset: Int
) {
    val v = vBuffer.duplicate()
    val u = uBuffer.duplicate()

    val chromaWidth = width / 2
    val chromaHeight = height / 2
    var outPos = outOffset

    // Each chroma row has width/2 samples (one V and one U per 2x2 luma block)
    for (row in 0 until chromaHeight) {
        var vPos = v.position()
        var uPos = u.position()

        for (col in 0 until chromaWidth) {
            out[outPos++] = v.get(vPos) // V
            out[outPos++] = u.get(uPos) // U
            vPos += vPixelStride
            uPos += uPixelStride
        }

        v.position(v.position() + vRowStride)
        u.position(u.position() + uRowStride)
    }
}