package com.contextionary.sudoku

import org.opencv.core.Mat
import java.util.ArrayDeque

object ImagePools {
    private val mats = ArrayDeque<Mat>()

    @Synchronized
    fun getMat(rows: Int, cols: Int, type: Int): Mat {
        val m = mats.firstOrNull()
        if (m != null && m.rows() == rows && m.cols() == cols && m.type() == type) {
            mats.removeFirst()
            return m
        }
        return Mat(rows, cols, type)
    }

    @Synchronized
    fun recycle(mat: Mat) {
        if (!mat.empty()) mats.addLast(mat)
    }

    @Synchronized
    fun clear() {
        while (mats.isNotEmpty()) mats.removeFirst().release()
    }
}