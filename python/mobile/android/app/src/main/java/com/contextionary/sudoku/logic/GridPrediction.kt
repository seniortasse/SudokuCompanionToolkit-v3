package com.contextionary.sudoku.logic

/**
 * Flat representation of a 9×9 Sudoku prediction.
 *
 * - digits: 81-length IntArray (row-major, 0 = empty)
 * - confidences: 81-length FloatArray (0.0f..1.0f)
 * - avgConfidence: mean of confidences
 * - lowConfidenceCount: how many cells are below [lowConfThreshold] at creation time
 * - lowConfidenceIndices: indices [0..80] of those low-confidence cells (row * 9 + col)
 */
data class GridPrediction(
    val digits: IntArray,
    val confidences: FloatArray,
    val avgConfidence: Float,
    val lowConfidenceCount: Int,
    val lowConfidenceIndices: IntArray
) {
    init {
        require(digits.size == 81) { "GridPrediction: digits must have size 81" }
        require(confidences.size == 81) { "GridPrediction: confidences must have size 81" }
    }

    companion object {
        /**
         * Build from flat arrays with a given low-confidence threshold.
         *
         * Example:
         *   GridPrediction.fromFlat(digits, confs, lowConfThreshold = 0.60f)
         */
        fun fromFlat(
            digits: IntArray,
            confidences: FloatArray,
            lowConfThreshold: Float = 0.60f
        ): GridPrediction {
            require(digits.size == 81) { "digits must have size 81" }
            require(confidences.size == 81) { "confidences must have size 81" }

            val avg = confidences.average().toFloat()

            val lowIndices = mutableListOf<Int>()
            for (i in confidences.indices) {
                if (confidences[i] < lowConfThreshold) {
                    lowIndices.add(i)
                }
            }

            return GridPrediction(
                digits = digits.copyOf(),
                confidences = confidences.copyOf(),
                avgConfidence = avg,
                lowConfidenceCount = lowIndices.size,
                lowConfidenceIndices = lowIndices.toIntArray()
            )
        }
    }

    /** Convenience helpers for row/col access. */
    fun digitAt(row: Int, col: Int): Int = digits[row * 9 + col]
    fun confidenceAt(row: Int, col: Int): Float = confidences[row * 9 + col]
}