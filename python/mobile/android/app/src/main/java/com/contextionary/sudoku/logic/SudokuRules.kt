package com.contextionary.sudoku.logic

/**
 * Basic Sudoku constraints checker.
 *
 * - Checks that each row has no duplicate digits (1..9)
 * - Checks that each column has no duplicate digits
 * - Checks that each 3x3 box has no duplicate digits
 *
 * digit = 0 is treated as empty and ignored.
 */
object SudokuRules {

    private const val SIZE = 9
    private const val BOX_SIZE = 3

    private fun idx(row: Int, col: Int) = row * SIZE + col

    /** Check a single row. */
    fun isRowValid(digits: IntArray, row: Int): Boolean {
        var mask = 0
        for (col in 0 until SIZE) {
            val v = digits[idx(row, col)]
            if (v == 0) continue
            val bit = 1 shl v
            if ((mask and bit) != 0) return false
            mask = mask or bit
        }
        return true
    }

    /** Check a single column. */
    fun isColValid(digits: IntArray, col: Int): Boolean {
        var mask = 0
        for (row in 0 until SIZE) {
            val v = digits[idx(row, col)]
            if (v == 0) continue
            val bit = 1 shl v
            if ((mask and bit) != 0) return false
            mask = mask or bit
        }
        return true
    }

    /** Check a single 3×3 box. Index 0..8. */
    fun isBoxValid(digits: IntArray, boxIndex: Int): Boolean {
        var mask = 0
        val startRow = (boxIndex / BOX_SIZE) * BOX_SIZE
        val startCol = (boxIndex % BOX_SIZE) * BOX_SIZE
        for (dr in 0 until BOX_SIZE) {
            for (dc in 0 until BOX_SIZE) {
                val v = digits[idx(startRow + dr, startCol + dc)]
                if (v == 0) continue
                val bit = 1 shl v
                if ((mask and bit) != 0) return false
                mask = mask or bit
            }
        }
        return true
    }

    /** Check all rows, columns, and boxes for consistency. */
    fun isGridConsistent(digits: IntArray): Boolean {
        for (i in 0 until SIZE) {
            if (!isRowValid(digits, i)) return false
            if (!isColValid(digits, i)) return false
            if (!isBoxValid(digits, i)) return false
        }
        return true
    }
}