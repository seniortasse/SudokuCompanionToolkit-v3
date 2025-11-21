package com.contextionary.sudoku.logic

/**
 * Canonical 9x9 Sudoku grid.
 *
 * Conventions:
 * - digit = 0 means empty.
 * - digits 1..9 are valid Sudoku values.
 * - Stored row-major: index = row * 9 + col
 */
data class SudokuGrid(
    val digits: IntArray
) {
    init {
        require(digits.size == 81) { "SudokuGrid must have exactly 81 cells" }
    }

    fun get(row: Int, col: Int): Int = digits[row * 9 + col]

    fun withValue(row: Int, col: Int, value: Int): SudokuGrid {
        val copy = digits.copyOf()
        copy[row * 9 + col] = value
        return SudokuGrid(copy)
    }

    companion object {
        fun fromFlat(flat: IntArray): SudokuGrid = SudokuGrid(flat.copyOf())
    }
}