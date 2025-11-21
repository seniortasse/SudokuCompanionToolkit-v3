package com.contextionary.sudoku.logic

/**
 * Complete backtracking Sudoku solver with MRV and bitmasks.
 *
 * Conventions:
 * - Grid is always 9x9.
 * - digit = 0 means empty.
 * - digits 1..9 are valid values.
 *
 * Public API:
 *  - hasSolution(digits): Boolean
 *  - solve(digits): Boolean              // fills digits with one valid solution
 *  - countSolutions(digits, maxCount): Int
 */
class SudokuSolver(
    /**
     * Default maximum number of solutions to search for when using countSolutions()
     * without an explicit maxCount argument.
     */
    private val maxSolutionsToCheck: Int = 1
) {
    companion object {
        // Bitmask with 9 bits set (one per digit 1..9)
        private const val FULL_MASK: Int = (1 shl 9) - 1
    }

    /**
     * Return true if there is at least one solution for the given grid.
     *
     * Does NOT modify the input array.
     */
    fun hasSolution(digits: IntArray): Boolean {
        if (digits.size != 81) return false

        val work = digits.copyOf()
        val rowMask = IntArray(9)
        val colMask = IntArray(9)
        val boxMask = IntArray(9)

        if (!initializeMasks(work, rowMask, colMask, boxMask)) {
            // Initial grid is structurally invalid.
            return false
        }

        return solveInternal(work, rowMask, colMask, boxMask)
    }

    /**
     * Solve the Sudoku in-place.
     *
     * @return true if a solution was found. If true, [digits] is filled with a valid
     *         complete solution. If false, [digits] is left unchanged.
     */
    fun solve(digits: IntArray): Boolean {
        if (digits.size != 81) return false

        val work = digits.copyOf()
        val rowMask = IntArray(9)
        val colMask = IntArray(9)
        val boxMask = IntArray(9)

        if (!initializeMasks(work, rowMask, colMask, boxMask)) {
            return false
        }

        val solved = solveInternal(work, rowMask, colMask, boxMask)
        if (!solved) {
            return false
        }

        // Copy the solved grid back into the caller's array.
        for (i in 0 until 81) {
            digits[i] = work[i]
        }
        return true
    }

    /**
     * Count how many solutions exist, up to [maxCount].
     *
     * This is used by the auto-correction engine to enforce UNIQUE solutions
     * (e.g., require countSolutions(grid, 2) == 1).
     *
     * Does NOT modify the input array.
     */
    fun countSolutions(digits: IntArray, maxCount: Int = maxSolutionsToCheck): Int {
        if (digits.size != 81) return 0
        if (maxCount <= 0) return 0

        val work = digits.copyOf()
        val rowMask = IntArray(9)
        val colMask = IntArray(9)
        val boxMask = IntArray(9)

        if (!initializeMasks(work, rowMask, colMask, boxMask)) {
            // Structurally invalid grid: no solutions.
            return 0
        }

        return countSolutionsInternal(work, rowMask, colMask, boxMask, maxCount)
    }

    // ------------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------------

    /**
     * Map (row, col) to flat index.
     */
    private fun idx(row: Int, col: Int): Int = row * 9 + col

    /**
     * Map (row, col) to 3x3 box index (0..8).
     */
    private fun boxIndex(row: Int, col: Int): Int = (row / 3) * 3 + (col / 3)

    /**
     * Initialize row/column/box bitmasks from the current digits.
     *
     * @return false if the initial grid contains structural conflicts
     *         (duplicate non-zero digit in any row/col/box).
     */
    private fun initializeMasks(
        digits: IntArray,
        rowMask: IntArray,
        colMask: IntArray,
        boxMask: IntArray
    ): Boolean {
        for (index in 0 until 81) {
            val d = digits[index]
            if (d == 0) continue

            if (d < 0 || d > 9) return false // invalid digit

            val row = index / 9
            val col = index % 9
            val box = boxIndex(row, col)

            val bit = 1 shl (d - 1)

            // Conflict if bit is already set in any mask.
            if ((rowMask[row] and bit) != 0 ||
                (colMask[col] and bit) != 0 ||
                (boxMask[box] and bit) != 0
            ) {
                return false
            }

            rowMask[row] = rowMask[row] or bit
            colMask[col] = colMask[col] or bit
            boxMask[box] = boxMask[box] or bit
        }
        return true
    }

    /**
     * Internal backtracking solver that fills 'digits' in-place.
     *
     * Uses MRV (minimum remaining values) heuristic on each recursive step:
     *  - among all empty cells, choose the one with the fewest candidates.
     */
    private fun solveInternal(
        digits: IntArray,
        rowMask: IntArray,
        colMask: IntArray,
        boxMask: IntArray
    ): Boolean {
        // Select the empty cell with the fewest candidates (MRV).
        var bestIndex = -1
        var bestCount = Int.MAX_VALUE
        var bestMask = 0

        for (index in 0 until 81) {
            if (digits[index] != 0) continue

            val row = index / 9
            val col = index % 9
            val box = boxIndex(row, col)

            val used = rowMask[row] or colMask[col] or boxMask[box]
            val candidatesMask = FULL_MASK and used.inv()

            if (candidatesMask == 0) {
                // No valid digit can be placed here -> dead branch.
                return false
            }

            val count = Integer.bitCount(candidatesMask)
            if (count < bestCount) {
                bestCount = count
                bestIndex = index
                bestMask = candidatesMask

                if (bestCount == 1) break // can't do better than 1 candidate
            }
        }

        if (bestIndex == -1) {
            // No empty cells -> solved.
            return true
        }

        val row = bestIndex / 9
        val col = bestIndex % 9
        val box = boxIndex(row, col)

        var candidatesMask = bestMask
        while (candidatesMask != 0) {
            val bit = candidatesMask and -candidatesMask // lowest-set bit
            val digit = Integer.numberOfTrailingZeros(bit) + 1

            // Place digit
            digits[bestIndex] = digit
            rowMask[row] = rowMask[row] or bit
            colMask[col] = colMask[col] or bit
            boxMask[box] = boxMask[box] or bit

            if (solveInternal(digits, rowMask, colMask, boxMask)) {
                return true
            }

            // Backtrack
            digits[bestIndex] = 0
            rowMask[row] = rowMask[row] and bit.inv()
            colMask[col] = colMask[col] and bit.inv()
            boxMask[box] = boxMask[box] and bit.inv()

            candidatesMask = candidatesMask and (candidatesMask - 1)
        }

        return false
    }

    /**
     * Count how many solutions exist, up to [maxCount].
     *
     * This uses the same MRV strategy as [solveInternal] but accumulates
     * the number of complete solutions instead of stopping at the first one.
     */
    private fun countSolutionsInternal(
        digits: IntArray,
        rowMask: IntArray,
        colMask: IntArray,
        boxMask: IntArray,
        maxCount: Int
    ): Int {
        if (maxCount <= 0) return 0

        // Select the empty cell with the fewest candidates (MRV).
        var bestIndex = -1
        var bestCount = Int.MAX_VALUE
        var bestMask = 0

        for (index in 0 until 81) {
            if (digits[index] != 0) continue

            val row = index / 9
            val col = index % 9
            val box = boxIndex(row, col)

            val used = rowMask[row] or colMask[col] or boxMask[box]
            val candidatesMask = FULL_MASK and used.inv()

            if (candidatesMask == 0) {
                // No valid digit can be placed here -> dead branch.
                return 0
            }

            val count = Integer.bitCount(candidatesMask)
            if (count < bestCount) {
                bestCount = count
                bestIndex = index
                bestMask = candidatesMask

                if (bestCount == 1) break
            }
        }

        if (bestIndex == -1) {
            // No empty cells -> one complete solution.
            return 1
        }

        val row = bestIndex / 9
        val col = bestIndex % 9
        val box = boxIndex(row, col)

        var totalSolutions = 0
        var candidatesMask = bestMask

        while (candidatesMask != 0 && totalSolutions < maxCount) {
            val bit = candidatesMask and -candidatesMask
            val digit = Integer.numberOfTrailingZeros(bit) + 1

            // Place digit
            digits[bestIndex] = digit
            rowMask[row] = rowMask[row] or bit
            colMask[col] = colMask[col] or bit
            boxMask[box] = boxMask[box] or bit

            // Recurse, but don't exceed maxCount
            val remaining = maxCount - totalSolutions
            val found = countSolutionsInternal(digits, rowMask, colMask, boxMask, remaining)
            totalSolutions += found

            // Backtrack
            digits[bestIndex] = 0
            rowMask[row] = rowMask[row] and bit.inv()
            colMask[col] = colMask[col] and bit.inv()
            boxMask[box] = boxMask[box] and bit.inv()

            candidatesMask = candidatesMask and (candidatesMask - 1)
        }

        return totalSolutions
    }
}
