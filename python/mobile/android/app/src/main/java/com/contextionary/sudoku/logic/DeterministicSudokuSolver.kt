package com.contextionary.sudoku.logic

/**
 * Deterministic backtracking solver that:
 * - Finds ONE solution
 * - Counts solutions up to maxCount (default 2)
 * Used to deduce solution from GIVENS ONLY, deterministically (no LLM).
 */
object DeterministicSudokuSolver {

    data class SolveResult(
        val solutionCount: Int,      // 0,1,2 (capped)
        val solutionGrid: IntArray?  // non-null when solutionCount >= 1
    )

    fun solveCountCapped(grid81: IntArray, maxCount: Int = 2): SolveResult {
        require(grid81.size == 81)
        val grid = grid81.copyOf()
        var count = 0
        var firstSolution: IntArray? = null

        fun isValid(idx: Int, v: Int): Boolean {
            val r = idx / 9
            val c = idx % 9
            // row/col
            for (i in 0 until 9) {
                val ri = r * 9 + i
                val ci = i * 9 + c
                if (i != c && grid[ri] == v) return false
                if (i != r && grid[ci] == v) return false
            }
            // box
            val br = (r / 3) * 3
            val bc = (c / 3) * 3
            for (rr in br until br + 3) {
                for (cc in bc until bc + 3) {
                    val j = rr * 9 + cc
                    if (j != idx && grid[j] == v) return false
                }
            }
            return true
        }

        fun findNextEmpty(): Int {
            for (i in 0 until 81) if (grid[i] == 0) return i
            return -1
        }

        fun dfs() {
            if (count >= maxCount) return
            val idx = findNextEmpty()
            if (idx == -1) {
                count++
                if (firstSolution == null) firstSolution = grid.copyOf()
                return
            }

            for (v in 1..9) {
                if (!isValid(idx, v)) continue
                grid[idx] = v
                dfs()
                grid[idx] = 0
                if (count >= maxCount) return
            }
        }

        // Validate givens quickly: if any conflict exists, return 0
        for (i in 0 until 81) {
            val v = grid[i]
            if (v == 0) continue
            grid[i] = 0
            if (!isValid(i, v)) {
                grid[i] = v
                return SolveResult(solutionCount = 0, solutionGrid = null)
            }
            grid[i] = v
        }

        dfs()

        return SolveResult(
            solutionCount = count,
            solutionGrid = firstSolution
        )
    }
}