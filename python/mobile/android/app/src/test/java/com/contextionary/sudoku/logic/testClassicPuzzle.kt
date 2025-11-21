package com.contextionary.sudoku.logic

import org.junit.Test
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue

class SudokuSolverTest {

    @Test
    fun testClassicPuzzle() {
        val gridDigits = intArrayOf(
            5,3,0, 0,7,0, 0,0,0,
            6,0,0, 1,9,5, 0,0,0,
            0,9,8, 0,0,0, 0,6,0,

            8,0,0, 0,6,0, 0,0,3,
            4,0,0, 8,0,3, 0,0,1,
            7,0,0, 0,2,0, 0,0,6,

            0,6,0, 0,0,0, 2,8,0,
            0,0,0, 4,1,9, 0,0,5,
            0,0,0, 0,8,0, 0,7,9
        )

        val grid = SudokuGrid(gridDigits)
        val solver = SudokuSolver()

        val solved = solver.solve(grid)
        println("Solved? ${solved != null}")
        println(solved?.digits?.joinToString())

        assertNotNull("Solver should find a solution for the classic puzzle", solved)
    }

    @Test
    fun testAutoCorrector_noChangeOnValidGrid() {
        // A solved grid, already valid
        val solvedDigits = intArrayOf(
            5,3,4, 6,7,8, 9,1,2,
            6,7,2, 1,9,5, 3,4,8,
            1,9,8, 3,4,2, 5,6,7,

            8,5,9, 7,6,1, 4,2,3,
            4,2,6, 8,5,3, 7,9,1,
            7,1,3, 9,2,4, 8,5,6,

            9,6,1, 5,3,7, 2,8,4,
            2,8,7, 4,1,9, 6,3,5,
            3,4,5, 2,8,6, 1,7,9
        )

        // All cells high confidence
        val confidences = FloatArray(81) { 0.99f }

        // Build GridPrediction using the flat helper
        val prediction = GridPrediction.fromFlat(
            digits = solvedDigits,
            confidences = confidences,
            lowConfThreshold = 0.60f
        )

        // Build dummy full classProbs: for each cell, put 0.99 on the correct digit.
        val classProbs = Array(9) { r ->
            Array(9) { c ->
                FloatArray(10).also { arr ->
                    val d = solvedDigits[r * 9 + c]
                    arr[d] = 0.99f
                }
            }
        }

        val solver = SudokuSolver()
        val auto = SudokuAutoCorrector(solver)

        val result = auto.autoCorrect(prediction, classProbs)

        assertTrue("Solved grid should be solvable", result.wasSolvable)
        assertTrue("Auto-corrector should not change a valid solved grid", result.changedIndices.isEmpty())
        assertTrue("No unresolved cells expected on a valid solved grid", result.unresolvedIndices.isEmpty())
    }
}