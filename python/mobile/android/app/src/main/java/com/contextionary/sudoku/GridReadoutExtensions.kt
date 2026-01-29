package com.contextionary.sudoku

// 1-based accessor used by your logs (r=1..9, c=1..9)
fun CellGridReadout.at(row1: Int, col1: Int): CellReadout {
    require(row1 in 1..rows) { "row out of range: $row1" }
    require(col1 in 1..cols) { "col out of range: $col1" }
    return cells[row1 - 1][col1 - 1]
}

// True if every cell's solution head is blank (0)
fun CellGridReadout.isAllSolutionsBlank(): Boolean =
    cells.all { row -> row.all { it.solutionDigit == 0 } }

// True if at least one printed "given" is present (1..9)
fun CellGridReadout.hasAnyGiven(): Boolean =
    cells.any { row -> row.any { it.givenDigit in 1..9 } }