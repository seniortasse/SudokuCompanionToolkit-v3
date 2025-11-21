package com.contextionary.sudoku.logic

data class AutoCorrectionResult(
    val correctedGrid: SudokuGrid,
    val changedIndices: List<Int>,      // indices (0..80) where we changed the digit
    val unresolvedIndices: List<Int>,   // cells still suspicious / not auto-fixed
    val wasSolvable: Boolean,           // final grid uniquely solvable?
    val hadConflictsInitially: Boolean, // did the original grid break Sudoku rules?
    val initiallySolvable: Boolean      // was the original top-1 grid uniquely solvable?
)

/**
 * v0.8.1 auto-correction engine (unique-solution aware, zero banned in pair search):
 *
 * Inputs:
 *  - GridPrediction (flat digits + confidences + low-confidence indices)
 *  - classProbs[9][9][10] = calibrated prob per cell and per class (0..9)
 *
 * Behavior:
 *  - Stage 1 (Greedy, probability-bounded single-cell fixes)
 *      * Detect conflicts + low-confidence cells as "suspects".
 *      * For each suspect cell (up to [maxCellsToAdjust]):
 *          - Build a top-k candidate list from classProbs.
 *          - Try alternative digits (excluding the current top-1) while:
 *              - keeping Sudoku constraints valid, AND
 *              - ensuring the grid has a UNIQUE solution (via SudokuSolver.countSolutions).
 *
 *  - Stage 2 (Brute-force single-cell search on a few unresolved, low-confidence cells)
 *      * Focus on up to [fallbackMaxUnresolvedCells] low-confidence cells that remain unresolved.
 *      * For each such cell:
 *          - Try all digits 0..9 (ignoring CNN probabilities), except the current one.
 *          - Keep the first digit that makes the grid consistent AND uniquely solvable.
 *
 *  - Stage 3 (Brute-force pairs of cells)
 *      * If grid is still unsolvable and unresolved cell count is small (≤ [fallbackMaxUnresolvedCells]):
 *        consider up to [fallbackMaxUnresolvedCells] prioritized unresolved cells and
 *        brute-force all (1..9 × 1..9) combinations for each unordered pair.
 *      * Commit the first pair of digits that makes the grid consistent AND uniquely solvable.
 *      * Additionally logs whenever 0 would have been considered but is explicitly skipped.
 *
 *  - Stage 4 (Fallback combinatorial search, CNN-guided)
 *      * If grid is still unsolvable but only a small number of cells remain unresolved
 *        (≤ [fallbackMaxUnresolvedCells]), perform a combinatorial search over
 *        top-k alternatives for those cells (probability-bounded), again requiring
 *        a UNIQUE solution.
 *
 * This is intentionally conservative: we fix a *small* number of cells
 * that make the grid inconsistent or unsolvable, not rewrite the whole board,
 * and we only accept grids with a single valid completion.
 */
class SudokuAutoCorrector(
    private val solver: SudokuSolver,
    private val maxCellsToAdjust: Int = 4,
    private val maxAttempts: Int = 64,
    private val minAltProb: Float = 0.15f,   // ignore very low-probability classes
    private val maxAltPerCell: Int = 3,      // try up to 3 alternative digits per cell
    private val fallbackMaxUnresolvedCells: Int = 3  // last-chance search for ≤ N cells
)  {

    /**
     * Helper: returns true iff the grid has exactly ONE solution.
     * Implemented via SudokuSolver.countSolutions with maxCount = 2.
     */
    private fun hasUniqueSolution(grid: SudokuGrid): Boolean {
        // 0 -> unsolvable; 1 -> unique; >=2 -> multiple solutions.
        return solver.countSolutions(grid.digits, maxCount = 2) == 1
    }

    /**
     * @param prediction  flat grid summary (digits + confidences + low-confidence indices)
     * @param classProbs  [9][9][10] probabilities per cell/class from DigitClassifier
     */
    fun autoCorrect(
        prediction: GridPrediction,
        classProbs: Array<Array<FloatArray>>
    ): AutoCorrectionResult {

        // Small helper to log cell coordinates as rXcY
        fun idxToCell(idx: Int): String {
            val r = idx / 9 + 1
            val c = idx % 9 + 1
            return "r${r}c${c}"
        }

        // Base grid from top-1 predictions
        val baseGrid = SudokuGrid(prediction.digits)
        val originalDigits = baseGrid.digits.copyOf()

        val hadConflictsInitially = !SudokuRules.isGridConsistent(originalDigits)

        // Check initial unique solvability (for diagnostics only)
        val initiallySolvable = if (!hadConflictsInitially) {
            hasUniqueSolution(baseGrid)
        } else {
            false
        }

        // 1) Identify conflict cells in the original top-1 grid
        val conflictIndicesInitial = findConflictIndices(originalDigits)

        // 2) Low-confidence cells from prediction summary
        val lowConfIndicesInitial = prediction.lowConfidenceIndices.toList()

        // 3) Union of conflict + low-confidence cells
        val suspectSet = (conflictIndicesInitial + lowConfIndicesInitial).toSet().toList()
        val suspectsLimited = suspectSet.take(maxCellsToAdjust)

        // We'll mutate this working copy as we test candidate flips
        val workingDigits = originalDigits.copyOf()
        var attempts = 0

        // =========================
        // STAGE 1: GREEDY, PROB-BOUNDED SINGLE-CELL FIXES
        // =========================
        if (suspectsLimited.isNotEmpty()) {
            for (idx in suspectsLimited) {
                if (attempts >= maxAttempts) break

                val row = idx / 9
                val col = idx % 9
                val currentDigit = workingDigits[idx]

                val probs = classProbs[row][col]
                if (probs.size != 10) continue // defensive

                // Build candidate list: (digit, prob), sorted by prob desc, ignore tiny probs.
                val candidates = (0..9)
                    .map { d -> d to probs[d] }
                    .sortedByDescending { it.second }
                    .filter { (_, p) -> p >= minAltProb }

                if (candidates.isEmpty()) continue

                // Skip the current top-1 digit; we want alternative hypotheses.
                val altCandidates = candidates
                    .filter { (d, _) -> d != currentDigit }
                    .take(maxAltPerCell)

                if (altCandidates.isEmpty()) continue

                for ((altDigit, _) in altCandidates) {
                    if (attempts >= maxAttempts) break
                    attempts++

                    // Try alternative digit
                    val prev = workingDigits[idx]
                    workingDigits[idx] = altDigit

                    // Must satisfy Sudoku structural rules
                    if (!SudokuRules.isGridConsistent(workingDigits)) {
                        workingDigits[idx] = prev
                        continue
                    }

                    // Must be uniquely solvable as a Sudoku
                    val candidateGrid = SudokuGrid(workingDigits)
                    if (hasUniqueSolution(candidateGrid)) {
                        // Accept this flip
                        break
                    } else {
                        // Revert and try next candidate
                        workingDigits[idx] = prev
                    }
                }

                if (attempts >= maxAttempts) break
                // If not fixed, we leave this cell as-is; it remains "unresolved".
            }
        }

        // Assessment helper for the current workingDigits
        fun computeAssessmentForCurrentDigits(): Triple<Boolean, List<Int>, List<Int>> {
            val grid = SudokuGrid(workingDigits)
            val consistent = SudokuRules.isGridConsistent(workingDigits)
            val solvable = consistent && hasUniqueSolution(grid)  // UNIQUE solvability

            val conflicts = findConflictIndices(workingDigits)

            // "changed so far" = any index where workingDigits differ from originalDigits
            val changedSet = mutableSetOf<Int>()
            for (i in 0 until 81) {
                if (workingDigits[i] != originalDigits[i]) {
                    changedSet.add(i)
                }
            }

            val lowConf = prediction.lowConfidenceIndices.filter { it !in changedSet }

            // Return (solvable flag, conflict indices, low-confidence unresolved indices)
            return Triple(solvable, conflicts, lowConf)
        }

        var (solvableCurrent, conflictsCurrent, lowConfCurrent) =
            computeAssessmentForCurrentDigits()

        // =========================
        // STAGE 2: BRUTE-FORCE SINGLE-CELL 0..9 ON FEW LOW-CONF CELLS
        // =========================

        if (!solvableCurrent) {
            // unresolved = conflicts ∪ low-confidence
            val unresolvedAfterStage1 = (conflictsCurrent + lowConfCurrent).toSet().toList()

            if (unresolvedAfterStage1.isNotEmpty()) {
                // Focus for brute-force ONLY on low-confidence unresolved cells (bias)
                val bruteForceTargets = lowConfCurrent
                    .filter { it in unresolvedAfterStage1 }
                    .take(fallbackMaxUnresolvedCells)

                if (bruteForceTargets.isNotEmpty()) {
                    bruteLoop@ for (idx in bruteForceTargets) {
                        val prevDigit = workingDigits[idx]
                        val tried = mutableListOf<Int>()
                        var cellSuccess = false

                        for (digit in 0..9) {
                            if (digit == prevDigit) continue
                            tried.add(digit)

                            workingDigits[idx] = digit

                            // Must satisfy Sudoku rules
                            if (!SudokuRules.isGridConsistent(workingDigits)) {
                                continue
                            }

                            // Must be uniquely solvable as a Sudoku
                            val grid = SudokuGrid(workingDigits)
                            if (hasUniqueSolution(grid)) {
                                // Accept this digit and stop Stage 2 (one-cell fix success)
                                cellSuccess = true

                                // LOG: brute-force attempt for this cell
                                android.util.Log.i(
                                    "SudokuLogic",
                                    "autoCorrect: bruteForceSingle idx=${idxToCell(idx)} " +
                                            "tried=$tried chosen=$digit success=true"
                                )

                                // We keep this assignment and break out from Stage 2;
                                // further unresolved cells (if any) will be handled
                                // by later stages.
                                break@bruteLoop
                            }
                        }

                        if (!cellSuccess) {
                            // Restore previous digit if no candidate worked
                            workingDigits[idx] = prevDigit

                            android.util.Log.i(
                                "SudokuLogic",
                                "autoCorrect: bruteForceSingle idx=${idxToCell(idx)} " +
                                        "tried=$tried chosen=$prevDigit success=false"
                            )
                        }
                    }

                    // Re-assess after brute-force Stage 2
                    val assessment2 = computeAssessmentForCurrentDigits()
                    solvableCurrent = assessment2.first
                    conflictsCurrent = assessment2.second
                    lowConfCurrent = assessment2.third
                }
            }
        }

        // =========================
        // STAGE 3: BRUTE-FORCE PAIRS OF CELLS (digits 1..9 only)
        // =========================

        if (!solvableCurrent) {
            val unresolvedAfterStage2 =
                (conflictsCurrent + lowConfCurrent).toSet().toList()

            if (unresolvedAfterStage2.size > 1) {
                // Prioritize low-confidence cells first, then pure-conflict cells
                val prioritized = unresolvedAfterStage2.sortedBy { idx ->
                    if (idx in lowConfCurrent) 0 else 1
                }

                val pairCells = prioritized.take(fallbackMaxUnresolvedCells)

                var pairSuccess = false
                var totalPairTried = 0

                // Track where we explicitly skipped 0 in pair search
                val zeroSkippedIndices = mutableSetOf<Int>()

                outerPair@ for (i in 0 until pairCells.size) {
                    for (j in i + 1 until pairCells.size) {
                        val idx1 = pairCells[i]
                        val idx2 = pairCells[j]

                        val prev1 = workingDigits[idx1]
                        val prev2 = workingDigits[idx2]

                        // Try all combinations of digits for this pair
                        for (d1 in 0..9) {
                            // In pair search, we forbid 0 as a candidate.
                            if (d1 == 0) {
                                zeroSkippedIndices.add(idx1)
                                continue
                            }
                            if (d1 == prev1) continue

                            workingDigits[idx1] = d1

                            for (d2 in 0..9) {
                                // For the second cell, also forbid 0.
                                if (d2 == 0) {
                                    zeroSkippedIndices.add(idx2)
                                    continue
                                }
                                if (d2 == prev2) continue

                                workingDigits[idx2] = d2

                                // Now check Sudoku rules on the pair together
                                if (!SudokuRules.isGridConsistent(workingDigits)) {
                                    // Revert idx2 and continue with next d2
                                    workingDigits[idx2] = prev2
                                    continue
                                }

                                totalPairTried++

                                val grid = SudokuGrid(workingDigits)
                                if (hasUniqueSolution(grid)) {
                                    pairSuccess = true

                                    android.util.Log.i(
                                        "SudokuLogic",
                                        "autoCorrect: bruteForcePair cells=[${idxToCell(idx1)},${idxToCell(idx2)}] " +
                                                "chosen=[$d1,$d2] triedCount=$totalPairTried success=true"
                                    )

                                    // Keep this pair assignment and stop Stage 3 entirely
                                    break@outerPair
                                }

                                // revert idx2 before trying next d2
                                workingDigits[idx2] = prev2
                            }

                            if (pairSuccess) break

                            // revert idx1 before trying next d1
                            workingDigits[idx1] = prev1
                        }

                        if (!pairSuccess) {
                            // Ensure both are restored if this pair did not yield success
                            workingDigits[idx1] = prev1
                            workingDigits[idx2] = prev2
                        }
                    }
                }

                if (!pairSuccess && totalPairTried > 0) {
                    android.util.Log.i(
                        "SudokuLogic",
                        "autoCorrect: bruteForcePair noSolution triedCount=$totalPairTried success=false"
                    )
                }

                // Log where 0 was explicitly skipped in pair search (for diagnostics)
                if (zeroSkippedIndices.isNotEmpty()) {
                    android.util.Log.i(
                        "SudokuLogic",
                        "autoCorrect: bruteForcePair zeroSkipped=true cells=${zeroSkippedIndices.map { idxToCell(it) }}"
                    )
                }

                // Re-assess after Stage 3 (pair brute force), in case we found a fix
                val assessment3 = computeAssessmentForCurrentDigits()
                solvableCurrent = assessment3.first
                conflictsCurrent = assessment3.second
                lowConfCurrent = assessment3.third
            }
        }

        // =========================
        // STAGE 4: FALLBACK COMBINATORIAL SEARCH (TOP-K PER CELL, CNN-GUIDED)
        // =========================

        // unresolved after Stage 3
        var unresolvedForFallback =
            (conflictsCurrent + lowConfCurrent).toSet().toList()

        if (!solvableCurrent &&
            unresolvedForFallback.isNotEmpty() &&
            unresolvedForFallback.size <= fallbackMaxUnresolvedCells
        ) {
            val unresolvedList = unresolvedForFallback.toList()

            // LOG: candidate for fallback
            android.util.Log.i(
                "SudokuLogic",
                "autoCorrect: fallbackCandidate unresolved=${unresolvedList.map { idxToCell(it) }} count=${unresolvedList.size}"
            )

            // Build candidate digits per unresolved cell, using CNN probabilities.
            fun buildAltCandidates(idx: Int): List<Int> {
                val r = idx / 9
                val c = idx % 9
                val currentDigit = workingDigits[idx]
                val probs = classProbs[r][c]
                if (probs.size != 10) return emptyList()

                val cand = (0..9)
                    .map { d -> d to probs[d] }
                    .sortedByDescending { it.second }
                    .filter { (_, p) -> p >= minAltProb }
                    .filter { (d, _) -> d != currentDigit }
                    .take(maxAltPerCell)

                return cand.map { it.first }
            }

            // "changedBeforeFallback" = indices that already differ from originalDigits
            val changedBeforeFallback = mutableSetOf<Int>()
            for (i in 0 until 81) {
                if (workingDigits[i] != originalDigits[i]) {
                    changedBeforeFallback.add(i)
                }
            }

            val unresolvedBefore = unresolvedList.size

            // Collect only cells that actually have alternative candidates.
            val altIndices = mutableListOf<Int>()
            val altDigitsPerIndex = mutableListOf<List<Int>>()
            for (idx in unresolvedList) {
                val alts = buildAltCandidates(idx)
                if (alts.isNotEmpty()) {
                    altIndices.add(idx)
                    altDigitsPerIndex.add(alts)
                }
            }

            var fallbackApplied = false

            if (altIndices.isNotEmpty()) {
                // Recursive DFS over combinations of alt digits for these cells
                fun dfs(depth: Int): Boolean {
                    if (depth == altIndices.size) {
                        // At leaf: workingDigits has a full candidate assignment
                        val g = SudokuGrid(workingDigits)
                        return hasUniqueSolution(g)
                    }

                    val idx = altIndices[depth]
                    val prev = workingDigits[idx]
                    val candidates = altDigitsPerIndex[depth]

                    for (alt in candidates) {
                        workingDigits[idx] = alt

                        // Prune early if this partial assignment breaks Sudoku rules
                        if (!SudokuRules.isGridConsistent(workingDigits)) {
                            continue
                        }

                        if (dfs(depth + 1)) {
                            // Keep this assignment (do not revert) and bubble success up
                            return true
                        }
                    }

                    // No candidate worked for this index → revert and signal failure at this level
                    workingDigits[idx] = prev
                    return false
                }

                fallbackApplied = dfs(0)

                if (fallbackApplied) {
                    // All changes (Stage 1 + Stage 2 + Stage 3 + Stage 4) reflected in workingDigits
                    // We'll recompute "changed" later for the final result.
                }
            }

            // Compute unresolved AFTER fallback, for logging only
            val changedAfterFallback = mutableSetOf<Int>()
            for (i in 0 until 81) {
                if (workingDigits[i] != originalDigits[i]) {
                    changedAfterFallback.add(i)
                }
            }

            val conflictsAfter = findConflictIndices(workingDigits)
            val lowConfAfter = prediction.lowConfidenceIndices.filter { it !in changedAfterFallback }
            val unresolvedAfter = (conflictsAfter + lowConfAfter).toSet().size

            val changedDelta = changedAfterFallback.minus(changedBeforeFallback).map { idxToCell(it) }

            android.util.Log.i(
                "SudokuLogic",
                "autoCorrect: fallbackResult success=$fallbackApplied " +
                        "changedDelta=$changedDelta unresolvedBefore=$unresolvedBefore unresolvedAfter=$unresolvedAfter"
            )

            // Update our assessment based on possible fallback changes
            val assessment4 = computeAssessmentForCurrentDigits()
            solvableCurrent = assessment4.first
            conflictsCurrent = assessment4.second
            lowConfCurrent = assessment4.third

            unresolvedForFallback =
                (conflictsCurrent + lowConfCurrent).toSet().toList()
        }

        // =========================
        // FINAL ASSESSMENT & RETURN
        // =========================
        val finalGrid = SudokuGrid(workingDigits)
        val finalConsistent = SudokuRules.isGridConsistent(workingDigits)
        val finalSolvable = finalConsistent && hasUniqueSolution(finalGrid)

        val finalConflicts = findConflictIndices(workingDigits)

        // Final "changed" set = any index differing from originalDigits
        val finalChanged = mutableListOf<Int>()
        val finalChangedSet = mutableSetOf<Int>()
        for (i in 0 until 81) {
            if (workingDigits[i] != originalDigits[i]) {
                finalChanged.add(i)
                finalChangedSet.add(i)
            }
        }

        val finalLowConf =
            prediction.lowConfidenceIndices.filter { it !in finalChangedSet }

        // If the final grid is uniquely solvable, treat ALL cells as resolved.
        val unresolved = if (finalSolvable) {
            emptyList()
        } else {
            (finalConflicts + finalLowConf).toSet().toList()
        }

        return AutoCorrectionResult(
            correctedGrid = finalGrid,
            changedIndices = finalChanged,
            unresolvedIndices = unresolved,
            wasSolvable = finalSolvable,
            hadConflictsInitially = hadConflictsInitially,
            initiallySolvable = initiallySolvable
        )
    }

    /**
     * Finds indices (0..80) of cells that participate in any row/col/box conflict.
     */
    private fun findConflictIndices(digits: IntArray): List<Int> {
        val conflicts = mutableSetOf<Int>()

        // Rows
        for (row in 0 until 9) {
            val seen = mutableMapOf<Int, MutableList<Int>>() // digit -> indices
            for (col in 0 until 9) {
                val idx = row * 9 + col
                val v = digits[idx]
                if (v == 0) continue
                val list = seen.getOrPut(v) { mutableListOf() }
                list.add(idx)
            }
            for ((_, idxList) in seen) {
                if (idxList.size > 1) conflicts.addAll(idxList)
            }
        }

        // Columns
        for (col in 0 until 9) {
            val seen = mutableMapOf<Int, MutableList<Int>>()
            for (row in 0 until 9) {
                val idx = row * 9 + col
                val v = digits[idx]
                if (v == 0) continue
                val list = seen.getOrPut(v) { mutableListOf() }
                list.add(idx)
            }
            for ((_, idxList) in seen) {
                if (idxList.size > 1) conflicts.addAll(idxList)
            }
        }

        // Boxes
        for (boxRow in 0 until 3) {
            for (boxCol in 0 until 3) {
                val seen = mutableMapOf<Int, MutableList<Int>>()
                val startRow = boxRow * 3
                val startCol = boxCol * 3
                for (dr in 0 until 3) {
                    for (dc in 0 until 3) {
                        val row = startRow + dr
                        val col = startCol + dc
                        val idx = row * 9 + col
                        val v = digits[idx]
                        if (v == 0) continue
                        val list = seen.getOrPut(v) { mutableListOf() }
                        list.add(idx)
                    }
                }
                for ((_, idxList) in seen) {
                    if (idxList.size > 1) conflicts.addAll(idxList)
                }
            }
        }

        return conflicts.toList()
    }
}