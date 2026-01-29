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
 *  - classProbs[9][9][10] = probability distribution per cell (0..9)
 *      IMPORTANT: in the 3-head world, this should be the probs of the CHOSEN placed head
 *      (Given vs Solution), not a heuristic and not a mix.
 *
 * Behavior:
 *  - Stage 1 (Greedy, probability-bounded single-cell fixes)
 *  - Stage 2 (Brute-force single-cell search)
 *  - Stage 3 (Brute-force pairs)
 *  - Stage 4 (Fallback combinatorial top-k search)
 */
class SudokuAutoCorrector(
    private val solver: SudokuSolver,
    private val maxCellsToAdjust: Int = 4,
    private val maxAttempts: Int = 64,
    private val minAltProb: Float = 0.02f,   // ✅ default lowered to match your wiring
    private val maxAltPerCell: Int = 3,
    private val fallbackMaxUnresolvedCells: Int = 3
)  {

    private fun hasUniqueSolution(grid: SudokuGrid): Boolean {
        return solver.countSolutions(grid.digits, maxCount = 2) == 1
    }

    fun autoCorrect(
        prediction: GridPrediction,
        classProbs: Array<Array<FloatArray>>
    ): AutoCorrectionResult {

        fun idxToCell(idx: Int): String {
            val r = idx / 9 + 1
            val c = idx % 9 + 1
            return "r${r}c${c}"
        }

        val baseGrid = SudokuGrid(prediction.digits)
        val originalDigits = baseGrid.digits.copyOf()

        val hadConflictsInitially = !SudokuRules.isGridConsistent(originalDigits)

        val initiallySolvable = if (!hadConflictsInitially) {
            hasUniqueSolution(baseGrid)
        } else {
            false
        }

        val conflictIndicesInitial = findConflictIndices(originalDigits)
        val lowConfIndicesInitial = prediction.lowConfidenceIndices.toList()

        val suspectSet = (conflictIndicesInitial + lowConfIndicesInitial).toSet().toList()
        val suspectsLimited = suspectSet.take(maxCellsToAdjust)

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
                if (probs.size != 10) continue

                val candidates = (0..9)
                    .map { d -> d to probs[d] }
                    .sortedByDescending { it.second }
                    .filter { (_, p) -> p >= minAltProb }

                if (candidates.isEmpty()) continue

                val altCandidates = candidates
                    .filter { (d, _) -> d != currentDigit }
                    .take(maxAltPerCell)

                if (altCandidates.isEmpty()) continue

                for ((altDigit, _) in altCandidates) {
                    if (attempts >= maxAttempts) break
                    attempts++

                    val prev = workingDigits[idx]
                    workingDigits[idx] = altDigit

                    if (!SudokuRules.isGridConsistent(workingDigits)) {
                        workingDigits[idx] = prev
                        continue
                    }

                    val candidateGrid = SudokuGrid(workingDigits)
                    if (hasUniqueSolution(candidateGrid)) {
                        break
                    } else {
                        workingDigits[idx] = prev
                    }
                }

                if (attempts >= maxAttempts) break
            }
        }

        fun computeAssessmentForCurrentDigits(): Triple<Boolean, List<Int>, List<Int>> {
            val grid = SudokuGrid(workingDigits)
            val consistent = SudokuRules.isGridConsistent(workingDigits)
            val solvable = consistent && hasUniqueSolution(grid)

            val conflicts = findConflictIndices(workingDigits)

            val changedSet = mutableSetOf<Int>()
            for (i in 0 until 81) {
                if (workingDigits[i] != originalDigits[i]) changedSet.add(i)
            }

            val lowConf = prediction.lowConfidenceIndices.filter { it !in changedSet }
            return Triple(solvable, conflicts, lowConf)
        }

        var (solvableCurrent, conflictsCurrent, lowConfCurrent) =
            computeAssessmentForCurrentDigits()

        // =========================
        // STAGE 2: BRUTE-FORCE SINGLE-CELL 0..9 ON FEW LOW-CONF CELLS
        // =========================
        if (!solvableCurrent) {
            val unresolvedAfterStage1 = (conflictsCurrent + lowConfCurrent).toSet().toList()

            if (unresolvedAfterStage1.isNotEmpty()) {
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

                            if (!SudokuRules.isGridConsistent(workingDigits)) continue

                            val grid = SudokuGrid(workingDigits)
                            if (hasUniqueSolution(grid)) {
                                cellSuccess = true

                                android.util.Log.i(
                                    "SudokuLogic",
                                    "autoCorrect: bruteForceSingle idx=${idxToCell(idx)} tried=$tried chosen=$digit success=true"
                                )
                                break@bruteLoop
                            }
                        }

                        if (!cellSuccess) {
                            workingDigits[idx] = prevDigit
                            android.util.Log.i(
                                "SudokuLogic",
                                "autoCorrect: bruteForceSingle idx=${idxToCell(idx)} tried=$tried chosen=$prevDigit success=false"
                            )
                        }
                    }

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
            val unresolvedAfterStage2 = (conflictsCurrent + lowConfCurrent).toSet().toList()

            if (unresolvedAfterStage2.size > 1) {
                val prioritized = unresolvedAfterStage2.sortedBy { idx ->
                    if (idx in lowConfCurrent) 0 else 1
                }

                val pairCells = prioritized.take(fallbackMaxUnresolvedCells)

                var pairSuccess = false
                var totalPairTried = 0
                val zeroSkippedIndices = mutableSetOf<Int>()

                outerPair@ for (i in 0 until pairCells.size) {
                    for (j in i + 1 until pairCells.size) {
                        val idx1 = pairCells[i]
                        val idx2 = pairCells[j]

                        val prev1 = workingDigits[idx1]
                        val prev2 = workingDigits[idx2]

                        for (d1 in 0..9) {
                            if (d1 == 0) { zeroSkippedIndices.add(idx1); continue }
                            if (d1 == prev1) continue
                            workingDigits[idx1] = d1

                            for (d2 in 0..9) {
                                if (d2 == 0) { zeroSkippedIndices.add(idx2); continue }
                                if (d2 == prev2) continue
                                workingDigits[idx2] = d2

                                if (!SudokuRules.isGridConsistent(workingDigits)) {
                                    workingDigits[idx2] = prev2
                                    continue
                                }

                                totalPairTried++

                                val grid = SudokuGrid(workingDigits)
                                if (hasUniqueSolution(grid)) {
                                    pairSuccess = true
                                    android.util.Log.i(
                                        "SudokuLogic",
                                        "autoCorrect: bruteForcePair cells=[${idxToCell(idx1)},${idxToCell(idx2)}] chosen=[$d1,$d2] triedCount=$totalPairTried success=true"
                                    )
                                    break@outerPair
                                }

                                workingDigits[idx2] = prev2
                            }

                            if (pairSuccess) break
                            workingDigits[idx1] = prev1
                        }

                        if (!pairSuccess) {
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

                if (zeroSkippedIndices.isNotEmpty()) {
                    android.util.Log.i(
                        "SudokuLogic",
                        "autoCorrect: bruteForcePair zeroSkipped=true cells=${zeroSkippedIndices.map { idxToCell(it) }}"
                    )
                }

                val assessment3 = computeAssessmentForCurrentDigits()
                solvableCurrent = assessment3.first
                conflictsCurrent = assessment3.second
                lowConfCurrent = assessment3.third
            }
        }

        // =========================
        // STAGE 4: FALLBACK COMBINATORIAL SEARCH (TOP-K PER CELL, CNN-GUIDED)
        // =========================
        var unresolvedForFallback = (conflictsCurrent + lowConfCurrent).toSet().toList()

        if (!solvableCurrent &&
            unresolvedForFallback.isNotEmpty() &&
            unresolvedForFallback.size <= fallbackMaxUnresolvedCells
        ) {
            val unresolvedList = unresolvedForFallback.toList()

            android.util.Log.i(
                "SudokuLogic",
                "autoCorrect: fallbackCandidate unresolved=${unresolvedList.map { idxToCell(it) }} count=${unresolvedList.size}"
            )

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

            val changedBeforeFallback = mutableSetOf<Int>()
            for (i in 0 until 81) if (workingDigits[i] != originalDigits[i]) changedBeforeFallback.add(i)

            val unresolvedBefore = unresolvedList.size

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
                fun dfs(depth: Int): Boolean {
                    if (depth == altIndices.size) {
                        val g = SudokuGrid(workingDigits)
                        return hasUniqueSolution(g)
                    }

                    val idx = altIndices[depth]
                    val prev = workingDigits[idx]
                    val candidates = altDigitsPerIndex[depth]

                    for (alt in candidates) {
                        workingDigits[idx] = alt
                        if (!SudokuRules.isGridConsistent(workingDigits)) continue
                        if (dfs(depth + 1)) return true
                    }

                    workingDigits[idx] = prev
                    return false
                }

                fallbackApplied = dfs(0)
            }

            val changedAfterFallback = mutableSetOf<Int>()
            for (i in 0 until 81) if (workingDigits[i] != originalDigits[i]) changedAfterFallback.add(i)

            val conflictsAfter = findConflictIndices(workingDigits)
            val lowConfAfter = prediction.lowConfidenceIndices.filter { it !in changedAfterFallback }
            val unresolvedAfter = (conflictsAfter + lowConfAfter).toSet().size

            val changedDelta = changedAfterFallback.minus(changedBeforeFallback).map { idxToCell(it) }

            android.util.Log.i(
                "SudokuLogic",
                "autoCorrect: fallbackResult success=$fallbackApplied changedDelta=$changedDelta unresolvedBefore=$unresolvedBefore unresolvedAfter=$unresolvedAfter"
            )

            val assessment4 = computeAssessmentForCurrentDigits()
            solvableCurrent = assessment4.first
            conflictsCurrent = assessment4.second
            lowConfCurrent = assessment4.third

            unresolvedForFallback = (conflictsCurrent + lowConfCurrent).toSet().toList()
        }

        // =========================
        // FINAL ASSESSMENT & RETURN
        // =========================
        val finalGrid = SudokuGrid(workingDigits)
        val finalConsistent = SudokuRules.isGridConsistent(workingDigits)
        val finalSolvable = finalConsistent && hasUniqueSolution(finalGrid)

        val finalConflicts = findConflictIndices(workingDigits)

        val finalChanged = mutableListOf<Int>()
        val finalChangedSet = mutableSetOf<Int>()
        for (i in 0 until 81) {
            if (workingDigits[i] != originalDigits[i]) {
                finalChanged.add(i)
                finalChangedSet.add(i)
            }
        }

        val finalLowConf = prediction.lowConfidenceIndices.filter { it !in finalChangedSet }

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

    private fun findConflictIndices(digits: IntArray): List<Int> {
        val conflicts = mutableSetOf<Int>()

        for (row in 0 until 9) {
            val seen = mutableMapOf<Int, MutableList<Int>>()
            for (col in 0 until 9) {
                val idx = row * 9 + col
                val v = digits[idx]
                if (v == 0) continue
                seen.getOrPut(v) { mutableListOf() }.add(idx)
            }
            for ((_, idxList) in seen) if (idxList.size > 1) conflicts.addAll(idxList)
        }

        for (col in 0 until 9) {
            val seen = mutableMapOf<Int, MutableList<Int>>()
            for (row in 0 until 9) {
                val idx = row * 9 + col
                val v = digits[idx]
                if (v == 0) continue
                seen.getOrPut(v) { mutableListOf() }.add(idx)
            }
            for ((_, idxList) in seen) if (idxList.size > 1) conflicts.addAll(idxList)
        }

        for (boxRow in 0 until 3) for (boxCol in 0 until 3) {
            val seen = mutableMapOf<Int, MutableList<Int>>()
            val startRow = boxRow * 3
            val startCol = boxCol * 3
            for (dr in 0 until 3) for (dc in 0 until 3) {
                val row = startRow + dr
                val col = startCol + dc
                val idx = row * 9 + col
                val v = digits[idx]
                if (v == 0) continue
                seen.getOrPut(v) { mutableListOf() }.add(idx)
            }
            for ((_, idxList) in seen) if (idxList.size > 1) conflicts.addAll(idxList)
        }

        return conflicts.toList()
    }
}