package com.contextionary.sudoku.logic

/**
 * Convert overlay GridState into LLMGridState (the coordinator input).
 */
fun GridState.toLLMGridState(): LLMGridState {

    val lowConf = cells
        .asSequence()
        .filter { it.isLowConfidence }
        .map { it.index }
        .toList()

    val severity = when {
        hasNoSolution -> "serious"
        conflictIndices.isNotEmpty() -> "serious"
        unresolvedIndices.isNotEmpty() || changedByLogic.isNotEmpty() || lowConf.isNotEmpty() -> "mild"
        else -> "ok"
    }

    val solvability = when {
        hasUniqueSolution -> "unique"
        hasMultipleSolutions -> "multiple"
        hasNoSolution -> "none"
        else -> "none" // safe default
    }

    val retakeRecommendation = when {
        // Strong signal: too many conflicts/unresolved (your app threshold is often 6-8)
        (conflictIndices.size + unresolvedIndices.size) >= 8 -> "strong"
        (conflictIndices.size + unresolvedIndices.size) in 4..7 -> "soft"
        else -> "none"
    }

    return LLMGridState(
        correctedGrid = digits,
        unresolvedCells = unresolvedIndices,
        changedCells = changedByLogic,
        conflictCells = conflictIndices,
        lowConfidenceCells = lowConf,
        solvability = solvability,
        isStructurallyValid = isStructurallyValid,
        unresolvedCount = unresolvedIndices.size,
        severity = severity,
        retakeRecommendation = retakeRecommendation
    )
}