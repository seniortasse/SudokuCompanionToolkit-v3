package com.contextionary.sudoku.library.bridge

data class CatalogPuzzleSessionBootstrap(
    val puzzleSource: String = "LIBRARY_CATALOG",
    val puzzleUid: String,
    val friendlyPuzzleId: String?,
    val bookId: String?,
    val sectionId: String?,
    val sectionCode: String?,
    val title: String,
    val subtitle: String,
    val givens81: String,
    val solution81: String,
    val weight: Int,
    val difficultyLabel: String,
    val patternId: String?,
    val patternName: String?,
    val techniqueCount: Int,
    val techniquesUsed: List<String>,
)