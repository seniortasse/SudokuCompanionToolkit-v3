package com.contextionary.sudoku.library.bridge

import com.contextionary.sudoku.library.model.PuzzleCatalogRecord

object CatalogPuzzleLaunchAdapter {
    fun toSessionBootstrap(record: PuzzleCatalogRecord): CatalogPuzzleSessionBootstrap {
        return CatalogPuzzleSessionBootstrap(
            puzzleUid = record.puzzleUid,
            friendlyPuzzleId = record.friendlyPuzzleId,
            bookId = record.bookId,
            sectionId = record.sectionId,
            sectionCode = record.sectionCode,
            title = record.title,
            subtitle = record.subtitle,
            givens81 = record.givens81,
            solution81 = record.solution81,
            weight = record.weight,
            difficultyLabel = record.difficultyLabel,
            patternId = record.patternId,
            patternName = record.patternName,
            techniqueCount = record.techniqueCount,
            techniquesUsed = record.techniquesUsed,
        )
    }
}