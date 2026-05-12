package com.contextionary.sudoku.library.repository

import com.contextionary.sudoku.library.model.PuzzleCatalogRecord

class PuzzleRepository(
    private val loader: CatalogAssetLoader,
) {
    fun getAllPuzzles(): List<PuzzleCatalogRecord> {
        return loader.list("puzzles")
            .filter { it.endsWith(".json", ignoreCase = true) }
            .map { fileName ->
                PuzzleCatalogRecord.fromJson(
                    loader.readJsonObject("puzzles/$fileName")
                )
            }
            .sortedBy { it.puzzleUid }
    }

    fun getPuzzleByUid(puzzleUid: String): PuzzleCatalogRecord? {
        val relativePath = "puzzles/$puzzleUid.json"
        if (!loader.exists(relativePath)) return null
        return PuzzleCatalogRecord.fromJson(loader.readJsonObject(relativePath))
    }

    fun findPuzzleByAnyId(rawId: String): PuzzleCatalogRecord? {
        val needle = rawId.trim()
        if (needle.isBlank()) return null

        return getAllPuzzles().firstOrNull { record ->
            record.puzzleUid.equals(needle, ignoreCase = true) ||
                    (record.friendlyPuzzleId?.equals(needle, ignoreCase = true) == true) ||
                    (record.localPuzzleCode?.equals(needle, ignoreCase = true) == true)
        }
    }

    fun searchByDifficultyLabel(label: String): List<PuzzleCatalogRecord> {
        val needle = label.trim().lowercase()
        if (needle.isBlank()) return emptyList()
        return getAllPuzzles().filter { it.difficultyLabel.lowercase() == needle }
    }

    fun searchByTechnique(technique: String): List<PuzzleCatalogRecord> {
        val needle = technique.trim().lowercase()
        if (needle.isBlank()) return emptyList()
        return getAllPuzzles().filter { record ->
            record.techniquesUsed.any { it.lowercase() == needle }
        }
    }
}