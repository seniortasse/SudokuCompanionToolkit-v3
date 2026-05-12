package com.contextionary.sudoku.library.model

import org.json.JSONObject

data class PrintHeaderModel(
    val displayCode: String,
    val difficultyLabel: String,
    val effortLabel: String,
) {
    companion object {
        fun fromJson(json: JSONObject?): PrintHeaderModel {
            return PrintHeaderModel(
                displayCode = json?.optString("display_code").orEmpty(),
                difficultyLabel = json?.optString("difficulty_label").orEmpty(),
                effortLabel = json?.optString("effort_label").orEmpty(),
            )
        }
    }
}

data class PuzzleCatalogRecord(
    val puzzleUid: String,
    val libraryId: String,
    val aisleId: String,
    val bookId: String?,
    val sectionId: String?,
    val sectionCode: String?,
    val localPuzzleCode: String?,
    val friendlyPuzzleId: String?,
    val title: String,
    val subtitle: String,
    val layoutType: String,
    val gridSize: Int,
    val charset: String,
    val givens81: String,
    val solution81: String,
    val patternId: String?,
    val patternName: String?,
    val clueCount: Int,
    val symmetryType: String?,
    val isUnique: Boolean,
    val isHumanSolvable: Boolean,
    val generationMethod: String,
    val generationSeed: Int?,
    val generatorVersion: String?,
    val weight: Int,
    val difficultyLabel: String,
    val difficultyBandCode: String?,
    val techniqueCount: Int,
    val techniquesUsed: List<String>,
    val appSearchTags: List<String>,
    val positionInSection: Int?,
    val positionInBook: Int?,
    val printHeader: PrintHeaderModel,
) {
    companion object {
        fun fromJson(json: JSONObject): PuzzleCatalogRecord {
            return PuzzleCatalogRecord(
                puzzleUid = json.optString("puzzle_uid"),
                libraryId = json.optString("library_id"),
                aisleId = json.optString("aisle_id"),
                bookId = json.optString("book_id").ifBlank { null },
                sectionId = json.optString("section_id").ifBlank { null },
                sectionCode = json.optString("section_code").ifBlank { null },
                localPuzzleCode = json.optString("local_puzzle_code").ifBlank { null },
                friendlyPuzzleId = json.optString("friendly_puzzle_id").ifBlank { null },
                title = json.optString("title"),
                subtitle = json.optString("subtitle"),
                layoutType = json.optString("layout_type"),
                gridSize = json.optInt("grid_size"),
                charset = json.optString("charset"),
                givens81 = json.optString("givens81"),
                solution81 = json.optString("solution81"),
                patternId = json.optString("pattern_id").ifBlank { null },
                patternName = json.optString("pattern_name").ifBlank { null },
                clueCount = json.optInt("clue_count"),
                symmetryType = json.optString("symmetry_type").ifBlank { null },
                isUnique = json.optBoolean("is_unique"),
                isHumanSolvable = json.optBoolean("is_human_solvable"),
                generationMethod = json.optString("generation_method"),
                generationSeed = if (json.has("generation_seed")) json.optInt("generation_seed") else null,
                generatorVersion = json.optString("generator_version").ifBlank { null },
                weight = json.optInt("weight"),
                difficultyLabel = json.optString("difficulty_label"),
                difficultyBandCode = json.optString("difficulty_band_code").ifBlank { null },
                techniqueCount = json.optInt("technique_count"),
                techniquesUsed = json.optJSONArray("techniques_used").toStringList(),
                appSearchTags = json.optJSONArray("app_search_tags").toStringList(),
                positionInSection = if (json.has("position_in_section")) json.optInt("position_in_section") else null,
                positionInBook = if (json.has("position_in_book")) json.optInt("position_in_book") else null,
                printHeader = PrintHeaderModel.fromJson(json.optJSONObject("print_header")),
            )
        }
    }
}