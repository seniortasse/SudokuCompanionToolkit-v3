package com.contextionary.sudoku.library.model

import org.json.JSONArray
import org.json.JSONObject

data class CatalogLibrarySummary(
    val libraryId: String,
    val title: String,
    val subtitle: String,
    val aisleIds: List<String>,
)

data class CatalogBookSummary(
    val bookId: String,
    val title: String,
    val subtitle: String,
    val aisleId: String,
    val puzzleCount: Int,
)

data class CatalogManifestModel(
    val catalogVersion: String,
    val generatedAt: String,
    val libraryIds: List<String>,
    val libraries: List<CatalogLibrarySummary>,
    val bookSummaries: List<CatalogBookSummary>,
    val indexFiles: Map<String, String>,
) {
    companion object {
        fun fromJson(json: JSONObject): CatalogManifestModel {
            val libraries = json.optJSONArray("libraries").toObjectList { obj ->
                CatalogLibrarySummary(
                    libraryId = obj.optString("library_id"),
                    title = obj.optString("title"),
                    subtitle = obj.optString("subtitle"),
                    aisleIds = obj.optJSONArray("aisle_ids").toStringList(),
                )
            }

            val bookSummaries = json.optJSONArray("book_summaries").toObjectList { obj ->
                CatalogBookSummary(
                    bookId = obj.optString("book_id"),
                    title = obj.optString("title"),
                    subtitle = obj.optString("subtitle"),
                    aisleId = obj.optString("aisle_id"),
                    puzzleCount = obj.optInt("puzzle_count"),
                )
            }

            val indexFiles = buildMap {
                val indexObject = json.optJSONObject("index_files") ?: JSONObject()
                val keys = indexObject.keys()
                while (keys.hasNext()) {
                    val key = keys.next()
                    put(key, indexObject.optString(key))
                }
            }

            return CatalogManifestModel(
                catalogVersion = json.optString("catalog_version"),
                generatedAt = json.optString("generated_at"),
                libraryIds = json.optJSONArray("library_ids").toStringList(),
                libraries = libraries,
                bookSummaries = bookSummaries,
                indexFiles = indexFiles,
            )
        }
    }
}

internal inline fun <T> JSONArray?.toObjectList(mapper: (JSONObject) -> T): List<T> {
    if (this == null) return emptyList()
    val out = mutableListOf<T>()
    for (i in 0 until length()) {
        val obj = optJSONObject(i) ?: continue
        out += mapper(obj)
    }
    return out
}

internal fun JSONArray?.toStringList(): List<String> {
    if (this == null) return emptyList()
    val out = mutableListOf<String>()
    for (i in 0 until length()) {
        out += optString(i)
    }
    return out
}