package com.contextionary.sudoku.library.model

import org.json.JSONObject

data class AisleManifestModel(
    val aisleId: String,
    val libraryId: String,
    val slug: String,
    val title: String,
    val description: String,
    val sortOrder: Int,
    val organizationPrinciple: String,
    val bookIds: List<String>,
) {
    companion object {
        fun fromJson(json: JSONObject): AisleManifestModel {
            return AisleManifestModel(
                aisleId = json.optString("aisle_id"),
                libraryId = json.optString("library_id"),
                slug = json.optString("slug"),
                title = json.optString("title"),
                description = json.optString("description"),
                sortOrder = json.optInt("sort_order"),
                organizationPrinciple = json.optString("organization_principle"),
                bookIds = json.optJSONArray("book_ids").toStringList(),
            )
        }
    }
}