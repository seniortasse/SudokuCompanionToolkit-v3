package com.contextionary.sudoku.library.repository

import com.contextionary.sudoku.library.model.AisleManifestModel
import com.contextionary.sudoku.library.model.CatalogBookSummary
import com.contextionary.sudoku.library.model.CatalogManifestModel

class LibraryCatalogRepository(
    private val loader: CatalogAssetLoader,
) {
    fun getCatalogManifest(): CatalogManifestModel {
        return CatalogManifestModel.fromJson(
            loader.readJsonObject("catalog_manifest.json")
        )
    }

    fun getBookSummaries(): List<CatalogBookSummary> {
        return getCatalogManifest().bookSummaries
    }

    fun getAisleManifests(): List<AisleManifestModel> {
        val catalog = getCatalogManifest()
        return catalog.libraries
            .flatMap { it.aisleIds }
            .distinct()
            .map { aisleId ->
                AisleManifestModel.fromJson(
                    loader.readJsonObject("books/$aisleId.json")
                )
            }
            .sortedBy { it.sortOrder }
    }
}