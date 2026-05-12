package com.contextionary.sudoku.library.repository

import android.content.Context
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader

class CatalogAssetLoader(
    private val context: Context,
    private val basePath: String = "sudoku_library/classic9",
) {
    fun readJsonObject(relativePath: String): JSONObject {
        val text = readText(relativePath)
        return JSONObject(text)
    }

    fun readText(relativePath: String): String {
        val assetPath = "$basePath/$relativePath"
        context.assets.open(assetPath).use { input ->
            BufferedReader(InputStreamReader(input, Charsets.UTF_8)).use { reader ->
                return reader.readText()
            }
        }
    }

    fun list(relativeDir: String): List<String> {
        val assetPath = "$basePath/$relativeDir"
        return context.assets.list(assetPath)?.toList().orEmpty().sorted()
    }

    fun exists(relativePath: String): Boolean {
        return try {
            context.assets.open("$basePath/$relativePath").close()
            true
        } catch (_: Exception) {
            false
        }
    }
}