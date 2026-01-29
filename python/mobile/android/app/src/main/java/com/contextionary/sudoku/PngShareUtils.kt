package com.contextionary.sudoku

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import androidx.core.content.FileProvider
import java.io.File
import java.io.FileOutputStream

object PngShareUtils {

    /**
     * Save a PNG into cache and return a content:// Uri via FileProvider.
     * Parameter names are intentionally 'context' and 'fileName'
     * so you can call with named args from MainActivity.
     */
    @JvmStatic
    fun savePngToCache(
        context: Context,
        fileName: String,
        bmp: Bitmap
    ): Uri {
        // /cache/shared/...
        val sharedDir = File(context.cacheDir, "shared")
        if (!sharedDir.exists()) sharedDir.mkdirs()

        val safeName = if (fileName.endsWith(".png", ignoreCase = true)) fileName else "$fileName.png"
        val outFile = File(sharedDir, safeName)

        FileOutputStream(outFile).use { fos ->
            bmp.compress(Bitmap.CompressFormat.PNG, 100, fos)
        }

        val authority = "${context.packageName}.provider"
        return FileProvider.getUriForFile(context, authority, outFile)
    }

    /**
     * Standard chooser share. Grants read permission for the Uri.
     */
    @JvmStatic
    fun shareImage(activity: Activity, uri: Uri, title: String = "Share Sudoku") {
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "image/png"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        activity.startActivity(Intent.createChooser(intent, title))
    }
}