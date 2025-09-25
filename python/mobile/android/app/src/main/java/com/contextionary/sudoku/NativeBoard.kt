package com.contextionary.sudoku

object NativeBoard {
    init { System.loadLibrary("native-lib") }

    external fun warpBoard(
        sourceBitmap: android.graphics.Bitmap,
        corners8f: FloatArray, // [x0,y0, x1,y1, x2,y2, x3,y3] TL,TR,BR,BL
        boardSize: Int         // e.g., 900
    ): android.graphics.Bitmap?

    external fun tile81(
        boardBitmap: android.graphics.Bitmap, // square
        tileSize: Int                         // e.g., 64
    ): Array<android.graphics.Bitmap>?
}