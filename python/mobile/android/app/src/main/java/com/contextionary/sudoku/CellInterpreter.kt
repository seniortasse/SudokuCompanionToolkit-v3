package com.contextionary.sudoku

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

// -------------------------------
// Data models
// -------------------------------
data class CellReadout(
    val givenDigit: Int,
    val givenConf: Float,
    val solutionDigit: Int,
    val solutionConf: Float,
    val candidateMask: Int,
    val candidateConfs: FloatArray,
    val givenProbs10: FloatArray,
    val solutionProbs10: FloatArray
) {
    init {
        require(candidateConfs.size == 10) { "candidateConfs must have size 10" }
        require(givenProbs10.size == 10) { "givenProbs10 must have size 10" }
        require(solutionProbs10.size == 10) { "solutionProbs10 must have size 10" }
    }
}

data class CellGridReadout(
    val rows: Int = 9,
    val cols: Int = 9,
    val cells: Array<Array<CellReadout>>
)

// -------------------------------
// Interpreter
// -------------------------------
class CellInterpreter(
    context: Context,
    modelAsset: String = MODEL_FP32,
    numThreads: Int = 2
) {
    companion object {
        private const val TAG = "CellInterpreter"

        // Assets
        const val MODEL_FP32 = "models/cell_interpreter_fp32.tflite"
        const val MODEL_INT8_DYNAMIC = "models/cell_interpreter_int8_dynamic.tflite"

        // === Preproc (mirror training) ===
        private const val APPLY_NORMALIZATION = true   // (x-0.5)/0.5 → [-1,+1]
        private const val APPLY_INVERSION    = false   // keep “white≈+1, black≈-1”

        // === Temperatures ===
        private const val T_GIVEN      = 1.0f
        private const val T_SOLUTION   = 1.0f
        private const val T_CANDIDATES = 1.0f

        // === Candidates operating point (your current choice) ===
        private const val CANDIDATE_THR = 0.58f

        // --- math helpers ---
        private fun softmaxTemp(logits: FloatArray, T: Float): FloatArray {
            val out = FloatArray(logits.size)
            var m = Float.NEGATIVE_INFINITY
            for (v in logits) if (v > m) m = v
            var s = 0f
            for (i in logits.indices) {
                val z = (logits[i] - m) / T
                val e = exp(z.toDouble()).toFloat()
                out[i] = e
                s += e
            }
            if (s > 0f) for (i in out.indices) out[i] /= s
            return out
        }

        private fun sigmoid(x: Float): Float = 1f / (1f + exp(-x.toDouble()).toFloat())

        private fun argmax(a: FloatArray): Int {
            var bi = 0
            var bv = a[0]
            for (i in 1 until a.size) if (a[i] > bv) {
                bv = a[i]
                bi = i
            }
            return bi
        }
    }

    // TFLite
    private val interpreter: Interpreter
    private val inputW: Int
    private val inputH: Int
    private val isCHW: Boolean
    private val outCount: Int

    // App context & debug toggles
    private val appContext: Context = context.applicationContext
    private val DUMP_PER_CELL = true         // CSV of probs (solution+given)
    private val DUMP_PER_CELL_LOGITS = true  // CSV of raw logits

    init {
        val opts = Interpreter.Options().apply {
            setNumThreads(numThreads)
            setUseXNNPACK(true)
        }

        val afd = context.assets.openFd(modelAsset)
        val fis = FileInputStream(afd.fileDescriptor)
        val fc = fis.channel
        val modelBuffer = fc.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)

        interpreter = Interpreter(modelBuffer, opts)

        // Introspect IO tensors
        val inT = interpreter.getInputTensor(0)
        val inShape = inT.shape()
        val inName = inT.name()

        isCHW = (inShape.size == 4 && inShape[1] == 1 && inShape[2] != 1)
        if (isCHW) {
            inputH = inShape[2]; inputW = inShape[3]
        } else {
            inputH = inShape[1]; inputW = inShape[2]
        }

        outCount = interpreter.outputTensorCount

        Log.i(TAG, "IN  name=$inName shape=${inShape.contentToString()} layout=${if (isCHW) "CHW" else "NHWC"} size=${inputW}x$inputH")
        Log.i(TAG, "OUT count=$outCount")
        for (i in 0 until outCount) {
            val t = interpreter.getOutputTensor(i)
            Log.i(TAG, "OUT[$i] name=${t.name()} shape=${t.shape().contentToString()}")
        }

        Log.i(TAG, "Fixed mapping for this model: out[0]=CANDIDATES, out[1]=SOLUTION, out[2]=GIVEN")
        Log.i(TAG, "Initialized with $modelAsset")
    }

    fun close() = interpreter.close()

    // ---------------------------------------
    // CSV dumpers (heads are already mapped)
    // ---------------------------------------
    private fun dumpPerCellCsv(
        givList: List<FloatArray>,
        solList: List<FloatArray>
    ) {
        if (!DUMP_PER_CELL) return
        try {
            val baseDir = File(appContext.filesDir, "runs")
            if (!baseDir.exists()) baseDir.mkdirs()
            val debugDir = File(baseDir, "cell_debug/${System.currentTimeMillis()}")
            if (!debugDir.exists()) debugDir.mkdirs()

            val csv = File(debugDir, "per_cell_android_SOL=head1_GIV=head2_fixed.csv")

            fun argmaxWithProbLocal(p: FloatArray): Pair<Int, Float> {
                var bi = 0; var bv = p[0]
                for (i in 1 until p.size) if (p[i] > bv) { bv = p[i]; bi = i }
                return bi to bv
            }

            csv.bufferedWriter().use { w ->
                val solCols = (0..9).joinToString(",") { i -> "sol$i" }
                val givCols = (0..9).joinToString(",") { i -> "giv$i" }
                w.appendLine("idx,r,c,$solCols,$givCols,pred_solution,conf_solution,pred_given,conf_given")

                var idx = 0
                for (r in 0 until 9) for (c in 0 until 9) {
                    val solProbs = softmaxTemp(solList[idx], T_SOLUTION)
                    val givProbs = softmaxTemp(givList[idx], T_GIVEN)
                    val (solArg, solConf) = argmaxWithProbLocal(solProbs)
                    val (givArg, givConf) = argmaxWithProbLocal(givProbs)

                    fun f(a: Float) = "%.6f".format(a)
                    val row = buildString {
                        append("${idx},$r,$c,")
                        append((0..9).joinToString(",") { i -> f(solProbs[i]) }); append(",")
                        append((0..9).joinToString(",") { i -> f(givProbs[i]) }); append(",")
                        append("$solArg,${f(solConf)},$givArg,${f(givConf)}")
                    }
                    w.appendLine(row)
                    idx++
                }
            }
            Log.i(TAG, "Wrote per-cell CSV at: ${csv.absolutePath}")
        } catch (t: Throwable) {
            Log.w(TAG, "Failed writing per-cell CSV: ${t.message}")
        }
    }

    private fun dumpPerCellLogitsCsv(
        candList: List<FloatArray>,
        solList: List<FloatArray>,
        givList: List<FloatArray>
    ) {
        if (!DUMP_PER_CELL_LOGITS) return
        try {
            val baseDir = File(appContext.filesDir, "runs")
            if (!baseDir.exists()) baseDir.mkdirs()
            val debugDir = File(baseDir, "cell_debug/${System.currentTimeMillis()}")
            if (!debugDir.exists()) debugDir.mkdirs()

            val csv = File(debugDir, "per_cell_android_logits.csv")
            csv.bufferedWriter().use { w ->
                val hCols = (0..9).joinToString(",") { "h$it" }
                w.appendLine("idx,r,c,cand_$hCols,sol_$hCols,giv_$hCols")
                var idx = 0
                for (r in 0 until 9) for (c in 0 until 9) {
                    fun f(a: Float) = "%.6f".format(a)
                    val row = buildString {
                        append("${(r*9+c)},$r,$c,")
                        append((0..9).joinToString(",") { i -> f(candList[idx][i]) }); append(",")
                        append((0..9).joinToString(",") { i -> f(solList[idx][i]) }); append(",")
                        append((0..9).joinToString(",") { i -> f(givList[idx][i]) })
                    }
                    w.appendLine(row)
                    idx++
                }
            }
            Log.i(TAG, "Wrote per-cell LOGITS CSV at: ${csv.absolutePath}")
        } catch (t: Throwable) {
            Log.w(TAG, "Failed writing per-cell LOGITS CSV: ${t.message}")
        }
    }

    // -------------------------------
    // Main inference (deterministic)
    // -------------------------------
    fun interpretTiles(tiles: Array<Array<Bitmap>>): Array<Array<CellReadout>> {
        require(tiles.size == 9 && tiles[0].size == 9) { "Expected 9x9 tiles" }

        val candLogitsList = ArrayList<FloatArray>(81)
        val solLogitsList = ArrayList<FloatArray>(81)
        val givLogitsList = ArrayList<FloatArray>(81)

        var loggedFirst = false

        for (r in 0 until 9) for (c in 0 until 9) {
            val pre = preprocessTile(tiles[r][c])
            val input = if (isCHW) pre.chwBuffer else pre.nhwcBuffer
            val inputClean = input.duplicate().order(ByteOrder.nativeOrder())
            inputClean.clear()

            // Deterministic outputs: out[0]=CAND, out[1]=SOL, out[2]=GIV
            val outCand = Array(1) { FloatArray(10) }
            val outSol = Array(1) { FloatArray(10) }
            val outGiv = Array(1) { FloatArray(10) }

            if (outCount == 3) {
                val outputs = hashMapOf<Int, Any>(
                    0 to outCand,
                    1 to outSol,
                    2 to outGiv
                )
                try { interpreter.runForMultipleInputsOutputs(arrayOf<Any>(inputClean), outputs) }
                catch (t: Throwable) { throw enrichNio(t) }
            } else if (outCount == 1) {
                // Fallback for packed output: [1,30] as (cand, sol, giv)
                val packed = Array(1) { FloatArray(30) }
                val outputs = hashMapOf<Int, Any>(0 to packed)
                try { interpreter.runForMultipleInputsOutputs(arrayOf<Any>(inputClean), outputs) }
                catch (t: Throwable) { throw enrichNio(t) }

                val v = packed[0]
                for (i in 0 until 10) outCand[0][i] = v[i]
                for (i in 0 until 10) outSol[0][i] = v[10 + i]
                for (i in 0 until 10) outGiv[0][i] = v[20 + i]
            } else {
                throw IllegalStateException("Unexpected output tensor count: $outCount")
            }

            val cand = outCand[0]
            val sol = outSol[0]
            val giv = outGiv[0]

            if (!loggedFirst) {
                fun FloatArray.h(n: Int) =
                    take(kotlin.math.min(n, size)).joinToString(prefix = "[", postfix = "]") { "%.2f".format(it) }

                val sp = softmaxTemp(sol, T_SOLUTION)
                val gp = softmaxTemp(giv, T_GIVEN)

                Log.i(TAG, "Preproc stats (first tile): min=${"%.3f".format(lastMin)} max=${"%.3f".format(lastMax)} mean=${"%.3f".format(lastMean)} inv=$APPLY_INVERSION norm=$APPLY_NORMALIZATION")
                Log.i(TAG, "Sanity logits: cand[0..5]=${cand.h(6)}  sol[0..5]=${sol.h(6)}  giv[0..5]=${giv.h(6)}")
                Log.i(TAG, "Sanity probs : sol [0..5]=${sp.h(6)}  giv [0..5]=${gp.h(6)}")

                // warn-only softmax sanity
                val sSum = sp.sum()
                val gSum = gp.sum()
                if (kotlin.math.abs(sSum - 1f) > 0.02f || kotlin.math.abs(gSum - 1f) > 0.02f) {
                    Log.w(TAG, "Sanity WARN: expected two softmax-like heads. sums: sol=$sSum giv=$gSum")
                }

                loggedFirst = true
            }

            candLogitsList.add(cand)
            solLogitsList.add(sol)
            givLogitsList.add(giv)
        }

        // Dumps
        dumpPerCellLogitsCsv(candLogitsList, solLogitsList, givLogitsList)
        dumpPerCellCsv(givLogitsList, solLogitsList)

        // Decode to CellReadout
        val out = Array(9) { Array(9) { dummyCell() } }
        var k = 0
        for (r in 0 until 9) for (c in 0 until 9) {
            val cell = postProcess(
                givenLogits = givLogitsList[k],
                solLogits = solLogitsList[k],
                candLogits = candLogitsList[k],
                dumpOnce = (k == 0)
            )
            out[r][c] = cell
            k++
        }
        return out
    }

    private fun dummyCell(): CellReadout {
        val oneHotBlank = FloatArray(10) { 0f }.apply { this[0] = 1f }
        return CellReadout(
            givenDigit = 0, givenConf = 1f,
            solutionDigit = 0, solutionConf = 1f,
            candidateMask = 0,
            candidateConfs = FloatArray(10) { 0f },
            givenProbs10 = oneHotBlank.copyOf(),
            solutionProbs10 = oneHotBlank.copyOf()
        )
    }

    private fun postProcess(
        givenLogits: FloatArray,
        solLogits: FloatArray,
        candLogits: FloatArray,
        dumpOnce: Boolean
    ): CellReadout {
        val givenProbs = softmaxTemp(givenLogits, T_GIVEN)      // length 10
        val solProbs = softmaxTemp(solLogits, T_SOLUTION)       // length 10

        val givenDigit = argmax(givenProbs)
        val givenConf = givenProbs[givenDigit]
        val solDigit = argmax(solProbs)
        val solConf = solProbs[solDigit]

        // Candidates = sigmoid per digit 0..9, but we only mask 1..9
        val candProbs = FloatArray(10) { 0f }
        for (d in 0..9) candProbs[d] = sigmoid(candLogits[d] / T_CANDIDATES)

        var mask = 0
        for (d in 1..9) if (candProbs[d] >= CANDIDATE_THR) mask = mask or (1 shl (d - 1))

        if (dumpOnce) {
            Log.i(
                TAG,
                "Pred → G=$givenDigit@${"%.3f".format(givenConf)}  " +
                        "S=$solDigit@${"%.3f".format(solConf)}  " +
                        "cand>=${"%.2f".format(CANDIDATE_THR)}: ${(1..9).filter { candProbs[it] >= CANDIDATE_THR }}"
            )
        }

        return CellReadout(
            givenDigit = givenDigit,
            givenConf = givenConf.coerceIn(0f, 1f),
            solutionDigit = solDigit,
            solutionConf = solConf.coerceIn(0f, 1f),
            candidateMask = mask,
            candidateConfs = candProbs,
            givenProbs10 = givenProbs,
            solutionProbs10 = solProbs
        )
    }

    private fun enrichNio(t: Throwable): Throwable {
        return RuntimeException(
            "TFLite run failed (layout=${if (isCHW) "CHW" else "NHWC"} ${inputW}x$inputH, outCount=$outCount): ${t::class.java.simpleName}: ${t.message}",
            t
        )
    }

    // -----------------------
    // Preprocessing
    // -----------------------
    private data class PreprocessedTile(
        val chwBuffer: ByteBuffer,
        val nhwcBuffer: ByteBuffer
    )

    private var hasLoggedStats = false
    private var lastMin = 0f
    private var lastMax = 1f
    private var lastMean = 0.5f

    private fun preprocessTile(tile: Bitmap): PreprocessedTile {
        val gray = toGrayscale(tile)
        val scaled = Bitmap.createScaledBitmap(gray, inputW, inputH, true)

        val n = inputW * inputH
        val arr = FloatArray(n)

        val pixels = IntArray(n)
        scaled.getPixels(pixels, 0, inputW, 0, 0, inputW, inputH)

        var minV = 1f
        var maxV = 0f
        var sumV = 0f
        var i = 0

        for (y in 0 until inputH) {
            for (x in 0 until inputW) {
                val color = pixels[i++]
                val r = (color shr 16) and 0xFF
                val g = (color shr 8) and 0xFF
                val b = (color) and 0xFF
                var v = (0.299f * r + 0.587f * g + 0.114f * b) / 255f  // [0,1], white≈1
                if (APPLY_INVERSION) v = 1f - v
                val norm = if (APPLY_NORMALIZATION) (v - 0.5f) / 0.5f else v

                minV = min(minV, norm)
                maxV = max(maxV, norm)
                sumV += norm
                arr[y * inputW + x] = norm
            }
        }

        if (!hasLoggedStats) {
            val meanV = sumV / n
            lastMin = minV; lastMax = maxV; lastMean = meanV
            Log.i(TAG, "Preproc stats (first tile): min=${"%.3f".format(minV)} max=${"%.3f".format(maxV)} mean=${"%.3f".format(meanV)} inv=$APPLY_INVERSION norm=$APPLY_NORMALIZATION")
            hasLoggedStats = true
        }

        val nhwc = ByteBuffer.allocateDirect(4 * n).order(ByteOrder.nativeOrder())
        val chw = ByteBuffer.allocateDirect(4 * n).order(ByteOrder.nativeOrder())

        for (y in 0 until inputH) for (x in 0 until inputW) nhwc.putFloat(arr[y * inputW + x])
        nhwc.rewind()

        for (y in 0 until inputH) for (x in 0 until inputW) chw.putFloat(arr[y * inputW + x])
        chw.rewind()

        return PreprocessedTile(chwBuffer = chw, nhwcBuffer = nhwc)
    }

    private fun toGrayscale(src: Bitmap): Bitmap {
        val out = Bitmap.createBitmap(src.width, src.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        val paint = Paint()
        val cm = ColorMatrix()
        cm.setSaturation(0f)
        paint.colorFilter = ColorMatrixColorFilter(cm)
        canvas.drawBitmap(src, 0f, 0f, paint)
        return out
    }
}