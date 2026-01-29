package com.contextionary.sudoku

import android.util.Log

enum class CellKind { EMPTY, GIVEN, SOLUTION }

data class DisplayCell(
    val kind: CellKind,
    val digit: Int,              // 1..9 or 0 if EMPTY
    val conf: Float,             // confidence of chosen head (0..1)
    val givenConf: Float,        // passthrough for logging
    val solutionConf: Float,     // passthrough for logging
    val candidates: List<Int>    // subset of 1..9 (visual thresholded)
) {
    companion object {
        val EMPTY = DisplayCell(CellKind.EMPTY, 0, 0f, 0f, 0f, emptyList())
    }
}

object DisplayPolicy {
    private const val TAG = "DisplayPolicy"

    // Tunables (chosen to match the expectations we aligned on)
    private const val GIVEN_MIN = 0.35f
    private const val SOL_MIN   = 0.55f
    private const val CAND_VISUAL_THR = 0.50f   // draw candidates at/above this (visual only)

    fun choose(readout: CellReadout): DisplayCell {
        val givenDigit = readout.givenDigit
        val solDigit   = readout.solutionDigit
        val givenConf  = readout.givenConf
        val solConf    = readout.solutionConf

        // 1) Prefer printed clue if confident enough
        if (givenDigit in 1..9 && givenConf >= GIVEN_MIN) {
            return DisplayCell(
                kind = CellKind.GIVEN,
                digit = givenDigit,
                conf = givenConf,
                givenConf = givenConf,
                solutionConf = solConf,
                candidates = collectCandidates(readout)
            )
        }

        // 2) Otherwise handwritten answer if confident enough
        if (solDigit in 1..9 && solConf >= SOL_MIN) {
            return DisplayCell(
                kind = CellKind.SOLUTION,
                digit = solDigit,
                conf = solConf,
                givenConf = givenConf,
                solutionConf = solConf,
                candidates = collectCandidates(readout)
            )
        }

        // 3) Else empty (but still expose candidate list visually)
        return DisplayCell(
            kind = CellKind.EMPTY,
            digit = 0,
            conf = 0f,
            givenConf = givenConf,
            solutionConf = solConf,
            candidates = collectCandidates(readout)
        )
    }

    private fun collectCandidates(r: CellReadout): List<Int> {
        val out = ArrayList<Int>(9)
        for (d in 1..9) {
            val p = r.candidateConfs.getOrElse(d) { 0f }
            if (p >= CAND_VISUAL_THR) out += d
        }
        return out
    }

    // Per-cell diagnostic line
    fun logDecision(r: Int, c: Int, readout: CellReadout, disp: DisplayCell) {
        val candStr = if (disp.candidates.isEmpty()) "[]" else disp.candidates.joinToString(",","[","]")
        Log.i(TAG,
            "r${r+1}c${c+1}  given=${readout.givenDigit}@${"%.3f".format(readout.givenConf)}  " +
                    "sol=${readout.solutionDigit}@${"%.3f".format(readout.solutionConf)}  " +
                    "→ shown=${disp.kind.name}:${disp.digit}@${"%.3f".format(disp.conf)}  " +
                    "cands>=0.50=$candStr"
        )
    }

    // Grid-level summary (call once per capture)
    fun logSummary(grid: Array<Array<DisplayCell>>) {
        var g = 0; var s = 0; var e = 0
        var gConf = 0f; var sConf = 0f
        for (r in 0 until 9) for (c in 0 until 9) {
            when (grid[r][c].kind) {
                CellKind.GIVEN -> { g++; gConf += grid[r][c].conf }
                CellKind.SOLUTION -> { s++; sConf += grid[r][c].conf }
                CellKind.EMPTY -> e++
            }
        }
        val gAvg = if (g > 0) gConf / g else 0f
        val sAvg = if (s > 0) sConf / s else 0f
        Log.i(TAG, "summary shown: givens=$g (avg=${"%.3f".format(gAvg)}), solutions=$s (avg=${"%.3f".format(sAvg)}), empty=$e")
    }
}