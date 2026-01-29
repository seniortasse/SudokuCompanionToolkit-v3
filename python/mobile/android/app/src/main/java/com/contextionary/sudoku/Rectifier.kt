package com.contextionary.sudoku

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.RectF
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.core.Core.*
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

class Rectifier(private val ctx: Context) {

    data class Options(
        val tileSize: Int = 64,
        val shrink: Float = 0.14f,
        val precompress: Boolean = false,
        val targetKb: Int = 120,
        val maxSide: Int = 1600,
        val minJpegQuality: Int = 45,
        val robust: Boolean = false,
        // NOTE: use IntProgression (.. step ..) returns IntProgression, not IntRange
        val angleSweepStrict: IntProgression = (-18..18 step 2),
        val angleSweepRobust: IntProgression = (-24..24 step 2),
        val debugTag: String = "rectify"
    )

    data class Result(
        val boardWarped: Bitmap,
        val boardClean: Bitmap,
        val tiles: Array<Array<Bitmap>>,        // [9][9]
        val points10x10: Array<Array<PointF>>,  // [10][10], ROI coords
        val roiSrc: Rect,                       // in source coords (OpenCV)
        val outDir: File,
        val cellsJson: File,
        val pointsJson: File
    )

    // ─────────────────────────────────────────────────────────────────────────
    // Public entry
    // ─────────────────────────────────────────────────────────────────────────
    fun rectify(
        srcBmp: Bitmap,
        detectorRoi: android.graphics.Rect,
        opts: Options = Options(),
        outRoot: File? = null
    ): Result {
        require(detectorRoi.width() > 4 && detectorRoi.height() > 4) { "Empty ROI" }

        val caseRoot = outRoot ?: File(
            ctx.getExternalFilesDir(null),
            "runs/grid_rectification/${System.currentTimeMillis()}/android"
        ).apply { mkdirs() }



        Log.i("Rectifier", "caseRoot=${caseRoot.absolutePath} (parent=${caseRoot.parentFile?.absolutePath})")

        // Crop ROI
        val roi = Rect(
            detectorRoi.left.coerceAtLeast(0),
            detectorRoi.top.coerceAtLeast(0),
            detectorRoi.right.coerceAtMost(srcBmp.width),
            detectorRoi.bottom.coerceAtMost(srcBmp.height)
        )
        val roiBmp = Bitmap.createBitmap(srcBmp, roi.x, roi.y, roi.width, roi.height)
        savePng(roiBmp, File(caseRoot.parentFile!!, "roi.png")) // also at case root for convenience

        // Work gray Mat
        val gray = toGray(roiBmp)

        // 1) angle sweep strict
        val strictAngles = sequenceFromRange(opts.angleSweepStrict)
        var bestAngle: Double? = null
        var bestScore = Double.NEGATIVE_INFINITY
        var bestCache: Cache? = null

        for (ang in strictAngles) {
            val g = rotate(gray, ang.toDouble())
            val (mhAll, mvAll, maskAll) = buildOrientedMasks(g)
            val (mh, mv, roiBox) = sudokuRoiFromMasks(mhAll, mvAll)
            if (roiBox.width <= 0 || roiBox.height <= 0) continue
            val approxCell = min(roiBox.width, roiBox.height) / 9.0
            try {
                val hsel = select8Lines(mh, 'h', approxCell, mv)
                val vsel = select8Lines(mv, 'v', approxCell, mh)
                val cvh = cvGaps(lineCenters(hsel, 'h'))
                val cvv = cvGaps(lineCenters(vsel, 'v'))
                val score = 100.0 - 100.0 * (cvh + cvv) - 0.5 * abs(ang.toDouble())
                if (score > bestScore) {
                    bestScore = score
                    bestAngle = ang.toDouble()
                    bestCache = Cache(g, mhAll, mvAll, maskAll, roiBox, hsel, vsel)
                }
            } catch (_: Exception) {
                // try next angle
            }
        }

        // Robust fallback (pre-crop + center-biased)
        if (bestAngle == null && opts.robust) {
            val grayR = precropHeaderMargins(gray)
            val robustAngles = sequenceFromRange(opts.angleSweepRobust)
            for (ang in robustAngles) {
                val g = rotate(grayR, ang.toDouble())
                val (mhAll, mvAll, maskAll) = buildOrientedMasks(g)
                val (mh, mv, roiBox) = sudokuRoiFromMasks(mhAll, mvAll)
                if (roiBox.width <= 0 || roiBox.height <= 0) continue
                val approxCell = min(roiBox.width, roiBox.height) / 9.0
                try {
                    val hsel = pick8CenterBiased(mh, 'h', approxCell)
                    val vsel = pick8CenterBiased(mv, 'v', approxCell)
                    val cvh = cvGaps(lineCenters(hsel, 'h'))
                    val cvv = cvGaps(lineCenters(vsel, 'v'))
                    val score = 100.0 - 100.0 * (cvh + cvv) - 0.3 * abs(ang.toDouble())
                    if (score > bestScore) {
                        bestScore = score
                        bestAngle = ang.toDouble()
                        bestCache = Cache(g, mhAll, mvAll, maskAll, roiBox, hsel, vsel)
                    }
                } catch (_: Exception) { }
            }
        }

        require(bestAngle != null && bestCache != null) { "Failed to find angle with 8×8 internal lines" }

        // Full-res at best angle (we already have bestCache.g = gray rotated)
        val grayBest = bestCache!!.g
        val mhAll = bestCache!!.mhAll
        val mvAll = bestCache!!.mvAll
        val maskAll = bestCache!!.maskAll
        val roiBox = bestCache!!.roi
        var hMasks = bestCache!!.hsel
        var vMasks = bestCache!!.vsel

        // Debug S1 saves
        saveMat(mhAll, File(caseRoot, "rectify_debug/S1_lines_h.png"))
        saveMat(mvAll, File(caseRoot, "rectify_debug/S1_lines_v.png"))
        saveMat(maskAll, File(caseRoot, "rectify_debug/S1_grid_mask.png"))
        // ROI rect on grayBest (draw box)
        run {
            val vis = Mat()
            Imgproc.cvtColor(grayBest, vis, Imgproc.COLOR_GRAY2BGR)
            Imgproc.rectangle(vis, roiBox, Scalar(0.0, 255.0, 0.0), 2)
            saveMat(vis, File(caseRoot, "rectify_debug/S1_roi_rect.png"))
            vis.release()
        }
        val grayRoi = Mat(grayBest, roiBox).clone()
        saveMat(grayRoi, File(caseRoot, "rectify_debug/S1_gray_roi.png"))

        val Hroi = grayRoi.rows()
        val Wroi = grayRoi.cols()
        val approxCell2 = min(Hroi, Wroi) / 9.0

        // Refit/extend bowed lines
        hMasks = refineSelectedMasks(hMasks, 'h', Wroi, Hroi, approxCell2)
        vMasks = refineSelectedMasks(vMasks, 'v', Wroi, Hroi, approxCell2)

        // Intersections 8x8
        val P8 = intersectionsFromMasks(hMasks, vMasks, 3)
        // Outer ring 10x10
        var G = completeLatticeAdaptive(P8)
        // Clip to ROI bounds
        G = clipGrid(G, Wroi, Hroi)

        // S2 overlay
        val overlay = overlayMasksAndPoints(grayRoi, hMasks, vMasks, G)
        saveMat(overlay, File(caseRoot, "rectify_debug/S2_lines_and_points.png"))

        // Save points_10x10.json (ROI coordinates, [y,x])
        val ptsJson = File(caseRoot, "points_10x10.json")
        writePointsJson(ptsJson, G, roi) // include source ROI for context

        // Warp 81 tiles with shrink
        val boardWarped = matToBitmap(grayRoi)
        val boardClean = matToBitmap(grayRoi)  // reserved for future cleaning
        savePng(boardWarped, File(caseRoot, "board_warped.png"))
        savePng(boardClean, File(caseRoot, "board_clean.png"))

        val cellsDir = File(caseRoot, "cells").apply { mkdirs() }
        Log.i("Rectifier", "cellsDir=${cellsDir.absolutePath}")
        val tiles = Array(9) { Array(9) { Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888) } }
        val paths = mutableListOf<String>()
        for (r in 0 until 9) {
            for (c in 0 until 9) {
                val tileMat = warpCell(grayRoi, G, r, c, opts.tileSize, opts.shrink.toDouble())
                val tileBmp = matToBitmap(tileMat)
                val f = File(cellsDir, "r${r + 1}c${c + 1}.png")
                savePng(tileBmp, f)
                tiles[r][c] = tileBmp
                paths.add(f.absolutePath)
                tileMat.release()
            }
        }

        // cells.json
        val cellsJson = File(caseRoot, "cells.json")
        val cellsContent = """
            {
              "tiles": ${jsonStringArray(paths)},
              "roi": {"y0": ${roi.y}, "y1": ${roi.y + roi.height}, "x0": ${roi.x}, "x1": ${roi.x + roi.width}}
            }
        """.trimIndent()
        cellsJson.writeText(cellsContent, Charsets.UTF_8)

        // Done
        gray.release(); grayBest.release(); grayRoi.release()
        return Result(
            boardWarped = boardWarped,
            boardClean = boardClean,
            tiles = tiles,
            points10x10 = toPointFGrid(G),
            roiSrc = roi,
            outDir = caseRoot,
            cellsJson = cellsJson,
            pointsJson = ptsJson
        )
    }

    // ───────────────── Algorithm building blocks (OpenCV) ────────────────────

    private data class Cache(
        val g: Mat,
        val mhAll: Mat, val mvAll: Mat, val maskAll: Mat,
        val roi: Rect,
        val hsel: List<Mat>, val vsel: List<Mat>
    )

    private fun toGray(bmp: Bitmap): Mat {
        val m = Mat()
        val bgr = Mat()
        Utils.bitmapToMat(bmp, bgr)
        Imgproc.cvtColor(bgr, m, Imgproc.COLOR_BGR2GRAY)
        bgr.release()
        return m
    }

    private fun rotate(gray: Mat, angle: Double): Mat {
        if (abs(angle) < 1e-6) return gray.clone()
        val center = Point(gray.cols() / 2.0, gray.rows() / 2.0)
        val M = Imgproc.getRotationMatrix2D(center, angle, 1.0)
        val out = Mat()
        Imgproc.warpAffine(
            gray, out, M,
            Size(gray.cols().toDouble(), gray.rows().toDouble()),
            Imgproc.INTER_LINEAR, BORDER_REPLICATE, Scalar.all(0.0)
        )
        return out
    }

    private data class MaskTriple(val mh: Mat, val mv: Mat, val all: Mat)

    private fun buildOrientedMasks(gray: Mat): MaskTriple {
        val H = gray.rows(); val W = gray.cols()

        // CLAHE
        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        val eq = Mat(); clahe.apply(gray, eq)

        // adaptive threshold
        val block = max(21, ((min(H, W) / 32) or 1))
        val bw = Mat()
        Imgproc.adaptiveThreshold(
            eq, bw, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
            Imgproc.THRESH_BINARY_INV, block, 6.0
        )

        val approxCell = min(H, W) / 9.0
        val L = max(15, (0.85 * approxCell).toInt())
        val k_h1 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(L.toDouble(), 1.0))
        val k_hd = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size((L + 2).toDouble(), 3.0))
        val k_v1 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(1.0, L.toDouble()))
        val k_vd = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, (L + 2).toDouble()))

        val mask_h = Mat(); val mask_v = Mat()
        val tmp = Mat()
        Imgproc.erode(bw, tmp, k_h1); Imgproc.dilate(tmp, mask_h, k_hd)
        Imgproc.erode(bw, tmp, k_v1); Imgproc.dilate(tmp, mask_v, k_vd)

        val gap = max(5, (0.28 * approxCell).toInt())
        val hgap = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(gap.toDouble(), 1.0))
        val vgap = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(1.0, gap.toDouble()))
        Imgproc.morphologyEx(mask_h, mask_h, Imgproc.MORPH_CLOSE, hgap)
        Imgproc.morphologyEx(mask_v, mask_v, Imgproc.MORPH_CLOSE, vgap)

        val m3 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        Imgproc.morphologyEx(mask_h, mask_h, Imgproc.MORPH_CLOSE, m3)
        Imgproc.morphologyEx(mask_v, mask_v, Imgproc.MORPH_CLOSE, m3)

        val all = Mat()
        bitwise_or(mask_h, mask_v, all)

        tmp.release(); eq.release(); bw.release()
        return MaskTriple(mask_h, mask_v, all)
    }

    private data class RoiPair(val mh: Mat, val mv: Mat, val roi: Rect)

    private fun sudokuRoiFromMasks(mask_h: Mat, mask_v: Mat): RoiPair {
        val fused = Mat()
        bitwise_or(mask_h, mask_v, fused)
        val H = fused.rows(); val W = fused.cols()
        val d = max(7, (0.06 * min(H, W)).toInt())
        val k = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(d.toDouble(), d.toDouble()))
        Imgproc.dilate(fused, fused, k)

        val labels = Mat()
        val stats = Mat()
        val cent = Mat()
        val n = Imgproc.connectedComponentsWithStats(
            binarize(fused), labels, stats, cent, 8, CvType.CV_32S
        )

        if (n <= 1) {
            val roi = Rect(0, 0, W, H)
            return RoiPair(mask_h.submat(roi).clone(), mask_v.submat(roi).clone(), roi)
        }

        var bestA = -1
        var best = Rect(0, 0, W, H)
        for (i in 1 until n) {
            val x = stats.get(i, Imgproc.CC_STAT_LEFT)[0].toInt()
            val y = stats.get(i, Imgproc.CC_STAT_TOP)[0].toInt()
            val w = stats.get(i, Imgproc.CC_STAT_WIDTH)[0].toInt()
            val h = stats.get(i, Imgproc.CC_STAT_HEIGHT)[0].toInt()
            val A = w * h
            if (A > bestA) {
                bestA = A
                best = Rect(x, y, w, h)
            }
        }
        val mh = mask_h.submat(best).clone()
        val mv = mask_v.submat(best).clone()
        fused.release(); labels.release(); stats.release(); cent.release()
        return RoiPair(mh, mv, best)
    }

    private fun binarize(m: Mat): Mat {
        val out = Mat()
        Imgproc.threshold(m, out, 0.0, 255.0, Imgproc.THRESH_BINARY)
        return out
    }

    private data class Comp(val id: Int, val bbox: Rect, val median: Double)

    private fun componentsFromMask(mask: Mat): Pair<List<Comp>, Mat> {
        val lab = Mat()
        val stats = Mat()
        val cent = Mat()
        val n = Imgproc.connectedComponentsWithStats(binarize(mask), lab, stats, cent, 8, CvType.CV_32S)
        val comps = mutableListOf<Comp>()
        for (i in 1 until n) {
            val x = stats.get(i, Imgproc.CC_STAT_LEFT)[0].toInt()
            val y = stats.get(i, Imgproc.CC_STAT_TOP)[0].toInt()
            val w = stats.get(i, Imgproc.CC_STAT_WIDTH)[0].toInt()
            val h = stats.get(i, Imgproc.CC_STAT_HEIGHT)[0].toInt()
            if (w <= 0 || h <= 0) continue
            val rect = Rect(x, y, w, h)
            // median along orthogonal axis approx by center
            val median = if (w >= h) (y + h / 2.0) else (x + w / 2.0)
            comps.add(Comp(i, rect, median))
        }
        stats.release(); cent.release()
        return Pair(comps, lab)
    }

    private fun groupCollinear(mask: Mat, axis: Char, approxCell: Double): List<Mat> {
        val (comps, lab) = componentsFromMask(mask)
        if (comps.isEmpty()) { lab.release(); return emptyList() }

        val entries = comps.map {
            val coord = if (axis == 'v') (it.bbox.x + it.bbox.width / 2.0) else (it.bbox.y + it.bbox.height / 2.0)
            Triple(coord, it.id, it.bbox)
        }.sortedBy { it.first }

        val tol = 0.24 * approxCell
        val groups = mutableListOf<List<Triple<Double, Int, Rect>>>()
        val curr = mutableListOf<Triple<Double, Int, Rect>>()
        var last: Double? = null
        for (e in entries) {
            if (last == null || abs(e.first - last!!) <= tol) curr.add(e)
            else { groups.add(curr.toList()); curr.clear(); curr.add(e) }
            last = e.first
        }
        if (curr.isNotEmpty()) groups.add(curr.toList())

        val merged = mutableListOf<Mat>()
        for (g in groups) {
            val gm = Mat.zeros(mask.size(), CvType.CV_8U)
            for (e in g) {
                val id = e.second
                val single = Mat()
                Core.compare(lab, Scalar(id.toDouble()), single, Core.CMP_EQ)
                single.convertTo(single, CvType.CV_8U, 255.0)
                bitwise_or(gm, single, gm)
                single.release()
            }
            merged.add(gm)
        }
        lab.release()
        return merged
    }

    private fun select8Lines(mask: Mat, axis: Char, approxCell: Double, otherMask: Mat): List<Mat> {
        val H = mask.rows(); val W = mask.cols()
        val merged = groupCollinear(mask, axis, approxCell)

        val spanThick = arrayOf(
            Pair(0.55, 0.33), Pair(0.45, 0.45), Pair(0.35, 0.60), Pair(0.30, 0.80), Pair(0.25, 0.95), Pair(0.22, 1.10)
        )
        val gates = arrayOf(
            Quad(0.50, 0.18, 0.40, 6), Quad(0.45, 0.16, 0.35, 5), Quad(0.40, 0.14, 0.30, 5), Quad(0.36, 0.12, 0.22, 4)
        )
        val minCov = 0.18

        // build base with span/thickness check
        var base: List<Triple<Mat, Rect, Double>>
        for ((minSpanFrac, thickFrac) in spanThick) {
            val maxThickPx = max(5, (thickFrac * approxCell).toInt())
            base = merged.map { m ->
                val b = boundingRectOf(m)
                val coord = if (axis == 'h') (b.y + b.height / 2.0) else (b.x + b.width / 2.0)
                Triple(m, b, coord)
            }.filter { t ->
                val b = t.second
                if (axis == 'h') b.width >= (minSpanFrac * W).toInt() && b.height <= maxThickPx
                else b.height >= (minSpanFrac * H).toInt() && b.width <= maxThickPx
            }
            if (base.size < 8) continue

            for (g in gates) {
                val centerMargin = g.a * approxCell
                val ignorePx = (g.c * approxCell).toInt()
                val cands = mutableListOf<Pair<Double, Mat>>()
                for ((m, b, coord) in base) {
                    if (axis == 'h') {
                        if (coord < centerMargin || (H - coord) < centerMargin) continue
                    } else {
                        if (coord < centerMargin || (W - coord) < centerMargin) continue
                    }
                    if (centralCoverage(m, axis, g.b) < minCov) continue
                    val crossings = countCrossings(m, otherMask, axis, 2, ignorePx, g.b)
                    if (crossings >= g.d) cands.add(Pair(coord, m))
                }
                if (cands.size >= 8) return chooseWindow(cands)
            }

            // fallback: coverage + spacing only
            val fb = gates.last().b
            val cands = base.filter { (m, _, _) -> centralCoverage(m, axis, fb) >= minCov }
                .map { Pair(it.third, it.first) }
            if (cands.size >= 8) return chooseWindow(cands)
        }

        // final resort: most central with coverage
        if (merged.size >= 8) {
            val center = if (axis == 'h') H / 2.0 else W / 2.0
            val fb = 0.18
            val filt = merged.map { m ->
                val b = boundingRectOf(m)
                val coord = if (axis == 'h') (b.y + b.height / 2.0) else (b.x + b.width / 2.0)
                Triple(abs(center - coord), m, centralCoverage(m, axis, fb))
            }.filter { it.third >= minCov }
                .sortedBy { it.first }
            if (filt.size >= 8) return filt.take(8).map { it.second }
        }

        throw RuntimeException("Not enough components to select 8 ${axis}-lines")
    }

    // ───── NEW: robust center-biased fallback used in the robust sweep
    private fun pick8CenterBiased(mask: Mat, axis: Char, approxCell: Double): List<Mat> {
        val H = mask.rows(); val W = mask.cols()
        val merged = groupCollinear(mask, axis, approxCell)
        if (merged.size < 8) throw RuntimeException("pick8CenterBiased: not enough components")

        val center = if (axis == 'h') H / 2.0 else W / 2.0
        val scored = merged.map { m ->
            val b = boundingRectOf(m)
            val coord = if (axis == 'h') (b.y + b.height / 2.0) else (b.x + b.width / 2.0)
            val cov = centralCoverage(m, axis, 0.18)
            Triple(abs(center - coord), m, cov)
        }.filter { it.third >= 0.12 }
            .sortedBy { it.first }

        val cands = scored.map { Pair(
            if (axis == 'h') boundingRectOf(it.second).y + boundingRectOf(it.second).height / 2.0
            else boundingRectOf(it.second).x + boundingRectOf(it.second).width / 2.0,
            it.second
        ) }

        return chooseWindow(cands)
    }

    private data class Quad(val a: Double, val b: Double, val c: Double, val d: Int)

    private fun boundingRectOf(m: Mat): Rect {
        val contours = mutableListOf<MatOfPoint>()
        Imgproc.findContours(m, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        var best = Rect(0, 0, 0, 0)
        var bestA = -1
        for (c in contours) {
            val r = Imgproc.boundingRect(c)
            val A = r.width * r.height
            if (A > bestA) { bestA = A; best = r }
        }
        contours.forEach { it.release() }
        return if (bestA >= 0) best else Rect(0, 0, m.cols(), m.rows())
    }

    private fun centralCoverage(lineMask: Mat, axis: Char, bandFrac: Double): Double {
        val H = lineMask.rows(); val W = lineMask.cols()
        return if (axis == 'h') {
            val y0 = (bandFrac * H).toInt(); val y1 = ((1.0 - bandFrac) * H).toInt()
            val roi = lineMask.submat(Rect(0, y0, W, y1 - y0))
            val covered = (0 until W).count { x -> Core.countNonZero(roi.col(x)) > 0 }
            roi.release()
            covered / max(1.0, W.toDouble())
        } else {
            val x0 = (bandFrac * W).toInt(); val x1 = ((1.0 - bandFrac) * W).toInt()
            val roi = lineMask.submat(Rect(x0, 0, x1 - x0, H))
            val covered = (0 until H).count { y -> Core.countNonZero(roi.row(y)) > 0 }
            roi.release()
            covered / max(1.0, H.toDouble())
        }
    }

    private fun countCrossings(
        lineMask: Mat,
        otherMask: Mat,
        axis: Char,
        dilatePx: Int,
        ignoreMarginPx: Int,
        bandFrac: Double
    ): Int {
        val H = lineMask.rows(); val W = lineMask.cols()
        val band = Mat.zeros(H, W, CvType.CV_8U)
        if (axis == 'h') {
            val y0 = (bandFrac * H).toInt(); val y1 = ((1.0 - bandFrac) * H).toInt()
            Imgproc.rectangle(band, Rect(0, y0, W, y1 - y0), Scalar(255.0), -1)
        } else {
            val x0 = (bandFrac * W).toInt(); val x1 = ((1.0 - bandFrac) * W).toInt()
            Imgproc.rectangle(band, Rect(x0, 0, x1 - x0, H), Scalar(255.0), -1)
        }
        val cand = Mat(); bitwise_and(lineMask, band, cand)
        val k = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(2 * dilatePx + 1.0, 2 * dilatePx + 1.0))
        val inter = Mat()
        val d1 = Mat(); val d2 = Mat()
        Imgproc.dilate(cand, d1, k); Imgproc.dilate(otherMask, d2, k)
        bitwise_and(d1, d2, inter)

        if (ignoreMarginPx > 0) {
            if (axis == 'h') {
                inter.rowRange(0, ignoreMarginPx).setTo(Scalar(0.0))
                inter.rowRange(H - ignoreMarginPx, H).setTo(Scalar(0.0))
            } else {
                inter.colRange(0, ignoreMarginPx).setTo(Scalar(0.0))
                inter.colRange(W - ignoreMarginPx, W).setTo(Scalar(0.0))
            }
        }

        val labels = Mat()
        val n = Imgproc.connectedComponents(binarize(inter), labels)
        labels.release(); k.release(); d1.release(); d2.release(); band.release(); cand.release()
        inter.release()
        return max(0, n - 1)
    }

    private fun lineCenters(lineMasks: List<Mat>, axis: Char): DoubleArray {
        val coords = DoubleArray(lineMasks.size) { 0.0 }
        for ((i, m) in lineMasks.withIndex()) {
            val nonzero = Core.countNonZero(m)
            if (nonzero == 0) { coords[i] = 0.0; continue }
            val b = boundingRectOf(m)
            coords[i] = if (axis == 'h') (b.y + b.height / 2.0) else (b.x + b.width / 2.0)
        }
        return coords.sortedArray()
    }

    private fun cvGaps(coords: DoubleArray): Double {
        if (coords.size < 2) return 1e6
        val gaps = DoubleArray(coords.size - 1) { i -> coords[i + 1] - coords[i] }
        val mu = gaps.average()
        return if (mu <= 1e-6) 1e6 else (stddev(gaps) / mu)
    }

    private fun stddev(a: DoubleArray): Double {
        val m = a.average()
        var s = 0.0
        for (v in a) s += (v - m) * (v - m)
        return sqrt(s / max(1, a.size - 1).toDouble())
    }

    private fun refineSelectedMasks(masks: List<Mat>, axis: Char, W: Int, H: Int, approxCell: Double): List<Mat> {
        val out = mutableListOf<Mat>()
        for (m in masks) {
            val b = boundingRectOf(m)
            val span = if (axis == 'h') b.width else b.height
            val need = span < (0.90 * (if (axis == 'h') W else H))
            out.add(if (need) refitMaskLine(m, axis, W, H, approxCell) else m)
        }
        return out
    }

    private fun refitMaskLine(m: Mat, axis: Char, W: Int, H: Int, approxCell: Double): Mat {
        val nz = Mat()
        Core.findNonZero(m, nz)
        if (nz.empty() || nz.rows() < 50) { nz.release(); return m }
        val thickness = max(2, (0.12 * approxCell).toInt())
        val out = Mat.zeros(m.size(), CvType.CV_8U)

        if (axis == 'h') {
            // Fit y = a2 x^2 + a1 x + a0 (or linear fallback)
            val xs = DoubleArray(nz.rows()) { i -> nz.get(i, 0)[0] } // x
            val ys = DoubleArray(nz.rows()) { i -> nz.get(i, 0)[1] } // y
            val coeff = polyfit(xs, ys, 2) ?: polyfit(xs, ys, 1)!!
            val xx = (0 until W).map { it.toDouble() }
            val yy = xx.map { polyval(coeff, it).coerceIn(0.0, H - 1.0) }
            val polyPts = MatOfPoint(*xx.zip(yy).map { Point(it.first, it.second) }.toTypedArray())
            Imgproc.polylines(out, listOf(polyPts), false, Scalar(255.0), thickness)
            polyPts.release()
        } else {
            val xs = DoubleArray(nz.rows()) { i -> nz.get(i, 0)[0] }
            val ys = DoubleArray(nz.rows()) { i -> nz.get(i, 0)[1] }
            val coeff = polyfit(ys, xs, 2) ?: polyfit(ys, xs, 1)!!
            val yy = (0 until H).map { it.toDouble() }
            val xx = yy.map { polyval(coeff, it).coerceIn(0.0, W - 1.0) }
            val polyPts = MatOfPoint(*xx.zip(yy).map { Point(it.first, it.second) }.toTypedArray())
            Imgproc.polylines(out, listOf(polyPts), false, Scalar(255.0), thickness)
            polyPts.release()
        }

        nz.release()
        return out
    }

    // simple polyfit helpers
    private fun polyfit(x: DoubleArray, y: DoubleArray, deg: Int): DoubleArray? {
        return try {
            // Use normal equations (not super stable but fine for our line-like fits)
            val X = Mat.zeros(x.size, deg + 1, CvType.CV_64F)
            val Y = Mat.zeros(x.size, 1, CvType.CV_64F)
            for (i in x.indices) {
                var v = 1.0
                for (d in 0..deg) { X.put(i, d, v); v *= x[i] }
                Y.put(i, 0, y[i])
            }
            val Xt = X.t()
            val XtX = Mat()
            gemm(Xt, X, 1.0, Mat(), 0.0, XtX)
            val XtY = Mat()
            gemm(Xt, Y, 1.0, Mat(), 0.0, XtY)
            val coeff = Mat()
            solve(XtX, XtY, coeff, DECOMP_SVD)
            val arr = DoubleArray(deg + 1) { d -> coeff.get(d, 0)[0] }
            X.release(); Y.release(); Xt.release(); XtX.release(); XtY.release(); coeff.release()
            arr
        } catch (_: Exception) { null }
    }
    private fun polyval(coeff: DoubleArray, x: Double): Double {
        var v = 0.0
        var p = 1.0
        for (c in coeff) { v += c * p; p *= x }
        return v
    }

    private fun intersectionsFromMasks(hm: List<Mat>, vm: List<Mat>, dilatePx: Int): Array<Array<Point>> {
        val k = Imgproc.getStructuringElement(
            Imgproc.MORPH_ELLIPSE,
            Size((2 * dilatePx + 1).toDouble(), (2 * dilatePx + 1).toDouble())
        )
        val vd = vm.map { m -> val t = Mat(); Imgproc.dilate(m, t, k); t }
        val pts = Array(8) { Array(8) { Point(0.0, 0.0) } }
        for (i in 0 until 8) {
            val hd = Mat(); Imgproc.dilate(hm[i], hd, k)
            for (j in 0 until 8) {
                val inter = Mat()
                bitwise_and(hd, vd[j], inter)
                val nz = Mat()
                Core.findNonZero(inter, nz)
                if (!nz.empty()) {
                    // median of xs, ys
                    val xs = DoubleArray(nz.rows()) { t -> nz.get(t, 0)[0] }
                    val ys = DoubleArray(nz.rows()) { t -> nz.get(t, 0)[1] }
                    xs.sort(); ys.sort()
                    pts[i][j] = Point(xs[xs.size / 2], ys[ys.size / 2])
                } else {
                    // distance transform fallback (approx)
                    val inv = Mat()
                    bitwise_not(inter, inv)
                    val dt = Mat()
                    Imgproc.distanceTransform(inv, dt, Imgproc.DIST_L2, 3)
                    val mm = Core.minMaxLoc(dt)
                    pts[i][j] = Point(mm.maxLoc.x, mm.maxLoc.y)
                    inv.release(); dt.release()
                }
                nz.release(); inter.release()
            }
            hd.release()
        }
        vd.forEach { it.release() }
        return pts
    }

    private fun completeLatticeAdaptive(P8: Array<Array<Point>>): Array<Array<Point>> {
        val G = Array(10) { Array(10) { Point(0.0, 0.0) } }
        for (i in 0 until 8) for (j in 0 until 8) G[i + 1][j + 1] = P8[i][j]

        // columns → top/bottom
        for (j in 0 until 8) {
            val xs = DoubleArray(8) { i -> P8[i][j].x }
            val ys = DoubleArray(8) { i -> P8[i][j].y }
            val (xt, xb, yt, yb) = extrapAdaptiveXY(xs, ys)
            val top = clampExtrap(P8[0][j], P8[1][j], Point(xt, yt), 1.6)
            val bot = clampExtrap(P8[7][j], P8[6][j], Point(xb, yb), 1.6)
            G[0][j + 1] = top; G[9][j + 1] = bot
        }
        // rows → left/right
        for (i in 0 until 8) {
            val xs = DoubleArray(8) { j -> P8[i][j].x }
            val ys = DoubleArray(8) { j -> P8[i][j].y }
            val (xl, xr, yl, yr) = extrapAdaptiveXY(xs, ys)
            val lef = clampExtrap(P8[i][0], P8[i][1], Point(xl, yl), 1.6)
            val rig = clampExtrap(P8[i][7], P8[i][6], Point(xr, yr), 1.6)
            G[i + 1][0] = lef; G[i + 1][9] = rig
        }

        // corners from fitted border intersections
        fun fitH(points: Array<Point>) = fitLineYX(points) // y = a*x + b
        fun fitV(points: Array<Point>) = fitLineXY(points) // x = a*y + b
        fun inter(h: Pair<Double, Double>, v: Pair<Double, Double>): Point {
            val (ah, bh) = h; val (av, bv) = v
            val denom = 1.0 - ah * av
            val y = if (abs(denom) > 1e-9) (ah * bv + bh) / denom else bh
            val x = av * y + bv
            return Point(x, y)
        }

        val topPts    = G[0].slice(1..8).toTypedArray()
        val bottomPts = G[9].slice(1..8).toTypedArray()
        val leftPts   = Array(8) { k -> G[k + 1][0] }
        val rightPts  = Array(8) { k -> G[k + 1][9] }

        G[0][0] = inter(fitH(topPts.take(3).toTypedArray()),       fitV(leftPts.take(3).toTypedArray()))
        G[0][9] = inter(fitH(topPts.takeLast(3).toTypedArray()),    fitV(rightPts.take(3).toTypedArray()))
        G[9][0] = inter(fitH(bottomPts.take(3).toTypedArray()),     fitV(leftPts.takeLast(3).toTypedArray()))
        G[9][9] = inter(fitH(bottomPts.takeLast(3).toTypedArray()), fitV(rightPts.takeLast(3).toTypedArray()))

        return G
    }

    private fun fitLineYX(points: Array<Point>): Pair<Double, Double> {
        if (points.size < 2) return Pair(0.0, points.firstOrNull()?.y ?: 0.0)
        val X = Mat.zeros(points.size, 2, CvType.CV_64F)
        val Y = Mat.zeros(points.size, 1, CvType.CV_64F)
        for ((i, p) in points.withIndex()) { X.put(i, 0, p.x); X.put(i, 1, 1.0); Y.put(i, 0, p.y) }
        val Xt = X.t()
        val XtX = Mat(); gemm(Xt, X, 1.0, Mat(), 0.0, XtX)
        val XtY = Mat(); gemm(Xt, Y, 1.0, Mat(), 0.0, XtY)
        val AB = Mat(); solve(XtX, XtY, AB, DECOMP_SVD)
        val a = AB.get(0, 0)[0]; val b = AB.get(1, 0)[0]
        X.release(); Y.release(); Xt.release(); XtX.release(); XtY.release(); AB.release()
        return Pair(a, b)
    }
    private fun fitLineXY(points: Array<Point>): Pair<Double, Double> {
        if (points.size < 2) return Pair(0.0, points.firstOrNull()?.x ?: 0.0)
        val X = Mat.zeros(points.size, 2, CvType.CV_64F)
        val Y = Mat.zeros(points.size, 1, CvType.CV_64F)
        for ((i, p) in points.withIndex()) { X.put(i, 0, p.y); X.put(i, 1, 1.0); Y.put(i, 0, p.x) }
        val Xt = X.t()
        val XtX = Mat(); gemm(Xt, X, 1.0, Mat(), 0.0, XtX)
        val XtY = Mat(); gemm(Xt, Y, 1.0, Mat(), 0.0, XtY)
        val AB = Mat(); solve(XtX, XtY, AB, DECOMP_SVD)
        val a = AB.get(0, 0)[0]; val b = AB.get(1, 0)[0]
        X.release(); Y.release(); Xt.release(); XtX.release(); XtY.release(); AB.release()
        return Pair(a, b)
    }

    private fun extrapAdaptiveXY(xs: DoubleArray, ys: DoubleArray): QuadD {
        val (f1x, m1x) = fitMSE(xs, 1); val (f2x, m2x) = fitMSE(xs, 2)
        val (f1y, m1y) = fitMSE(ys, 1); val (f2y, m2y) = fitMSE(ys, 2)
        val useQ = (m2x < m1x * 0.85) || (m2y < m1y * 0.85)
        val fx = if (useQ) f2x else f1x
        val fy = if (useQ) f2y else f1y
        return QuadD(fx(0.0), fx(9.0), fy(0.0), fy(9.0))
    }
    private data class QuadD(val a: Double, val b: Double, val c: Double, val d: Double)

    private fun fitMSE(v: DoubleArray, deg: Int): Pair<(Double) -> Double, Double> {
        val idx = DoubleArray(8) { (it + 1).toDouble() }
        val coeff = polyfit(idx, v, deg)!!
        fun f(t: Double): Double = polyval(coeff, t)
        val mse = idx.map { (f(it) - v[it.toInt() - 1]).pow(2.0) }.average()
        return Pair(::f, mse)
    }

    private fun clampExtrap(pRef: Point, pNext: Point, pPred: Point, factor: Double): Point {
        val sx = pRef.x - pNext.x; val sy = pRef.y - pNext.y
        val rx = pPred.x - pRef.x; val ry = pPred.y - pRef.y
        val nstep = hypot(sx, sy) + 1e-6
        val nref = hypot(rx, ry)
        return if (nref > factor * nstep) {
            val scale = factor * nstep / nref
            Point(pRef.x + rx * scale, pRef.y + ry * scale)
        } else pPred
    }

    private fun clipGrid(G: Array<Array<Point>>, W: Int, H: Int): Array<Array<Point>> {
        for (i in 0 until 10) for (j in 0 until 10) {
            val x = G[i][j].x.coerceIn(0.0, W - 1.0)
            val y = G[i][j].y.coerceIn(0.0, H - 1.0)
            G[i][j] = Point(x, y)
        }
        return G
    }

    private fun overlayMasksAndPoints(gray: Mat, h: List<Mat>, v: List<Mat>, G: Array<Array<Point>>): Mat {
        val vis = Mat()
        Imgproc.cvtColor(gray, vis, Imgproc.COLOR_GRAY2BGR)
        val H = gray.rows(); val W = gray.cols()
        val thick = max(2, (min(H, W) * 0.003).toInt())
        val radius = max(2, (min(H, W) * 0.008).toInt())

        fun drawMask(m: Mat, color: Scalar) {
            val contours = mutableListOf<MatOfPoint>()
            Imgproc.findContours(m, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            Imgproc.drawContours(vis, contours, -1, color, thick)
            contours.forEach { it.release() }
        }
        h.forEach { drawMask(it, Scalar(0.0, 255.0, 0.0)) }
        v.forEach { drawMask(it, Scalar(0.0, 165.0, 255.0)) }

        for (i in 0 until 10) for (j in 0 until 10) {
            val p = G[i][j]
            Imgproc.circle(vis, p, radius, Scalar(0.0, 0.0, 255.0), -1)
        }
        return vis
    }

    private fun warpCell(gray: Mat, G: Array<Array<Point>>, r: Int, c: Int, out: Int, shrink: Double): Mat {
        val quad = arrayOf(G[r][c], G[r][c + 1], G[r + 1][c + 1], G[r + 1][c])
        val cx = quad.map { it.x }.average(); val cy = quad.map { it.y }.average()
        val shrunk = arrayOf(
            Point(cx + (quad[0].x - cx) * (1.0 - shrink), cy + (quad[0].y - cy) * (1.0 - shrink)),
            Point(cx + (quad[1].x - cx) * (1.0 - shrink), cy + (quad[1].y - cy) * (1.0 - shrink)),
            Point(cx + (quad[2].x - cx) * (1.0 - shrink), cy + (quad[2].y - cy) * (1.0 - shrink)),
            Point(cx + (quad[3].x - cx) * (1.0 - shrink), cy + (quad[3].y - cy) * (1.0 - shrink))
        )
        val src = MatOfPoint2f(*shrunk)
        val dst = MatOfPoint2f(
            Point(0.0, 0.0), Point((out - 1).toDouble(), 0.0),
            Point((out - 1).toDouble(), (out - 1).toDouble()), Point(0.0, (out - 1).toDouble())
        )
        val M = Imgproc.getPerspectiveTransform(src, dst)
        val tile = Mat()
        Imgproc.warpPerspective(gray, tile, M, Size(out.toDouble(), out.toDouble()), Imgproc.INTER_LINEAR)
        src.release(); dst.release(); M.release()
        return tile
    }

    private fun writePointsJson(f: File, G: Array<Array<Point>>, roiSrc: Rect) {
        val flat = buildString {
            append("[")
            var first = true
            for (i in 0 until 10) for (j in 0 until 10) {
                val p = G[i][j]
                if (!first) append(",") else first = false
                append("[${p.y},${p.x}]")
            }
            append("]")
        }
        val s = """
          {
            "points": $flat,
            "grid_shape": [10,10],
            "coord_space": "roi",
            "roi": {"y0": ${roiSrc.y}, "y1": ${roiSrc.y + roiSrc.height}, "x0": ${roiSrc.x}, "x1": ${roiSrc.x + roiSrc.width}}
          }
        """.trimIndent()
        f.parentFile?.mkdirs()
        f.writeText(s, Charsets.UTF_8)
    }

    // ───────── small helpers

    private fun precropHeaderMargins(gray: Mat): Mat {
        val H = gray.rows(); val W = gray.cols()
        val y0 = (0.12 * H).toInt()
        val y1 = H - (0.02 * H).toInt()
        val x0 = (0.06 * W).toInt()
        val x1 = W - (0.06 * W).toInt()
        return Mat(gray, Rect(x0, y0, x1 - x0, y1 - y0)).clone()
    }

    private fun matToBitmap(m: Mat): Bitmap {
        val bgr = if (m.channels() == 1) {
            val c = Mat()
            Imgproc.cvtColor(m, c, Imgproc.COLOR_GRAY2BGR); c
        } else m
        val bmp = Bitmap.createBitmap(bgr.cols(), bgr.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(bgr, bmp)
        if (bgr !== m) bgr.release()
        return bmp
    }

    // Android-safe Mat writer (PNG via Bitmap)
    private fun saveMat(m: Mat, f: File) {
        f.parentFile?.mkdirs()
        val bmp = matToBitmap(m)
        FileOutputStream(f).use { out -> bmp.compress(Bitmap.CompressFormat.PNG, 100, out) }
    }

    private fun savePng(b: Bitmap, f: File) {
        f.parentFile?.mkdirs()
        FileOutputStream(f).use { out -> b.compress(Bitmap.CompressFormat.PNG, 100, out) }
    }

    private fun jsonStringArray(paths: List<String>): String {
        val esc = paths.joinToString(",") { "\"${it.replace("\\", "/")}\"" }
        return "[$esc]"
    }

    private fun toPointFGrid(G: Array<Array<Point>>): Array<Array<PointF>> {
        return Array(10) { i -> Array(10) { j -> PointF(G[i][j].x.toFloat(), G[i][j].y.toFloat()) } }
    }

    private fun Rect.toAndroidRect(): android.graphics.Rect =
        android.graphics.Rect(this.x, this.y, this.x + this.width, this.y + this.height)

    private fun sequenceFromRange(r: IntProgression): List<Int> = r.toList()

    // ───── NEW: chooseWindow helper (tightest span window of size 8)
    private fun chooseWindow(cands: List<Pair<Double, Mat>>, want: Int = 8): List<Mat> {
        val sorted = cands.sortedBy { it.first }
        if (sorted.size == want) return sorted.map { it.second }
        require(sorted.size >= want) { "chooseWindow: not enough candidates" }
        var bestI = 0
        var bestSpan = Double.POSITIVE_INFINITY
        for (i in 0..(sorted.size - want)) {
            val span = sorted[i + want - 1].first - sorted[i].first
            if (span < bestSpan) { bestSpan = span; bestI = i }
        }
        return sorted.subList(bestI, bestI + want).map { it.second }
    }
}