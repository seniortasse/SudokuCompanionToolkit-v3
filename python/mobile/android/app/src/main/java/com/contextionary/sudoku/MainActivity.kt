package com.contextionary.sudoku

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.SystemClock
import android.util.Size
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var detector: Detector
    private lateinit var analyzerExecutor: ExecutorService

    private var frameIndex = 0
    private var lastInferMs = 0L
    private val minInferIntervalMs = 120L
    private val skipEvery = 1

    private val askCameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) startCamera() else finish() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // DEV ONLY: quick OpenCV smoke check (kept commented by default)
        /*
        val mat = org.opencv.core.Mat(100, 100, org.opencv.core.CvType.CV_8UC1)
        org.opencv.imgproc.Imgproc.circle(mat, org.opencv.core.Point(50.0, 50.0), 20, org.opencv.core.Scalar(255.0))
        mat.release()
        */

        previewView = findViewById(R.id.preview)
        overlay = findViewById(R.id.overlay)
        previewView.scaleType = PreviewView.ScaleType.FIT_CENTER
        overlay.setUseFillCenter(previewView.scaleType == PreviewView.ScaleType.FILL_CENTER)

        previewView.viewTreeObserver.addOnGlobalLayoutListener {
            android.util.Log.d(
                "MainActivity",
                "PreviewView laid out: ${previewView.width}x${previewView.height}, scaleType=${previewView.scaleType}"
            )
        }

        detector = Detector(this, "best_float32.tflite", "labels.txt")
        analyzerExecutor = Executors.newSingleThreadExecutor()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) startCamera() else askCameraPermission.launch(Manifest.permission.CAMERA)
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::analyzerExecutor.isInitialized) analyzerExecutor.shutdown()
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()

            val preview = Preview.Builder()
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(960, 720))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            val HUD_THRESH = 0.55f
            val HUD_MAX_DETS = 6

            analysis.setAnalyzer(analyzerExecutor) { proxy ->
                try {
                    frameIndex++
                    if (frameIndex % (skipEvery + 1) != 0) { proxy.close(); return@setAnalyzer }

                    val now = SystemClock.elapsedRealtime()
                    if (now - lastInferMs < minInferIntervalMs) { proxy.close(); return@setAnalyzer }
                    lastInferMs = now

                    val bmp = proxy.toBitmap() ?: run { proxy.close(); return@setAnalyzer }

                    // --- DEV smoke op: exercise OpenCV path when flag is ON ----
                    if (DevFlags.showRectifierDebug) {
                        try {
                            val mat = org.opencv.core.Mat(100, 100, org.opencv.core.CvType.CV_8UC1)
                            org.opencv.imgproc.Imgproc.circle(
                                mat,
                                org.opencv.core.Point(50.0, 50.0),
                                20,
                                org.opencv.core.Scalar(255.0),
                                2
                            )
                            mat.release()
                        } catch (t: Throwable) {
                            android.util.Log.e("OpenCV", "Smoke op failed", t)
                        }
                    }

                    val dets = detector.infer(bmp, scoreThresh = HUD_THRESH, maxDets = HUD_MAX_DETS)

                    runOnUiThread {
                        overlay.setSourceSize(bmp.width, bmp.height)
                        overlay.updateBoxes(dets, HUD_THRESH, HUD_MAX_DETS)
                    }
                } catch (_: Throwable) {
                } finally {
                    proxy.close()
                }
            }

            provider.unbindAll()
            provider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis
            )
        }, ContextCompat.getMainExecutor(this))
    }
}