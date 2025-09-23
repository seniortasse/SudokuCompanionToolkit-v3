#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <opencv2/opencv.hpp>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "WarpTiler", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "WarpTiler", __VA_ARGS__)

// --- Helpers: Bitmap <-> Mat ---

static cv::Mat BitmapToMat(JNIEnv* env, jobject bitmap) {
    AndroidBitmapInfo info;
    void* pixels = nullptr;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) {
        throw std::runtime_error("AndroidBitmap_getInfo failed");
    }
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        throw std::runtime_error("Bitmap format must be ARGB_8888");
    }
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
        throw std::runtime_error("AndroidBitmap_lockPixels failed");
    }
    cv::Mat rgba(info.height, info.width, CV_8UC4, pixels);
    cv::Mat bgr;
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
    AndroidBitmap_unlockPixels(env, bitmap);
    return bgr;
}

static jobject MatToBitmap(JNIEnv* env, const cv::Mat& bgr) {
    cv::Mat rgba;
    cv::cvtColor(bgr, rgba, cv::COLOR_BGR2RGBA);

    jclass bmpCls = env->FindClass("android/graphics/Bitmap");
    jclass cfgCls = env->FindClass("android/graphics/Bitmap$Config");
    jmethodID createBitmap = env->GetStaticMethodID(
            bmpCls, "createBitmap",
            "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;"
    );
    jmethodID valueOf = env->GetStaticMethodID(
            cfgCls, "valueOf",
            "(Ljava/lang/String;)Landroid/graphics/Bitmap$Config;"
    );
    jstring argbName = env->NewStringUTF("ARGB_8888");
    jobject argb8888 = env->CallStaticObjectMethod(cfgCls, valueOf, argbName);
    env->DeleteLocalRef(argbName);

    jobject outBmp = env->CallStaticObjectMethod(
            bmpCls, createBitmap,
            rgba.cols, rgba.rows, argb8888
    );

    AndroidBitmapInfo info;
    void* pixels = nullptr;
    AndroidBitmap_getInfo(env, outBmp, &info);
    AndroidBitmap_lockPixels(env, outBmp, &pixels);
    cv::Mat dst(info.height, info.width, CV_8UC4, pixels);
    rgba.copyTo(dst);
    AndroidBitmap_unlockPixels(env, outBmp);
    return outBmp;
}

// --- JNI: warpBoard ---

extern "C"
JNIEXPORT jobject JNICALL
Java_com_contextionary_sudoku_NativeBoard_warpBoard(
        JNIEnv* env, jclass /*cls*/,
        jobject sourceBitmap,  // ARGB_8888
        jfloatArray corners8f, // [x0,y0, x1,y1, x2,y2, x3,y3] TL,TR,BR,BL
        jint boardSize         // e.g., 900
) {
    try {
        if (sourceBitmap == nullptr || corners8f == nullptr || boardSize <= 0) {
            LOGE("warpBoard: bad args");
            return nullptr;
        }
        cv::Mat src = BitmapToMat(env, sourceBitmap);
        jfloat* c = env->GetFloatArrayElements(corners8f, nullptr);
        cv::Point2f srcPts[4] = {
                {c[0], c[1]}, {c[2], c[3]}, {c[4], c[5]}, {c[6], c[7]}
        };
        env->ReleaseFloatArrayElements(corners8f, c, 0);

        cv::Point2f dstPts[4] = {
                {0.f, 0.f},
                {float(boardSize-1), 0.f},
                {float(boardSize-1), float(boardSize-1)},
                {0.f, float(boardSize-1)}
        };
        cv::Mat H = cv::getPerspectiveTransform(srcPts, dstPts);
        cv::Mat warped;
        cv::warpPerspective(src, warped, H, cv::Size(boardSize, boardSize), cv::INTER_LINEAR);
        return MatToBitmap(env, warped);
    } catch (const std::exception& e) {
        LOGE("warpBoard error: %s", e.what());
        return nullptr;
    }
}

// --- JNI: tile81 ---

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_contextionary_sudoku_NativeBoard_tile81(
        JNIEnv* env, jclass /*cls*/,
        jobject boardBitmap, // ARGB_8888, square (e.g., 900x900)
        jint tileSize        // e.g., 64
) {
    try {
        if (boardBitmap == nullptr || tileSize <= 0) {
            LOGE("tile81: bad args");
            return nullptr;
        }
        cv::Mat board = BitmapToMat(env, boardBitmap);
        if (board.cols != board.rows) {
            throw std::runtime_error("Board must be square");
        }
        const int W = board.cols;
        const int cell = W / 9;

        jclass bmpCls = env->FindClass("android/graphics/Bitmap");
        jobjectArray arr = env->NewObjectArray(81, bmpCls, nullptr);

        for (int r = 0; r < 9; ++r) {
            for (int c = 0; c < 9; ++c) {
                const int x0 = c * cell;
                const int y0 = r * cell;
                const int x1 = std::min(x0 + cell, W);
                const int y1 = std::min(y0 + cell, W);
                cv::Rect roi(x0, y0, x1 - x0, y1 - y0);
                cv::Mat crop = board(roi).clone();

                // Optional inner-crop ratio (e.g., 0.92) could be added here if desired

                cv::Mat resized;
                cv::resize(crop, resized, cv::Size(tileSize, tileSize), 0, 0, cv::INTER_AREA);
                jobject tileBmp = MatToBitmap(env, resized);
                env->SetObjectArrayElement(arr, r * 9 + c, tileBmp);
            }
        }
        return arr;
    } catch (const std::exception& e) {
        LOGE("tile81 error: %s", e.what());
        return nullptr;
    }
}