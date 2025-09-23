#include <jni.h>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include <cmath>
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "Refiner", __VA_ARGS__)

static std::pair<std::array<float,8>, float>
refine_in_roi_nv21(const uint8_t* nv21, int W, int H, const cv::Rect& roi) {
    cv::Mat yuv(H + H/2, W, CV_8UC1, (void*)nv21);
    cv::Mat bgr; cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV21);
    cv::Mat img = bgr(roi).clone();

    cv::Mat gray; cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);
    cv::Mat edges; cv::Canny(gray, edges, 60, 160);

    auto lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_ADV);
    std::vector<cv::Vec4f> segs; lsd->detect(edges, segs);
    if (segs.empty()) return {{0,0,0,0,0,0,0,0}, 0.f};

    // angles in [0,pi)
    std::vector<float> ang; ang.reserve(segs.size());
    for (auto&s: segs){ float a=std::atan2(s[3]-s[1], s[2]-s[0]); a=fmod(a+CV_PI, CV_PI); ang.push_back(a); }

    cv::Mat samples((int)ang.size(),1,CV_32F, ang.data()), labels, centers;
    cv::kmeans(samples, 2, labels,
               cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::MAX_ITER, 30, 1e-3),
               4, cv::KMEANS_PP_CENTERS, centers);

    std::vector<cv::Vec4f> fam[2];
    for (int i=0;i<labels.rows;i++) fam[labels.at<int>(i)].push_back(segs[i]);
    if (fam[0].size()<6 || fam[1].size()<6) return {{0,0,0,0,0,0,0,0}, 0.f};

    auto to_normal = [](const cv::Vec4f&s){
        float a = std::atan2(s[3]-s[1], s[2]-s[0]);
        float n0 = std::sin(a), n1 = -std::cos(a);
        float cx = 0.5f*(s[0]+s[2]), cy = 0.5f*(s[1]+s[3]);
        float rho = n0*cx + n1*cy;
        float theta = fmod(a + CV_PI/2, CV_PI);
        return cv::Vec2f(rho, theta);
    };
    auto norm_merge = [&](const std::vector<cv::Vec4f>& in){
        std::vector<cv::Vec2f> L; L.reserve(in.size());
        for (auto &s: in) L.push_back(to_normal(s));
        std::sort(L.begin(), L.end(), [](auto&a, auto&b){ return a[0]<b[0]; });
        std::vector<cv::Vec2f> M;
        for (auto &v: L) {
            if (M.empty() || std::abs(v[0]-M.back()[0])>8 || std::abs(v[1]-M.back()[1])>0.05) M.push_back(v);
            else { M.back()[0]=0.5f*(M.back()[0]+v[0]); M.back()[1]=0.5f*(M.back()[1]+v[1]); }
        }
        return M;
    };
    auto L0 = norm_merge(fam[0]);
    auto L1 = norm_merge(fam[1]);
    if (L0.size()<6 || L1.size()<6) return {{0,0,0,0,0,0,0,0}, 0.f};

    auto gapsStd = [](const std::vector<cv::Vec2f>& L){
        if (L.size()<3) return 1e6f;
        std::vector<float> g; g.reserve(L.size()-1);
        for (int i=1;i<(int)L.size();++i) g.push_back(L[i][0]-L[i-1][0]);
        cv::Scalar mean, stddev; cv::meanStdDev(g, mean, stddev);
        return (float)(stddev[0]+1e-6);
    };
    float gp0 = gapsStd(L0), gp1 = gapsStd(L1);
    float score_period = 1.f/gp0 + 1.f/gp1;
    auto countScore = [](int n){ float d=n-10; return std::exp(-(d*d)/18.f); };
    float score_count = countScore((int)L0.size()) + countScore((int)L1.size());

    auto to_abc = [](cv::Vec2f lh){ float a=std::cos(lh[1]), b=std::sin(lh[1]), c=-lh[0]; return cv::Vec3f(a,b,c); };
    auto inter = [](cv::Vec3f l1, cv::Vec3f l2){
        float d = l1[0]*l2[1]-l2[0]*l1[1];
        if (std::abs(d)<1e-5) return cv::Point2f(NAN,NAN);
        float x = (l1[1]*l2[2]-l2[1]*l1[2])/d;
        float y = (l1[2]*l2[0]-l2[2]*l1[0])/d;
        return cv::Point2f(x,y);
    };

    auto A0=to_abc(L0.front()), A1=to_abc(L0.back());
    auto B0=to_abc(L1.front()), B1=to_abc(L1.back());
    cv::Point2f q0 = inter(A0,B0), q1 = inter(A1,B0), q2 = inter(A1,B1), q3 = inter(A0,B1);
    std::array<cv::Point2f,4> q = {q0,q1,q2,q3};
    for (auto&p:q) if (!cv::Rect2f(0,0,roi.width,roi.height).contains(p)) return {{0,0,0,0,0,0,0,0}, 0.f};

    float w = cv::norm(q1-q0), h = cv::norm(q3-q0);
    float score_size = std::min(w,h);
    float conf = 0.6f*score_period + 0.3f*score_count + 0.1f*score_size;
    conf = 1.f - 1.f/(1.f + conf*0.02f);

    std::array<float,8> pts = {
            q0.x+roi.x, q0.y+roi.y, q1.x+roi.x, q1.y+roi.y,
            q2.x+roi.x, q2.y+roi.y, q3.x+roi.x, q3.y+roi.y
    };
    return {pts, conf};
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_contextionary_sudoku_GridRefiner_nativeRefine(
        JNIEnv* env, jclass, jbyteArray nv21, jint width, jint height,
        jfloat left, jfloat top, jfloat right, jfloat bottom
){
    jbyte* data = env->GetByteArrayElements(nv21, nullptr);
    cv::Rect roi((int)left, (int)top, (int)(right-left), (int)(bottom-top));
    auto [pts, conf] = refine_in_roi_nv21(reinterpret_cast<uint8_t*>(data), width, height, roi);
    env->ReleaseByteArrayElements(nv21, data, JNI_ABORT);

    jfloat out[9] = { pts[0],pts[1],pts[2],pts[3],pts[4],pts[5],pts[6],pts[7], conf };
    jfloatArray arr = env->NewFloatArray(9);
    env->SetFloatArrayRegion(arr, 0, 9, out);
    return arr;
}