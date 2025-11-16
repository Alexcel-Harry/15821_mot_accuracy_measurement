#include <jni.h>
#include <vector>
#include <string>

// 用于从 C++ 打印日志到 Android Logcat
#include <android/log.h>
#define LOG_TAG "ByteTrackJNI"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// 包含 ByteTrack 库的头文件
// BYTETracker.h 会自动包含 dataType.h (定义 Object) 和 STrack.h
#include "BYTETracker.h"

/**
 * 辅助函数：将 Java 的 float[] (YOLO检测结果) 转换为 C++ 的 std::vector<Object>
 * * Java 格式 (6 个字段): [cx_norm, cy_norm, w_norm, h_norm, classId, conf]
 * * C++ Object 格式 (在 dataType.h 中定义):
 * struct Object
 * {
 * cv::Rect_<float> rect;
 * int label; // 我们将把 classId 存在这里
 * float prob;
 * };
 */
std::vector<Object> javaToCppObjects(JNIEnv *env, jfloatArray javaDetections, int img_w, int img_h) {
    std::vector<Object> cppDetections;

    jsize numFloats = env->GetArrayLength(javaDetections);
    if (numFloats == 0) {
        return cppDetections;
    }

    jfloat* detData = env->GetFloatArrayElements(javaDetections, 0);

    // 我们的Java代码打包成6个字段一组
    int fieldsPerDetection = 6;
    int numDetections = numFloats / fieldsPerDetection;

    for (int i = 0; i < numDetections; ++i) {
        float cx_norm = detData[i * fieldsPerDetection + 0];
        float cy_norm = detData[i * fieldsPerDetection + 1];
        float w_norm  = detData[i * fieldsPerDetection + 2];
        float h_norm  = detData[i * fieldsPerDetection + 3];
        int classId   = (int)detData[i * fieldsPerDetection + 4];
        float score   = detData[i * fieldsPerDetection + 5];

        // 从归一化中心坐标转换为像素 [x, y, w, h]
        float w_pixel = w_norm * img_w;
        float h_pixel = h_norm * img_h;
        float x1 = (cx_norm * img_w) - (w_pixel / 2.0f);
        float y1 = (cy_norm * img_h) - (h_pixel / 2.0f);

        Object obj;
        obj.rect.x = x1;
        obj.rect.y = y1;
        obj.rect.width = w_pixel;
        obj.rect.height = h_pixel;
        obj.prob = score;
        obj.label = classId; // 将 classId 存储在 Object.label 中

        cppDetections.push_back(obj);
    }

    // 释放 Java 数组内存
    env->ReleaseFloatArrayElements(javaDetections, detData, JNI_ABORT);
    // LOGD("Converted %d Java detections to C++ Object vector", numDetections);
    return cppDetections;
}

/**
 * 辅助函数：将 C++ 的 std::vector<STrack> (跟踪结果) 转换为 Java 的 float[]
 * * C++ STrack 包含: track.tlbr, track.score, track.track_id
 * * C++ STrack (你提供的版本) *不包含* class_id
 * * Java 格式 (7 个字段): [cx_norm, cy_norm, w_norm, h_norm, classId, conf, track_id]
 */
jfloatArray cppToJavaTracks(JNIEnv *env, const std::vector<STrack>& cppTracks, int img_w, int img_h) {
    if (cppTracks.empty()) {
        return env->NewFloatArray(0); // 返回一个空数组
    }

    // 我们的Java代码期望7个字段
    int fieldsPerTrack = 7;
    int numTracks = cppTracks.size();
    int numFloats = numTracks * fieldsPerTrack;

    jfloat* trackData = new jfloat[numFloats];

    for (int i = 0; i < numTracks; ++i) {
        const auto& track = cppTracks[i];

        // STrack::tlbr 是一个 vector<float> {x1, y1, x2, y2}
        // (基于 STrack.cpp 中 static_tlbr() 的实现)
        const auto& tlbr = track.tlbr;

        float x1 = tlbr[0];
        float y1 = tlbr[1];
        float x2 = tlbr[2];
        float y2 = tlbr[3];

        // 转换为归一化 [cx, cy, w, h]
        float w_pixel = x2 - x1;
        float h_pixel = y2 - y1;
        float cx_norm = (x1 + w_pixel / 2.0f) / img_w;
        float cy_norm = (y1 + h_pixel / 2.0f) / img_h;
        float w_norm  = w_pixel / img_w;
        float h_norm  = h_pixel / img_h;

        // 填充缓冲区
        trackData[i * fieldsPerTrack + 0] = cx_norm;
        trackData[i * fieldsPerTrack + 1] = cy_norm;
        trackData[i * fieldsPerTrack + 2] = w_norm;
        trackData[i * fieldsPerTrack + 3] = h_norm;

        // === 关键修复 ===
        // 你提供的 STrack.cpp/h 文件没有 class_id 成员。
        // C++ 跟踪器在 update 过程中丢弃了它。
        // 我们在此处硬编码 -1.0f 来通知 Java 端。
        // trackData[i * fieldsPerTrack + 4] = -1.0f; // class_id
        trackData[i * fieldsPerTrack + 4] = (float)track.class_id;

        trackData[i * fieldsPerTrack + 5] = track.score;
        trackData[i * fieldsPerTrack + 6] = (float)track.track_id;
    }

    jfloatArray javaTracks = env->NewFloatArray(numFloats);
    env->SetFloatArrayRegion(javaTracks, 0, numFloats, trackData);
    delete[] trackData;

    // LOGD("Converted %d C++ STracks to Java float[]", numTracks);
    return javaTracks;
}


// JNI 函数必须包裹在 extern "C" 中
extern "C" {

/**
 * JNI 函数：初始化 BYTETracker
 * 对应 Java: native long nativeInitTracker(int frameRate)
 */
JNIEXPORT jlong JNICALL
Java_edu_cmu_cs_face_MainActivity_nativeInitTracker(
        JNIEnv *env,
        jobject thiz,
        // 只保留 jint frame_rate
        jint frame_rate) {

    int track_buffer = 30; // 默认值
    BYTETracker* tracker = new BYTETracker(frame_rate, track_buffer);

    // LOGD("Native tracker initialized. FrameRate: %d. Using C++ hard-coded thresholds.", (int)frame_rate);

    return reinterpret_cast<jlong>(tracker);
}

/**
 * JNI 函数：释放 BYTETracker
 * 对应 Java: native void nativeReleaseTracker(long trackerPtr)
 */
JNIEXPORT void JNICALL
Java_edu_cmu_cs_face_MainActivity_nativeReleaseTracker(
        JNIEnv *env,
        jobject thiz,
        jlong tracker_ptr) {

    BYTETracker* tracker = reinterpret_cast<BYTETracker*>(tracker_ptr);
    delete tracker;
    // LOGD("Native tracker released.");
}

/**
 * JNI 函数：运行跟踪器更新
 * 对应 Java: native float[] nativeUpdate(long trackerPtr, float[] detections, int w, int h)
 */
JNIEXPORT jfloatArray JNICALL
Java_edu_cmu_cs_face_MainActivity_nativeUpdate(
        JNIEnv *env,
        jobject thiz,
        jlong tracker_ptr,
        jfloatArray java_detections,
        jint img_w,
        jint img_h) {

    BYTETracker* tracker = reinterpret_cast<BYTETracker*>(tracker_ptr);
    if (tracker == nullptr) {
        LOGE("Tracker pointer is null!");
        return env->NewFloatArray(0); // 返回空数组
    }

    // === 关键修复 1: ===
    // 将 Java float[] 转换为 C++ std::vector<Object>
    std::vector<Object> cppDetections = javaToCppObjects(env, java_detections, img_w, img_h);

    // === 关键修复 2: ===
    // 调用 C++ 库的 update(const vector<Object>& objects)
    std::vector<STrack> cppTracks = tracker->update(cppDetections);

    // 3. 将 C++ 的跟踪结果 (std::vector<STrack>) 转换回 Java 的 (float[])
    //    (cppToJavaTracks 内部已修复了 class_id 的问题)
    return cppToJavaTracks(env, cppTracks, img_w, img_h);
}

} // extern "C"