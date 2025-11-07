#include <jni.h>
#include <vector>
#include <string>
#include <android/log.h>

#define LOG_TAG "HybridTrackerJNI"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#include "HybridTracker.h"

/**
 * Convert Java float[] detections to C++ vector<Object>
 * Java format: [cx_norm, cy_norm, w_norm, h_norm, classId, conf] per detection
 */
std::vector<Object> javaToCppObjects(JNIEnv *env, jfloatArray javaDetections, int img_w, int img_h) {
    std::vector<Object> cppDetections;

    jsize numFloats = env->GetArrayLength(javaDetections);
    if (numFloats == 0) {
        return cppDetections;
    }

    jfloat* detData = env->GetFloatArrayElements(javaDetections, 0);

    int fieldsPerDetection = 6;
    int numDetections = numFloats / fieldsPerDetection;

    for (int i = 0; i < numDetections; ++i) {
        float cx_norm = detData[i * fieldsPerDetection + 0];
        float cy_norm = detData[i * fieldsPerDetection + 1];
        float w_norm  = detData[i * fieldsPerDetection + 2];
        float h_norm  = detData[i * fieldsPerDetection + 3];
        int classId   = (int)detData[i * fieldsPerDetection + 4];
        float score   = detData[i * fieldsPerDetection + 5];

        // Convert normalized center coords to pixel [x, y, w, h]
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
        obj.label = classId;

        cppDetections.push_back(obj);
    }

    env->ReleaseFloatArrayElements(javaDetections, detData, JNI_ABORT);
    LOGD("Converted %d Java detections to C++ Object vector", numDetections);
    return cppDetections;
}

/**
 * Convert C++ vector<STrack> to Java float[]
 * Java format: [cx_norm, cy_norm, w_norm, h_norm, classId, conf, track_id] per track
 */
jfloatArray cppToJavaTracks(JNIEnv *env, const std::vector<STrack>& cppTracks, int img_w, int img_h) {
    if (cppTracks.empty()) {
        return env->NewFloatArray(0);
    }

    int fieldsPerTrack = 7;
    int numTracks = cppTracks.size();
    int numFloats = numTracks * fieldsPerTrack;

    jfloat* trackData = new jfloat[numFloats];

    for (int i = 0; i < numTracks; ++i) {
        const auto& track = cppTracks[i];
        const auto& tlbr = track.tlbr;

        float x1 = tlbr[0];
        float y1 = tlbr[1];
        float x2 = tlbr[2];
        float y2 = tlbr[3];

        // Convert to normalized [cx, cy, w, h]
        float w_pixel = x2 - x1;
        float h_pixel = y2 - y1;
        float cx_norm = (x1 + w_pixel / 2.0f) / img_w;
        float cy_norm = (y1 + h_pixel / 2.0f) / img_h;
        float w_norm  = w_pixel / img_w;
        float h_norm  = h_pixel / img_h;

        trackData[i * fieldsPerTrack + 0] = cx_norm;
        trackData[i * fieldsPerTrack + 1] = cy_norm;
        trackData[i * fieldsPerTrack + 2] = w_norm;
        trackData[i * fieldsPerTrack + 3] = h_norm;
        trackData[i * fieldsPerTrack + 4] = (float)track.class_id;
        trackData[i * fieldsPerTrack + 5] = track.score;
        trackData[i * fieldsPerTrack + 6] = (float)track.track_id;
    }

    jfloatArray javaTracks = env->NewFloatArray(numFloats);
    env->SetFloatArrayRegion(javaTracks, 0, numFloats, trackData);
    delete[] trackData;

    LOGD("Converted %d C++ STracks to Java float[]", numTracks);
    return javaTracks;
}

/**
 * Convert Java byte[] (grayscale or color image) to OpenCV Mat
 * This is needed for MOSSE tracking
 */
cv::Mat javaByteArrayToMat(JNIEnv *env, jbyteArray javaImageData, int width, int height, bool isGrayscale) {
    jbyte* imageData = env->GetByteArrayElements(javaImageData, 0);
    
    cv::Mat mat;
    if (isGrayscale) {
        // Single channel grayscale
        mat = cv::Mat(height, width, CV_8UC1);
        memcpy(mat.data, imageData, width * height);
    } else {
        // Assume NV21 format (YUV) - convert to grayscale for MOSSE
        // We only need Y plane for grayscale
        mat = cv::Mat(height, width, CV_8UC1);
        memcpy(mat.data, imageData, width * height);
    }
    
    env->ReleaseByteArrayElements(javaImageData, imageData, JNI_ABORT);
    return mat;
}

extern "C" {

/**
 * Initialize HybridTracker
 * Java: native long nativeInitHybridTracker(int frameRate, int trackBuffer, int keyframeInterval)
 */
JNIEXPORT jlong JNICALL
Java_edu_cmu_cs_face_MainActivity_nativeInitHybridTracker(
        JNIEnv *env,
        jobject thiz,
        jint frame_rate,
        jint track_buffer,
        jint keyframe_interval) {

    HybridTracker* tracker = new HybridTracker(frame_rate, track_buffer, keyframe_interval);
    LOGD("HybridTracker initialized: frame_rate=%d, track_buffer=%d, keyframe_interval=%d",
         (int)frame_rate, (int)track_buffer, (int)keyframe_interval);
    return reinterpret_cast<jlong>(tracker);
}

/**
 * Release HybridTracker
 * Java: native void nativeReleaseHybridTracker(long trackerPtr)
 */
JNIEXPORT void JNICALL
Java_edu_cmu_cs_face_MainActivity_nativeReleaseHybridTracker(
        JNIEnv *env,
        jobject thiz,
        jlong tracker_ptr) {

    HybridTracker* tracker = reinterpret_cast<HybridTracker*>(tracker_ptr);
    delete tracker;
    LOGD("HybridTracker released");
}

/**
 * Check if current frame should be a keyframe
 * Java: native boolean nativeIsKeyframe(long trackerPtr)
 */
JNIEXPORT jboolean JNICALL
Java_edu_cmu_cs_face_MainActivity_nativeIsKeyframe(
        JNIEnv *env,
        jobject thiz,
        jlong tracker_ptr) {

    HybridTracker* tracker = reinterpret_cast<HybridTracker*>(tracker_ptr);
    if (tracker == nullptr) {
        LOGE("Tracker pointer is null in nativeIsKeyframe!");
        return JNI_FALSE;
    }
    
    return tracker->isKeyframe() ? JNI_TRUE : JNI_FALSE;
}

/**
 * Update tracker with YOLO detections (keyframe)
 * Java: native float[] nativeUpdateWithDetections(long trackerPtr, float[] detections, 
 *                                                   byte[] imageData, int w, int h)
 */
JNIEXPORT jfloatArray JNICALL
Java_edu_cmu_cs_face_MainActivity_nativeUpdateWithDetections(
        JNIEnv *env,
        jobject thiz,
        jlong tracker_ptr,
        jfloatArray java_detections,
        jbyteArray java_image_data,
        jint img_w,
        jint img_h) {

    HybridTracker* tracker = reinterpret_cast<HybridTracker*>(tracker_ptr);
    if (tracker == nullptr) {
        LOGE("Tracker pointer is null!");
        return env->NewFloatArray(0);
    }

    // Convert Java detections to C++ Objects
    std::vector<Object> cppDetections = javaToCppObjects(env, java_detections, img_w, img_h);

    // Convert Java image data to OpenCV Mat (for MOSSE initialization)
    cv::Mat frame = javaByteArrayToMat(env, java_image_data, img_w, img_h, true);

    // Update hybrid tracker with detections
    std::vector<STrack> cppTracks = tracker->updateWithDetections(frame, cppDetections, img_w, img_h);

    // Convert back to Java format
    return cppToJavaTracks(env, cppTracks, img_w, img_h);
}

/**
 * Update tracker without detections (intermediate frame, MOSSE only)
 * Java: native float[] nativeUpdateWithoutDetections(long trackerPtr, byte[] imageData, int w, int h)
 */
JNIEXPORT jfloatArray JNICALL
Java_edu_cmu_cs_face_MainActivity_nativeUpdateWithoutDetections(
        JNIEnv *env,
        jobject thiz,
        jlong tracker_ptr,
        jbyteArray java_image_data,
        jint img_w,
        jint img_h) {

    HybridTracker* tracker = reinterpret_cast<HybridTracker*>(tracker_ptr);
    if (tracker == nullptr) {
        LOGE("Tracker pointer is null!");
        return env->NewFloatArray(0);
    }

    // Convert Java image data to OpenCV Mat
    cv::Mat frame = javaByteArrayToMat(env, java_image_data, img_w, img_h, true);

    // Update hybrid tracker without detections (MOSSE tracking only)
    std::vector<STrack> cppTracks = tracker->updateWithoutDetections(frame, img_w, img_h);

    // Convert back to Java format
    return cppToJavaTracks(env, cppTracks, img_w, img_h);
}

/**
 * Reset the hybrid tracker
 * Java: native void nativeResetHybridTracker(long trackerPtr)
 */
JNIEXPORT void JNICALL
Java_edu_cmu_cs_face_MainActivity_nativeResetHybridTracker(
        JNIEnv *env,
        jobject thiz,
        jlong tracker_ptr) {

    HybridTracker* tracker = reinterpret_cast<HybridTracker*>(tracker_ptr);
    if (tracker == nullptr) {
        LOGE("Tracker pointer is null!");
        return;
    }
    
    tracker->reset();
    LOGD("HybridTracker reset");
}

} // extern "C"