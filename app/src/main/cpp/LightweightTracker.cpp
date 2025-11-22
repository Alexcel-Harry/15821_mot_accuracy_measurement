#include "LightweightTracker.h"
#include <android/log.h>

using namespace cv;
using namespace std;

#define LOG_TAG "LightweightTracker"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

const float MAX_SCALE_CHANGE_DOWN = 0.97f;
const float MAX_SCALE_CHANGE_UP   = 1.03f;

LightweightTracker::LightweightTracker(int original_width, int original_height, float scale)
        : original_size(original_width, original_height), klt_scale(scale) {
    klt_size = cv::Size(original_width * klt_scale, original_height * klt_scale);
    LOGD("LightweightTracker created (KLT)");
    LOGD("Original size: %dx%d, KLT size: %dx%d (Scale: %.2f)",
         original_size.width, original_size.height,
         klt_size.width, klt_size.height, klt_scale);
}

LightweightTracker::~LightweightTracker() {
    clearTrackers();
    LOGD("LightweightTracker destroyed");
}

vector<Point2f> LightweightTracker::extractFeaturePoints(const Mat& frame, const Rect2f& bbox) {
    vector<Point2f> points;
    
    // Ensure bbox is within frame
    Rect2f safe_bbox = bbox & Rect2f(0, 0, frame.cols, frame.rows);
    if (safe_bbox.width < 5 || safe_bbox.height < 5) {
        return points;
    }
    
    // Extract ROI
    Rect roi(safe_bbox.x, safe_bbox.y, safe_bbox.width, safe_bbox.height);
    Mat roi_frame = frame(roi);
    
    // Detect good features to track
    vector<Point2f> corners;
    goodFeaturesToTrack(roi_frame, corners, 20, 0.01, 10);
    
    // Convert to absolute coordinates
    for (const auto& corner : corners) {
        points.push_back(Point2f(corner.x + safe_bbox.x, corner.y + safe_bbox.y));
    }
    
    LOGD("Extracted %zu feature points from bbox", points.size());
    return points;
}

float LightweightTracker::calculateMedianScale(const vector<Point2f>& old_points,
                                               const vector<Point2f>& new_points) {
    if (old_points.empty() || old_points.size() != new_points.size()) {
        return 1.0f; // 无变化
    }

    vector<float> scale_ratios;

    // 1. 找到新旧点集的质心（平均点）
    Point2f old_centroid(0, 0), new_centroid(0, 0);
    for (const auto& p : old_points) { old_centroid += p; }
    for (const auto& p : new_points) { new_centroid += p; }
    old_centroid /= (int)old_points.size();
    new_centroid /= (int)new_points.size();

    // 2. 计算每个点到其质心的距离，并计算缩放比例
    for (size_t i = 0; i < old_points.size(); i++) {
        float d_old = norm(old_points[i] - old_centroid);
        float d_new = norm(new_points[i] - new_centroid);

        if (d_old > 1e-3) { // 避免除以零
            scale_ratios.push_back(d_new / d_old);
        }
    }

    if (scale_ratios.empty()) {
        return 1.0f;
    }

    // 3. 找到缩放比例的 "中位数"
    std::sort(scale_ratios.begin(), scale_ratios.end());
    return scale_ratios[scale_ratios.size() / 2];
}

void LightweightTracker::initializeTrackers(const Mat& frame,
                                           const int* track_ids,
                                           const int* class_ids,
                                           const float* scores,
                                           const Rect2f* bboxes,
                                           int count) {
    // Clear existing trackers
    clearTrackers();
    
    if (frame.empty()) {
        LOGW("Empty frame provided to initializeTrackers");
        return;
    }

    Mat small_gray;
    cv::resize(frame, small_gray, klt_size, 0, 0, cv::INTER_LINEAR);

    // Convert to grayscale
    prev_gray = small_gray;
    
    // Initialize trackers for each detection
    for (int i = 0; i < count; i++) {
        Rect2f original_bbox = bboxes[i];

        // 缩小 BBox
        Rect2f klt_bbox(original_bbox.x * klt_scale,
                        original_bbox.y * klt_scale,
                        original_bbox.width * klt_scale,
                        original_bbox.height * klt_scale);
        // === ADD THIS: Clamp to valid bounds ===
        klt_bbox.x = std::max(0.0f, klt_bbox.x);
        klt_bbox.y = std::max(0.0f, klt_bbox.y);

        if (klt_bbox.x + klt_bbox.width > klt_size.width) {
            klt_bbox.width = klt_size.width - klt_bbox.x;
        }
        if (klt_bbox.y + klt_bbox.height > klt_size.height) {
            klt_bbox.height = klt_size.height - klt_bbox.y;
        }
        // Validate bounding box
        if (!isValidBoundingBox(klt_bbox, klt_size)) {
            LOGW("Invalid *scaled* bounding box for track_id=%d: [%.1f, %.1f, %.1f, %.1f]",
                 track_ids[i], klt_bbox.x, klt_bbox.y, klt_bbox.width, klt_bbox.height);
            continue;
        }
        
        // Create tracked object
        TrackedObject obj(track_ids[i], class_ids[i], scores[i], original_bbox);
        
        // Extract feature points for optical flow
        obj.points = extractFeaturePoints(prev_gray, klt_bbox);

        if (obj.points.size() >= 4) {
            tracked_objects.push_back(obj);
            LOGD("Initialized optical flow tracker for track_id=%d, class=%d, points=%zu",
                 track_ids[i], class_ids[i], obj.points.size());
        } else {
            LOGW("Not enough feature points for track_id=%d", track_ids[i]);
        }
    }
    
    LOGD("Initialized %zu optical flow trackers from %d detections", 
         tracked_objects.size(), count);
}

int LightweightTracker::updateTrackers(const Mat& frame,
                                      int* out_track_ids,
                                      int* out_class_ids,
                                      float* out_scores,
                                      Rect2f* out_bboxes,
                                      int max_output_size) {
    if (frame.empty()) {
        LOGW("Empty frame provided to updateTrackers");
        return 0;
    }

    Mat curr_gray;
    cv::resize(frame, curr_gray, klt_size, 0, 0, cv::INTER_LINEAR);

    if (prev_gray.empty()) {
        LOGW("No previous frame for optical flow");
        prev_gray = curr_gray.clone();
        return 0;
    }
    
    int output_count = 0;
    
    // Update each tracker using optical flow
    for (size_t i = 0; i < tracked_objects.size(); i++) {
        TrackedObject& obj = tracked_objects[i];
        
        if (!obj.valid || obj.points.empty()) {
            continue;
        }
        
        // Track points using Lucas-Kanade optical flow
        vector<Point2f> new_points;
        vector<uchar> status;
        vector<float> err;

        try {
            calcOpticalFlowPyrLK(prev_gray, curr_gray, obj.points, new_points,
                                 status, err, Size(21, 21), 3);
        } catch (const cv::Exception& e) {
            LOGW("Optical flow exception for track_id=%d: %s", obj.track_id, e.what());
            obj.valid = false;
            continue;
        }

        vector<Point2f> good_old_points, good_new_points;
        for (size_t j = 0; j < status.size(); j++) {
            if (status[j]) {
                good_old_points.push_back(obj.points[j]);
                good_new_points.push_back(new_points[j]);
            }
        }
        
        // Update bounding box if we have enough good points
        if (good_new_points.size() >= 4) {
            // 1. 计算稳健的 "平均" 平移 (dx, dy)
            float dx = 0, dy = 0;
            for (size_t j = 0; j < good_new_points.size(); j++) {
                dx += (good_new_points[j].x - good_old_points[j].x);
                dy += (good_new_points[j].y - good_old_points[j].y);
            }
            dx /= good_new_points.size();
            dy /= good_new_points.size();

            // 2. 计算稳健的 "中位数" 缩放
            float scale_change = calculateMedianScale(good_old_points, good_new_points);

            // 3. [防抖] 约束缩放，防止微小抖动
            scale_change = std::max(MAX_SCALE_CHANGE_DOWN, std::min(MAX_SCALE_CHANGE_UP, scale_change));

            // 4. 获取当前的 klt_bbox
            Rect2f klt_bbox(obj.bbox.x * klt_scale,
                            obj.bbox.y * klt_scale,
                            obj.bbox.width * klt_scale,
                            obj.bbox.height * klt_scale);

            // 5. 应用平移和缩放
            // (我们从框的中心开始缩放，以保持稳定)
            float old_w = klt_bbox.width;
            float old_h = klt_bbox.height;
            float new_w = old_w * scale_change;
            float new_h = old_h * scale_change;

            klt_bbox.x += dx - (new_w - old_w) / 2.0f; // 应用平移，并根据缩放调整x
            klt_bbox.y += dy - (new_h - old_h) / 2.0f; // 应用平移，并根据缩放调整y
            klt_bbox.width = new_w;
            klt_bbox.height = new_h;
            // ----------------------------------------------------

            bool success = true; // 我们的逻辑总是成功的

            if (success && isValidBoundingBox(klt_bbox, klt_size)) {
                obj.bbox.x = klt_bbox.x / klt_scale;
                obj.bbox.y = klt_bbox.y / klt_scale;
                obj.bbox.width = klt_bbox.width / klt_scale;
                obj.bbox.height = klt_bbox.height / klt_scale;
                // === ADD THIS: Clamp to original frame size ===
                obj.bbox.x = std::max(0.0f, obj.bbox.x);
                obj.bbox.y = std::max(0.0f, obj.bbox.y);

                if (obj.bbox.x + obj.bbox.width > original_size.width) {
                    obj.bbox.width = original_size.width - obj.bbox.x;
                }
                if (obj.bbox.y + obj.bbox.height > original_size.height) {
                    obj.bbox.height = original_size.height - obj.bbox.y;
                }
                obj.points = good_new_points;
                obj.frames_tracked++;

                if (obj.points.size() < 10) {
                    LOGD("Refreshing feature points for track_id=%d (only %zu remaining)",
                         obj.track_id, obj.points.size());
                    vector<Point2f> new_features = extractFeaturePoints(curr_gray, klt_bbox);
                    obj.points.insert(obj.points.end(), new_features.begin(), new_features.end());
                }

                if (output_count < max_output_size) {
                    out_track_ids[output_count] = obj.track_id;
                    out_class_ids[output_count] = obj.class_id;
                    out_scores[output_count] = obj.score;
                    out_bboxes[output_count] = obj.bbox;
                    output_count++;
                }
            } else {
                obj.valid = false;
                LOGW("Optical flow tracker failed for track_id=%d after %d frames",
                     obj.track_id, obj.frames_tracked);
            }
        } else {
            obj.valid = false;
            LOGW("Not enough good points (%zu) for track_id=%d",
                 good_new_points.size(), obj.track_id);
        }
    }

    prev_gray = curr_gray.clone();

    LOGD("Updated %d/%zu optical flow trackers successfully",
         output_count, tracked_objects.size());
    return output_count;
}

void LightweightTracker::clearTrackers() {
    tracked_objects.clear();
    prev_gray.release();
    LOGD("Cleared all optical flow trackers");
}

bool LightweightTracker::isValidBoundingBox(const Rect2f& bbox, const Size& frame_size) const {
    // Check if bbox is within frame bounds
    if (bbox.x < 0 || bbox.y < 0) {
        return false;
    }
    
    if (bbox.x + bbox.width > frame_size.width || 
        bbox.y + bbox.height > frame_size.height) {
        return false;
    }
    
//    // Check minimum size
//    if (bbox.width < 5.0f || bbox.height < 5.0f) {
//        return false;
//    }
//
//    // Check maximum size
//    if (bbox.width > frame_size.width * 0.95f ||
//        bbox.height > frame_size.height * 0.95f) {
//        return false;
//    }
    
    return true;
}