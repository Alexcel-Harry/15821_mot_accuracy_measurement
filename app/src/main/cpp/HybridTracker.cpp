#include "HybridTracker.h"
#include <android/log.h>
#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;

#define LOG_TAG "HybridTracker"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

HybridTracker::HybridTracker(int frame_rate, int track_buffer, int frame_width, int frame_height, int keyframe_interval)
        : byteTracker(frame_rate, track_buffer),
          lightweight_tracker(frame_width, frame_height, 0.5f),
          frame_count(0),
          keyframe_interval(keyframe_interval),
          last_opflow_time_ms(0.0),
          last_tracking_time_ms(0.0) { // Initialize timing variables

    LOGI("HybridTracker created with Interval: %d", this->keyframe_interval);
}

HybridTracker::~HybridTracker() {
    LOGD("HybridTracker destroyed");
}

vector<STrack> HybridTracker::updateWithDetections(const Mat& frame,
                                                   const vector<Object>& objects,
                                                   int frame_width,
                                                   int frame_height) {
    frame_count++;

    if (this->keyframe_interval == 1) {
        // Pure ByteTrack mode (no MOSSE)
        last_opflow_time_ms = 0.0;

        auto track_start = high_resolution_clock::now();
        vector<STrack> byte_tracks = byteTracker.update(objects);
        auto track_end = high_resolution_clock::now();

        last_tracking_time_ms = duration<double, milli>(track_end - track_start).count();
        last_byte_tracks = byte_tracks;

        return byte_tracks;
    }

    // ----------------------------------------------------------------
    // HYBRID MODE (interval > 1)
    // On keyframes: ONLY run ByteTrack + Initialize MOSSE
    // NO updateTrackers() - that's wasteful!
    // ----------------------------------------------------------------

    auto tracking_start = high_resolution_clock::now();
    auto tracking_end = tracking_start;
    auto opflow_start = tracking_start;
    auto opflow_end = tracking_start;
    tracking_start = high_resolution_clock::now();
    // Step 1: Run ByteTrack with YOLO detections
    LOGD("Keyframe %d: Running ByteTrack with %zu detections", frame_count, objects.size());
    vector<STrack> byte_tracks = byteTracker.update(objects);
    last_byte_tracks = byte_tracks;

    LOGD("ByteTrack returned %zu tracks", byte_tracks.size());
    tracking_end = high_resolution_clock::now();
    opflow_start = high_resolution_clock::now();
    // Step 2: Initialize MOSSE trackers from ByteTrack for next frames
    if (!byte_tracks.empty() && !frame.empty()) {
        vector<int> track_ids;
        vector<int> class_ids;
        vector<float> scores;
        vector<Rect2f> bboxes;

        for (const auto& track : byte_tracks) {
            track_ids.push_back(track.track_id);
            class_ids.push_back(track.class_id);
            scores.push_back(track.score);

            float x1 = track.tlbr[0];
            float y1 = track.tlbr[1];
            float x2 = track.tlbr[2];
            float y2 = track.tlbr[3];

            bboxes.emplace_back(x1, y1, x2-x1, y2-y1);
        }

        lightweight_tracker.initializeTrackers(
                frame,
                track_ids.data(),
                class_ids.data(),
                scores.data(),
                bboxes.data(),
                track_ids.size()
        );
        opflow_end = high_resolution_clock::now();
        LOGD("Initialized %zu optical flow trackers", track_ids.size());
    }

//    auto tracking_end = high_resolution_clock::now();

    // Store timing
    last_opflow_time_ms = duration<double, milli>(opflow_end - opflow_start).count();
    last_tracking_time_ms = duration<double, milli>(tracking_end - tracking_start).count();

    LOGD("Keyframe timing: tracking=%.2fms (ByteTrack + MOSSE init)",
         last_tracking_time_ms);

    return byte_tracks;
}

vector<STrack> HybridTracker::updateWithoutDetections(const Mat& frame,
                                                      int frame_width,
                                                      int frame_height) {
    frame_count++;

    if (frame.empty()) {
        last_opflow_time_ms = 0.0;
        last_tracking_time_ms = 0.0;
        return vector<STrack>();
    }

    const int MAX_TRACKS = 100;
    vector<int> track_ids(MAX_TRACKS);
    vector<int> class_ids(MAX_TRACKS);
    vector<float> scores(MAX_TRACKS);
    vector<Rect2f> bboxes(MAX_TRACKS);

    // === OPFLOW TIMING START ===
    auto opflow_start = high_resolution_clock::now();

    int count = lightweight_tracker.updateTrackers(
            frame, track_ids.data(), class_ids.data(),
            scores.data(), bboxes.data(), MAX_TRACKS);

    auto opflow_end = high_resolution_clock::now();
    // === OPFLOW TIMING END ===

    // === TRACKING TIMING START ===
    auto tracking_start = high_resolution_clock::now();

    // Convert MOSSE results to STrack format (preserves track_ids!)
    vector<STrack> mosse_tracks = convertMOSSEResultsToSTracks(
            track_ids.data(), class_ids.data(), scores.data(),
            bboxes.data(), count, frame_width, frame_height);

    // Update ByteTrack's Kalman filters with MOSSE tracking results
    if (!mosse_tracks.empty()) {
        byteTracker.resync_kalman_filters(mosse_tracks);
    }

    auto tracking_end = high_resolution_clock::now();
    // === TRACKING TIMING END ===

    last_opflow_time_ms = duration<double, milli>(opflow_end - opflow_start).count();
    last_tracking_time_ms = duration<double, milli>(tracking_end - tracking_start).count();

    return mosse_tracks;
}

vector<STrack> HybridTracker::convertMOSSEResultsToSTracks(const int* track_ids,
                                                           const int* class_ids,
                                                           const float* scores,
                                                           const Rect2f* bboxes,
                                                           int count,
                                                           int frame_width,
                                                           int frame_height) {
    vector<STrack> tracks;

    for (int i = 0; i < count; i++) {
        // Convert pixel bbox to normalized tlwh
        float x = bboxes[i].x;
        float y = bboxes[i].y;
        float w = bboxes[i].width;
        float h = bboxes[i].height;

        vector<float> tlwh = {x, y, w, h};

        // Create STrack
        STrack track(tlwh, scores[i]);
        track.track_id = track_ids[i];
        track.class_id = class_ids[i];
        track.is_activated = true;

        track.tlbr.clear();
        track.tlbr.push_back(x);
        track.tlbr.push_back(y);
        track.tlbr.push_back(x + w);
        track.tlbr.push_back(y + h);

        tracks.push_back(track);
    }

    return tracks;
}

void HybridTracker::reset() {
    frame_count = 0;
    last_byte_tracks.clear();
    lightweight_tracker.clearTrackers();
    LOGI("HybridTracker reset");
}