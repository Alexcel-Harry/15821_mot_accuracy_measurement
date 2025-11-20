#include "HybridTracker.h"
#include <android/log.h>

using namespace cv;
using namespace std;

#define LOG_TAG "HybridTracker"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

HybridTracker::HybridTracker(int frame_rate, int track_buffer, int frame_width, int frame_height, int keyframe_interval)
        : byteTracker(frame_rate, track_buffer),
          lightweight_tracker(frame_width, frame_height, 0.5f),
          frame_count(0),
          keyframe_interval(keyframe_interval) { // Initialize the member variable

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
        // ----------------------------------------------------------------
        // OPTION A: PURE BYTETRACK MODE
        // Bypass MOSSE entirely. Save ~15ms of overhead.
        // ----------------------------------------------------------------
        LOGD("Interval 1 detected: Running Pure ByteTrack Logic");

        // 1. Direct ByteTrack Update
        vector<STrack> byte_tracks = byteTracker.update(objects);
        last_byte_tracks = byte_tracks;

        return byte_tracks;
    }
    const int MAX_TRACKS = 100;

    // [修改] 使用 std::vector 替代 C 风格数组，更安全
    vector<int> klt_track_ids(MAX_TRACKS);
    vector<int> klt_class_ids(MAX_TRACKS);
    vector<float> klt_scores(MAX_TRACKS);
    vector<Rect2f> klt_bboxes(MAX_TRACKS);

    int klt_count = lightweight_tracker.updateTrackers(
            frame,
            klt_track_ids.data(),   // 使用 .data() 传递指针
            klt_class_ids.data(),
            klt_scores.data(),
            klt_bboxes.data(),
            MAX_TRACKS
    );

    vector<STrack> klt_tracks = convertMOSSEResultsToSTracks(
            klt_track_ids.data(), klt_class_ids.data(), klt_scores.data(),
            klt_bboxes.data(), klt_count,
            frame_width, frame_height
    );

    if (!klt_tracks.empty()) {
        LOGD("Keyframe %d: Resyncing %zu ByteTrack KFs with KLT results", frame_count, klt_tracks.size());
        byteTracker.resync_kalman_filters(klt_tracks);
    }

    LOGD("Keyframe %d: Running ByteTrack with %zu detections", frame_count, objects.size());
    vector<STrack> byte_tracks = byteTracker.update(objects);
    last_byte_tracks = byte_tracks;

    LOGD("ByteTrack returned %zu tracks", byte_tracks.size());

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

            float width = x2 - x1;
            float height = y2 - y1;

            bboxes.emplace_back(x1, y1, width, height);
        }

        lightweight_tracker.initializeTrackers(
                frame,
                track_ids.data(),
                class_ids.data(),
                scores.data(),
                bboxes.data(),
                track_ids.size()
        );
    }

    return byte_tracks;
}

vector<STrack> HybridTracker::updateWithoutDetections(const Mat& frame,
                                                      int frame_width,
                                                      int frame_height) {
    frame_count++;

    LOGD("Intermediate frame %d: Running MOSSE tracking", frame_count);

    if (frame.empty()) {
        LOGD("Empty frame, returning empty tracks");
        return vector<STrack>();
    }

    // Update MOSSE trackers
    const int MAX_TRACKS = 100;

    vector<int> track_ids(MAX_TRACKS);
    vector<int> class_ids(MAX_TRACKS);
    vector<float> scores(MAX_TRACKS);
    vector<Rect2f> bboxes(MAX_TRACKS);

    int count = lightweight_tracker.updateTrackers(
            frame,
            track_ids.data(),
            class_ids.data(),
            scores.data(),
            bboxes.data(),
            MAX_TRACKS
    );

    LOGD("MOSSE tracking returned %d tracks", count);

    // Convert MOSSE results back to STrack format
    return convertMOSSEResultsToSTracks(
            track_ids.data(), class_ids.data(), scores.data(),
            bboxes.data(), count,
            frame_width, frame_height
    );
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