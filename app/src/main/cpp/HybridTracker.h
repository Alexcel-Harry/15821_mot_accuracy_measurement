#ifndef HYBRID_TRACKER_H
#define HYBRID_TRACKER_H

#include "BYTETracker.h"
#include "LightweightTracker.h"
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * HybridTracker combines ByteTrack (heavy but accurate) with MOSSE (lightweight).
 *
 * Strategy:
 * - Keyframes (every K frames): Run YOLO + ByteTrack for full detection & tracking
 * - Intermediate frames: Use MOSSE optical tracker to update bounding boxes
 *
 * This solves the problem where:
 * - YOLO is too slow (~70ms) to run every frame
 * - Skipping frames causes ByteTrack's Kalman filter to fail on non-linear motion
 * - MOSSE provides fast visual tracking between YOLO updates
 */
class HybridTracker {
public:
    /**
     * Constructor
     * @param frame_rate Camera frame rate (e.g., 30)
     * @param track_buffer ByteTrack parameter for how long to keep lost tracks
     * @param keyframe_interval How many frames between YOLO runs (e.g., 3 = YOLO every 3 frames)
     */
    explicit HybridTracker(int frame_rate = 30, int track_buffer = 30, int frame_width = 1280, int frame_height = 720, int keyframe_interval = 1);
    ~HybridTracker();

    /**
     * Process a frame with full YOLO detections + ByteTrack.
     * This should be called on keyframes.
     *
     * @param frame The current frame (for MOSSE initialization)
     * @param objects YOLO detections
     * @param frame_width Original frame width
     * @param frame_height Original frame height
     * @return Tracked objects with track IDs
     */
    vector<STrack> updateWithDetections(const Mat& frame,
                                        const vector<Object>& objects,
                                        int frame_width,
                                        int frame_height);

    /**
     * Process a frame without YOLO (lightweight tracking only).
     * This should be called on intermediate frames.
     *
     * @param frame The current frame
     * @param frame_width Original frame width
     * @param frame_height Original frame height
     * @return Tracked objects (from MOSSE updates)
     */
    vector<STrack> updateWithoutDetections(const Mat& frame,
                                           int frame_width,
                                           int frame_height);


    /**
     * Reset the tracker state.
     */
    void reset();

    /**
     * Get current frame count.
     */
    [[nodiscard]] int getFrameCount() const { return frame_count; }

    /**
     * Get last opflow time in milliseconds.
     */
    [[nodiscard]] double getLastOpflowTimeMs() const { return last_opflow_time_ms; }

    /**
     * Get last tracking time in milliseconds.
     */
    [[nodiscard]] double getLastTrackingTimeMs() const { return last_tracking_time_ms; }


private:
    BYTETracker byteTracker;
    LightweightTracker lightweight_tracker;

    int frame_count;
    int keyframe_interval;

    // Store last ByteTrack results for reference
    vector<STrack> last_byte_tracks;

    // Timing measurements (in milliseconds)
    double last_opflow_time_ms;
    double last_tracking_time_ms;

    /**
     * Convert MOSSE tracking results back to STrack format.
     */
    vector<STrack> convertMOSSEResultsToSTracks(const int* track_ids,
                                                const int* class_ids,
                                                const float* scores,
                                                const Rect2f* bboxes,
                                                int count,
                                                int frame_width,
                                                int frame_height);
};

#endif // HYBRID_TRACKER_H