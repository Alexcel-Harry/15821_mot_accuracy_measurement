#ifndef LIGHTWEIGHT_TRACKER_H
#define LIGHTWEIGHT_TRACKER_H

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>
#include <memory>

/**
 * Lightweight tracker for intermediate frames between YOLO detections.
 * Uses Optical Flow (KLT) for fast tracking since MOSSE may not be available.
 * Each tracked object gets its own set of feature points.
 */
class LightweightTracker {
public:
    struct TrackedObject {
        int track_id;           // The track ID from ByteTrack
        int class_id;           // The class ID
        float score;            // Confidence score
        cv::Rect2f bbox;        // Bounding box in pixel coordinates
        std::vector<cv::Point2f> points; // Feature points for optical flow tracking
        bool valid;             // Whether this tracker is still valid
        int frames_tracked;     // Number of frames tracked since last YOLO update

        TrackedObject(int tid, int cid, float s, const cv::Rect2f& box)
                : track_id(tid), class_id(cid), score(s), bbox(box),
                  valid(true), frames_tracked(0) {
        }
    };
    
    LightweightTracker(int original_width = 1280, int original_height = 720, float scale = 0.5f);
    ~LightweightTracker();
    
    /**
     * Initialize trackers with detections from YOLO + ByteTrack.
     * This should be called on keyframes.
     * 
     * @param frame The current frame (grayscale or color)
     * @param track_ids Array of track IDs from ByteTrack
     * @param class_ids Array of class IDs
     * @param scores Array of confidence scores
     * @param bboxes Array of bounding boxes (x, y, width, height in pixels)
     * @param count Number of objects
     */
    void initializeTrackers(const cv::Mat& frame,
                            const int* track_ids,
                            const int* class_ids,
                            const float* scores,
                            const cv::Rect2f* bboxes,
                            int count);
    
    /**
     * Update all trackers with a new frame (intermediate frame).
     * This is called on non-keyframes.
     * 
     * @param frame The current frame
     * @param out_track_ids Output array for track IDs
     * @param out_class_ids Output array for class IDs  
     * @param out_scores Output array for scores
     * @param out_bboxes Output array for updated bounding boxes
     * @param max_output_size Maximum size of output arrays
     * @return Number of successfully tracked objects
     */
    int updateTrackers(const cv::Mat& frame,
                       int* out_track_ids,
                       int* out_class_ids,
                       float* out_scores,
                       cv::Rect2f* out_bboxes,
                       int max_output_size);
    
    /**
     * Clear all trackers. Called when resetting.
     */
    void clearTrackers();
    
    /**
     * Get number of active trackers.
     */
    int getTrackerCount() const { return tracked_objects.size(); }
    
private:
    std::vector<TrackedObject> tracked_objects;
    cv::Mat prev_gray;

    cv::Size original_size;
    cv::Size klt_size;
    float klt_scale;
    
    /**
     * Extract feature points from a bounding box region.
     */
    std::vector<cv::Point2f> extractFeaturePoints(const cv::Mat& frame, const cv::Rect2f& bbox);
    
    /**
     * Update bounding box based on tracked feature points.
     */
    float calculateMedianScale(const std::vector<cv::Point2f>& old_points,
                               const std::vector<cv::Point2f>& new_points);

    /**
     * Validate bounding box to ensure it's within frame bounds and reasonable size.
     */
    bool isValidBoundingBox(const cv::Rect2f& bbox, const cv::Size& frame_size) const;
};

#endif // LIGHTWEIGHT_TRACKER_H