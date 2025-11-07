package edu.cmu.cs.face;

import android.util.Log;
import java.util.ArrayList;
import java.util.List;

import edu.cmu.cs.gabriel.protocol.Protos.ResultWrapper;

public class ResultParser {
    private static final String TAG = "ResultParser";

    /**
     * Parse a ResultWrapper into a list of Detection objects.
     * This method tolerates malformed entries and logs warnings instead of throwing.
     */
    public List<Detection> parse(ResultWrapper wrapper) {
        List<Detection> out = new ArrayList<>();
        if (wrapper == null) return out;
        try {
            for (ResultWrapper.Result r : wrapper.getResultsList()) {
                if (r.getPayloadType() != edu.cmu.cs.gabriel.protocol.Protos.PayloadType.TEXT) continue;
                String payload = r.getPayload().toStringUtf8();
                out.addAll(parsePayloadText(payload));
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to parse ResultWrapper", e);
        }
        return out;
    }

    /**
     * payload string example: "x,y,w,h,classID,conf,trackID;..." (7 fields)
     */
    public List<Detection> parsePayloadText(String payload) {
        List<Detection> out = new ArrayList<>();
        if (payload == null || payload.trim().isEmpty()) return out;
        String[] bboxes = payload.split(";");
        for (String bbox : bboxes) {
            if (bbox == null) continue;
            String[] parts = bbox.trim().split(",");

            // <-- 修复: 至少需要7个字段
            if (parts.length < 7) {
                // 如果仍然收到6个字段（例如，来自没有跟踪的检测），我们也可以处理它
                if (parts.length == 6) {
                    try {
                        float x = Float.parseFloat(parts[0]);
                        float y = Float.parseFloat(parts[1]);
                        float w = Float.parseFloat(parts[2]);
                        float h = Float.parseFloat(parts[3]);
                        int classID = Integer.parseInt(parts[4]);
                        float conf = Float.parseFloat(parts[5]);
                        out.add(new Detection(x, y, w, h, classID, conf, -1)); // 传入 -1 作为 trackId
                    } catch (NumberFormatException nfe) {
                        Log.w(TAG, "Skipping malformed 6-part bbox: " + bbox);
                    }
                } else {
                    Log.w(TAG, "Skipping bbox with unexpected part count: " + parts.length);
                }
                continue; // 跳过这个 bbox
            }

            try {
                float x = Float.parseFloat(parts[0]);
                float y = Float.parseFloat(parts[1]);
                float w = Float.parseFloat(parts[2]);
                float h = Float.parseFloat(parts[3]);
                int classID = Integer.parseInt(parts[4]);
                float conf = Float.parseFloat(parts[5]);
                int trackId = Integer.parseInt(parts[6]); // <-- 修复: 解析第7个字段 (track_id)

                // <-- 修复: 传入 trackId
                out.add(new Detection(x, y, w, h, classID, conf, trackId));
            } catch (NumberFormatException nfe) {
                // Malformed numbers: skip this bbox but continue parsing others.
                Log.w(TAG, "Skipping malformed 7-part bbox: " + bbox);
            } catch (Exception e) {
                Log.w(TAG, "Unexpected parse error for bbox: " + bbox, e);
            }
        }
        return out;
    }
}