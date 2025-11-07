package edu.cmu.cs.face;

public class Detection {
    public final float cx, cy, w, h;
    public final int classId;
    public final float conf;
    public final int trackId; // <-- 修复: 新增 trackId 字段

    public Detection(float cx, float cy, float w, float h, int classId, float conf, int trackId) { // <-- 修复: 增加 trackId 参数
        this.cx = cx; this.cy = cy; this.w = w; this.h = h;
        this.classId = classId; this.conf = conf;
        this.trackId = trackId; // <-- 修复: 初始化 trackId
    }
}