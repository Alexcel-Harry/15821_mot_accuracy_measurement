package edu.cmu.cs.face;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class OverlayView extends View {
    private List<Detection> detections = Collections.emptyList();
    private final Paint boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint bgPaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    private static final int[] COLORS = new int[] {
            0xFFFF0000, // RED
            0xFF00FF00, // GREEN
            0xFF0000FF, // BLUE
            0xFF00FFFF, // CYAN
            0xFFFF00FF, // MAGENTA
            0xFFFFA500, // ORANGE
            0xFF8A2BE2, // BLUEVIOLET
            0xFF00CED1, // DARKTURQUOISE
            0xFF7FFF00, // CHARTREUSE
            0xFFFFFF00  // YELLOW
    };

    // OverlayView.java （在类内新增）
    private String perfText = "";
    private final android.graphics.Paint perfPaint = new android.graphics.Paint();

    {
        perfPaint.setColor(android.graphics.Color.YELLOW);
        perfPaint.setTextSize(36f); // 根据需要调整大小
        perfPaint.setAntiAlias(true);
        perfPaint.setStyle(android.graphics.Paint.Style.FILL);
        perfPaint.setShadowLayer(4f, 1f, 1f, android.graphics.Color.BLACK);
    }

    /** Called by MainActivity to update displayed performance metrics. */
    public void setPerfText(String text) {
        this.perfText = text;
        postInvalidate(); // request redraw on UI thread
    }


    public OverlayView(Context context) {
        super(context);
        init();
    }

    public OverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(6f);

        textPaint.setStyle(Paint.Style.FILL);
        textPaint.setTextSize(36f);
        textPaint.setColor(0xFFFFFFFF);

        bgPaint.setStyle(Paint.Style.FILL);
        bgPaint.setColor(0x80000000);
    }

    public void setDetections(List<Detection> list) {
        if (list == null) {
            this.detections = Collections.emptyList();
        } else {
            // copy to avoid concurrent modification
            this.detections = new ArrayList<>(list);
        }
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        final int width = getWidth();
        final int height = getHeight();
        if (width == 0 || height == 0) return;
        final int imgW = 1280;
        final int imgH = 720;

        for (Detection d : detections) {
            // (坐标计算保持不变)
            float scale = Math.max((float)width / imgW, (float)height / imgH);
            float dispW = imgW * scale;
            float dispH = imgH * scale;
            float offsetX = (width - dispW) / 2f;
            float offsetY = (height - dispH) / 2f;
            float boxCx = d.cx * imgW;
            float boxCy = d.cy * imgH;
            float boxW  = d.w  * imgW;
            float boxH  = d.h  * imgH;

            int left   = (int)(offsetX + (boxCx - boxW/2f) * scale);
            int top    = (int)(offsetY + (boxCy - boxH/2f) * scale);
            int right  = (int)(offsetX + (boxCx + boxW/2f) * scale);
            int bottom = (int)(offsetY + (boxCy + boxH/2f) * scale);

            // <-- 修复: 让框的颜色基于 trackId，而不是 classId
            // 这样同一个被跟踪的对象会保持相同的颜色
            int color = COLORS[Math.abs(d.trackId) % COLORS.length];
            boxPaint.setColor(color);

            int bgColor = (color & 0x00FFFFFF) | 0x80000000;
            bgPaint.setColor(bgColor);

            canvas.drawRect(left, top, right, bottom, boxPaint);

            // <-- 修复: 在标签中同时显示 ClassName, TrackID 和 Conf
            @SuppressLint("DefaultLocale")
            String label = getClassName(d.classId) + String.format(" [%d] (%.2f)", d.trackId, d.conf);

            float textWidth = textPaint.measureText(label);
            int padding = 8;
            int bgTop = Math.max(0, top - Math.round(textPaint.getTextSize()) - padding * 2);
            // 确保背景框不会超出屏幕右侧
            int bgRight = Math.min(width, left + Math.round(textWidth) + padding * 2);
            int bgBottom = bgTop + Math.round(textPaint.getTextSize()) + padding * 2;

            canvas.drawRect(left, bgTop, bgRight, bgBottom, bgPaint);
            float textX = left + padding;
            float textY = bgTop + padding - textPaint.ascent(); // ascent is negative
            canvas.drawText(label, textX, textY, textPaint);
        }
        if (perfText != null && !perfText.isEmpty()) {
            float x = 10f;
            float y = 40f; // 首行 baseline
            for (String line : perfText.split("\n")) {
                canvas.drawText(line, x, y, perfPaint);
                y += perfPaint.getTextSize() + 6f; // 行间距
            }
        }
    }

    private int clamp(int v, int a, int b) {
        return Math.max(a, Math.min(b, v));
    }

    // NOTE: keep consistent with your CLASS_NAMES or later replace with resource lookup
    private static final String[] CLASS_NAMES = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
            "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
            "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
            "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
            "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
            "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    private String getClassName(int id) {
        if (id >= 0 && id < CLASS_NAMES.length) return CLASS_NAMES[id];
        // 修复：处理-1的 classId (如果服务器在未识别时发送-1)
        if (id == -1) return "Obj";
        return "ID:" + id;
    }
}