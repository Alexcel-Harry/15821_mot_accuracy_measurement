package edu.cmu.cs.face;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.RectF;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Minimal MOT Measurement App - No UI, Just Measurement
 *
 * Configure the path and FPS below, then run.
 * App will process the sequence and exit automatically.
 */
public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";

    static {
        System.loadLibrary("bytetrack_jni");
    }

    // ============================================================================
    // CONFIGURATION - MODIFY THESE VALUES
    // ============================================================================

    private static final String SEQUENCE_PATH = "/sdcard/Pictures/MOT17_resized_1280x720/train/MOT17-02-DPM";
    private static final int VIDEO_FPS = 30;
    private static final int TRACK_BUFFER = 30;

    // Set to 1 for Pure ByteTrack (C++ will bypass MOSSE), >1 for Hybrid
    private static final int KEYFRAME_INTERVAL = 3;

    private static final float CONFIDENCE_THRESHOLD = 0.1f;
    private static final int REQUEST_STORAGE_PERMISSION = 1002;
    private static final float NMS_THRESHOLD = 0.45f;

    // ============================================================================
    // END CONFIGURATION
    // ============================================================================

    private static final String MODEL_FILE = "yolo11n_full_integer_quant.tflite";

    private Interpreter tflite = null;
    private NnApiDelegate nnApiDelegate = null;
    private int[] inputShape = null;
    private DataType inputDataType = null;
    private int outputCount = 0;
    private int[][] outputShapes = null;
    private DataType[] outputDataTypes = null;
    private Bitmap reusableLetterboxBmp = null;  // Reuse to avoid allocations

    private long hybridTrackerHandle = 0;
    private int frameCounter = 0;

    // --- Global latency accumulators (in nanoseconds) ---
    private long totalPreprocessingTimeNs = 0;
    private long totalInferenceTimeNs = 0;
    private long totalPostprocessingTimeNs = 0;
    private long totalTrackingTimeNs = 0;
    private long totalOpticalFlowTimeNs = 0;

    // Native methods
    public native long nativeInitHybridTracker(int frameRate, int trackBuffer, int keyframeInterval);
    public native void nativeReleaseHybridTracker(long trackerPtr);
    public native float[] nativeUpdateWithDetections(long trackerPtr, float[] detections, byte[] imageData, int w, int h);
    public native float[] nativeUpdateWithoutDetections(long trackerPtr, byte[] imageData, int w, int h);

    private static class Detection {
        float cx, cy, w, h;
        int classId;
        float confidence;
        int trackId;

        Detection(float cx, float cy, float w, float h, int classId, float confidence, int trackId) {
            this.cx = cx;
            this.cy = cy;
            this.w = w;
            this.h = h;
            this.classId = classId;
            this.confidence = confidence;
            this.trackId = trackId;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Log.i(TAG, "=".repeat(60));
        Log.i(TAG, "MOT Measurement Starting");
        Log.i(TAG, "=".repeat(60));
        Log.i(TAG, "Sequence path: " + SEQUENCE_PATH);
        Log.i(TAG, "Video FPS: " + VIDEO_FPS);
        Log.i(TAG, "Track buffer: " + TRACK_BUFFER);
        Log.i(TAG, "Keyframe interval: " + KEYFRAME_INTERVAL);
        Log.i(TAG, "Model: " + MODEL_FILE);
        requestStoragePermission();
    }

    private void runMeasurement() {
        Log.i(TAG, "");
        Log.i(TAG, "Loading TFLite model...");
        if (!loadTFLiteModel(MODEL_FILE)) {
            Log.e(TAG, "ERROR: Failed to load model");
            return;
        }
        Log.i(TAG, "✓ Model loaded");

        Log.i(TAG, "Initializing tracker...");
        hybridTrackerHandle = nativeInitHybridTracker(VIDEO_FPS, TRACK_BUFFER, KEYFRAME_INTERVAL);
        if (hybridTrackerHandle == 0) {
            Log.e(TAG, "ERROR: Failed to initialize tracker");
            return;
        }
        Log.i(TAG, "✓ Tracker initialized (FPS=" + VIDEO_FPS + ")");

        File seqDir = new File(SEQUENCE_PATH);
        String sequenceName = seqDir.getName();
        File imgDir = new File(seqDir, "img1");

        Log.i(TAG, "");
        Log.i(TAG, "Checking sequence directory...");
        Log.i(TAG, "Path: " + imgDir.getAbsolutePath());

        if (!imgDir.exists()) {
            Log.e(TAG, "ERROR: Directory not found: " + imgDir.getAbsolutePath());
            return;
        }

        File[] allFiles = imgDir.listFiles();

        if (allFiles == null) {
            Log.e(TAG, "FATAL ERROR: allFiles array is NULL. Directory is inaccessible.");
            Log.e(TAG, ">>> LIKELY FIX: Grant READ_EXTERNAL_STORAGE permission in App Settings.");
            return;
        }

        List<File> imageFileList = new ArrayList<>();
        for (File f : allFiles) {
            String name = f.getName().toLowerCase();
            if (name.endsWith(".jpg") || name.endsWith(".png")) {
                imageFileList.add(f);
            }
        }
        File[] imageFiles = imageFileList.toArray(new File[0]);

        if (imageFiles.length == 0) {
            Log.e(TAG, "ERROR: No images found *after filtering*.");
            Log.e(TAG, "Found " + allFiles.length + " total items, but 0 matched the .jpg/.png filter.");
            return;
        }

        Arrays.sort(imageFiles, Comparator.comparing(File::getName));
        Log.i(TAG, "✓ Found " + imageFiles.length + " images");

        int framesToProcess = Math.min(imageFiles.length, 600);
        Log.i(TAG, "Processing " + framesToProcess + " frames for latency measurement");

        File appSpecificDir = getExternalFilesDir(null);
        File outputFile = new File(appSpecificDir, sequenceName + "_amortized_latency.csv");
        Log.i(TAG, "Output: " + outputFile.getAbsolutePath());

        Log.i(TAG, "");
        Log.i(TAG, "=".repeat(60));
        Log.i(TAG, "Starting processing...");
        Log.i(TAG, "=".repeat(60));

        long overallStartTime = System.currentTimeMillis();
        int processedFrames = 0;

        // Reset frame counter at start of measurement
        frameCounter = 0;

        for (int frameIdx = 0; frameIdx < framesToProcess; frameIdx++) {
            File imageFile = imageFiles[frameIdx];
            int frameNumber = frameIdx + 1;

            if (frameIdx % 100 == 0 && frameIdx > 0) {
                long elapsed = System.currentTimeMillis() - overallStartTime;
                float fps = (frameIdx * 1000.0f / elapsed);
                int progress = (frameIdx * 100) / framesToProcess;

                Log.i(TAG, String.format("Progress: %d%% (%d/%d) - %.1f FPS",
                        progress, frameNumber, framesToProcess, fps));
            }

            Bitmap frame = BitmapFactory.decodeFile(imageFile.getAbsolutePath());
            if (frame == null) {
                Log.w(TAG, "WARNING: Failed to load " + imageFile.getName());
                continue;
            }

            processFrame(frame, frameIdx);

            frame.recycle();
            processedFrames++;
        }

        long overallEndTime = System.currentTimeMillis();
        float totalSeconds = (overallEndTime - overallStartTime) / 1000f;

        double avgPrepMs = 0, avgInferenceMs = 0, avgPostMs = 0, avgTrackMs = 0, avgOpticalFlowMs = 0;

        if (processedFrames > 0) {
            avgPrepMs = (totalPreprocessingTimeNs / (double) processedFrames) / 1_000_000.0;
            avgInferenceMs = (totalInferenceTimeNs / (double) processedFrames) / 1_000_000.0;
            avgPostMs = (totalPostprocessingTimeNs / (double) processedFrames) / 1_000_000.0;
            avgTrackMs = (totalTrackingTimeNs / (double) processedFrames) / 1_000_000.0;
            avgOpticalFlowMs = (totalOpticalFlowTimeNs / (double) processedFrames) / 1_000_000.0;
        }
        double fps = 1000 / (avgPrepMs + avgInferenceMs + avgPostMs + avgTrackMs + avgOpticalFlowMs);
        double totalPrepMs = totalPreprocessingTimeNs / 1_000_000.0;
        double totalInferenceMs = totalInferenceTimeNs / 1_000_000.0;
        double totalPostMs = totalPostprocessingTimeNs / 1_000_000.0;
        double totalTrackMs = totalTrackingTimeNs / 1_000_000.0;
        double totalOpticalFlowMs = totalOpticalFlowTimeNs / 1_000_000.0;

        try (FileWriter writer = new FileWriter(outputFile)) {
            writer.write(String.format(Locale.US, "%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    avgPrepMs, avgInferenceMs, avgPostMs, avgTrackMs, avgOpticalFlowMs));
        } catch (IOException e) {
            Log.e(TAG, "ERROR: Failed to write amortized latency results", e);
        }

        Log.i(TAG, "");
        Log.i(TAG, "=".repeat(60));
        Log.i(TAG, "LATENCY MEASUREMENT COMPLETE");
        Log.i(TAG, "=".repeat(60));
        Log.i(TAG, "Sequence: " + sequenceName);
        Log.i(TAG, "Frames: " + processedFrames);
        Log.i(TAG, "Time: " + String.format(Locale.US, "%.1f", totalSeconds) + "s");
        Log.i(TAG, "FPS: " + String.format(Locale.US, "%.2f", fps));
        Log.i(TAG, "Output: " + outputFile.getAbsolutePath());
        Log.i(TAG, "=".repeat(60));

        Log.i(TAG, "Amortized Averages (ms):");
        Log.i(TAG, String.format(Locale.US, "  Preprocessing: %.4f ms", avgPrepMs));
        Log.i(TAG, String.format(Locale.US, "  Inference:     %.4f ms", avgInferenceMs));
        Log.i(TAG, String.format(Locale.US, "  Postprocessing:%.4f ms", avgPostMs));
        Log.i(TAG, String.format(Locale.US, "  Tracking:      %.4f ms", avgTrackMs));
        Log.i(TAG, String.format(Locale.US, "  Optical Flow:  %.4f ms", avgOpticalFlowMs));
        Log.i(TAG, "=".repeat(60));
        Log.i(TAG, "Total Stage Times (ms):");
        Log.i(TAG, String.format(Locale.US, "  Total Preprocessing: %.2f ms", totalPrepMs));
        Log.i(TAG, String.format(Locale.US, "  Total Inference:     %.2f ms", totalInferenceMs));
        Log.i(TAG, String.format(Locale.US, "  Total Postprocessing:%.2f ms", totalPostMs));
        Log.i(TAG, String.format(Locale.US, "  Total Tracking:      %.2f ms", totalTrackMs));
        Log.i(TAG, String.format(Locale.US, "  Total Optical Flow:  %.2f ms", totalOpticalFlowMs));
        Log.i(TAG, "=".repeat(60));
    }

    private List<Detection> processFrame(Bitmap frame, int frameIdx) {
        int originalW = frame.getWidth();
        int originalH = frame.getHeight();

        boolean isKeyframe = (frameCounter % KEYFRAME_INTERVAL == 0);
        List<Detection> result;

        // Grayscale conversion (runs on EVERY frame for optical flow)
        byte[] imageData = bitmapToGrayscale(frame);

        if (isKeyframe) {
            // ============================================================
            // KEYFRAME BRANCH - Run YOLO + Tracker Update
            // ============================================================
            Log.d(TAG, "=== KEYFRAME: Running YOLO ===");

            // YOLO detection (has its own internal timers)
            List<Detection> detections = runYOLODetection(frame);

            // Convert detections to flat array for JNI
            float[] detectArray = new float[detections.size() * 6];
            for (int i = 0; i < detections.size(); i++) {
                Detection d = detections.get(i);
                detectArray[i * 6] = d.cx;
                detectArray[i * 6 + 1] = d.cy;
                detectArray[i * 6 + 2] = d.w;
                detectArray[i * 6 + 3] = d.h;
                detectArray[i * 6 + 4] = (float)d.classId;
                detectArray[i * 6 + 5] = d.confidence;
            }

            // Update tracker with detections
            long trackStartTime = System.nanoTime();
            float[] trackerOutput = nativeUpdateWithDetections(
                    hybridTrackerHandle, detectArray, imageData, originalW, originalH);
            totalTrackingTimeNs += (System.nanoTime() - trackStartTime);

            result = parseTrackerOutput(trackerOutput);

        } else {
            // ============================================================
            // INTERMEDIATE BRANCH - Run Optical Flow Only
            // ============================================================
            Log.d(TAG, "=== INTERMEDIATE: Running MOSSE ===");

            // Native optical flow tracking
            long ofStartTime = System.nanoTime();
            float[] trackerOutput = nativeUpdateWithoutDetections(
                    hybridTrackerHandle, imageData, originalW, originalH);
            totalOpticalFlowTimeNs += (System.nanoTime() - ofStartTime);

            result = parseTrackerOutput(trackerOutput);
        }

        frameCounter++;
        return result;
    }

    private List<Detection> runYOLODetection(Bitmap originalBitmap) {
        if (tflite == null || inputShape == null) {
            return Collections.emptyList();
        }

        // ============================================================
        // START PREPROCESSING TIMING (includes letterboxing now!)
        // ============================================================
        long prepStartTime = System.nanoTime();

        int originalW = originalBitmap.getWidth();
        int originalH = originalBitmap.getHeight();
        int modelW = inputShape[2];
        int modelH = inputShape[1];

        // Calculate letterbox parameters
        float scale = Math.min((float) modelW / originalW, (float) originalH / originalH);
        int newW = Math.round(originalW * scale);
        int newH = Math.round(originalH * scale);
        int padX = (modelW - newW) / 2;
        int padY = (modelH - newH) / 2;

        // Reuse or create letterbox bitmap
        if (reusableLetterboxBmp == null ||
                reusableLetterboxBmp.getWidth() != modelW ||
                reusableLetterboxBmp.getHeight() != modelH) {
            reusableLetterboxBmp = Bitmap.createBitmap(modelW, modelH, Bitmap.Config.ARGB_8888);
        }

        // One-step letterboxing: draw with scaling directly using Rect
        android.graphics.Canvas canvas = new android.graphics.Canvas(reusableLetterboxBmp);
        canvas.drawColor(Color.rgb(128, 128, 128));
        android.graphics.Rect dst = new android.graphics.Rect(padX, padY, padX + newW, padY + newH);
        canvas.drawBitmap(originalBitmap, null, dst, null);

        // Get tensor info
        Tensor inputTensor = tflite.getInputTensor(0);
        DataType modelInputType = inputTensor.dataType();
        Tensor.QuantizationParams qp = inputTensor.quantizationParams();
        final int zeroPoint = qp.getZeroPoint();

        // Extract pixels
        int[] pixels = new int[modelW * modelH];
        reusableLetterboxBmp.getPixels(pixels, 0, modelW, 0, 0, modelW, modelH);

        // Prepare payload array first (more efficient than direct ByteBuffer writes)
        int pixelCount = modelW * modelH;
        byte[] payload = new byte[pixelCount * 3];

        // Convert pixels to appropriate format based on model data type
        int o = 0;
        if (modelInputType == DataType.FLOAT32) {
            // For float32, we still need to use ByteBuffer directly
            ByteBuffer inputBuffer = ByteBuffer.allocateDirect(inputTensor.numBytes());
            inputBuffer.order(ByteOrder.nativeOrder());
            for (int pixel : pixels) {
                float r = ((pixel >> 16) & 0xFF) / 255.0f;
                float g = ((pixel >> 8) & 0xFF) / 255.0f;
                float b = (pixel & 0xFF) / 255.0f;
                inputBuffer.putFloat(r);
                inputBuffer.putFloat(g);
                inputBuffer.putFloat(b);
            }
            inputBuffer.rewind();

            // END PREPROCESSING TIMING
            totalPreprocessingTimeNs += (System.nanoTime() - prepStartTime);

            // ============================================================
            // INFERENCE
            // ============================================================
            Object[] inputs = {inputBuffer};
            Map<Integer, Object> outputsMap = new HashMap<>();

            if (outputDataTypes[0] == DataType.FLOAT32) {
                outputsMap.put(0, new float[outputShapes[0][0]][outputShapes[0][1]][outputShapes[0][2]]);
            } else {
                outputsMap.put(0, new byte[outputShapes[0][0]][outputShapes[0][1]][outputShapes[0][2]]);
            }

            long inferenceStartTime = System.nanoTime();
            tflite.runForMultipleInputsOutputs(inputs, outputsMap);
            totalInferenceTimeNs += (System.nanoTime() - inferenceStartTime);

            // ============================================================
            // POSTPROCESSING
            // ============================================================
            long postStartTime = System.nanoTime();
            List<Detection> detections = decodeOutput(outputsMap, originalW, originalH, modelW, modelH, padX, padY, scale);
            totalPostprocessingTimeNs += (System.nanoTime() - postStartTime);

            return detections;

        } else if (modelInputType == DataType.UINT8) {
            for (int pixel : pixels) {
                payload[o++] = (byte) (((pixel >> 16) & 0xFF) - zeroPoint);
                payload[o++] = (byte) (((pixel >> 8) & 0xFF) - zeroPoint);
                payload[o++] = (byte) ((pixel & 0xFF) - zeroPoint);
            }
        } else if (modelInputType == DataType.INT8) {
            for (int pixel : pixels) {
                payload[o++] = (byte) (((pixel >> 16) & 0xFF) - zeroPoint);
                payload[o++] = (byte) (((pixel >> 8) & 0xFF) - zeroPoint);
                payload[o++] = (byte) ((pixel & 0xFF) - zeroPoint);
            }
        } else {
            Log.e(TAG, "Unsupported input data type: " + modelInputType);
            return Collections.emptyList();
        }

        // Bulk copy to ByteBuffer (more efficient)
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(inputTensor.numBytes());
        inputBuffer.order(ByteOrder.nativeOrder());
        inputBuffer.put(payload);
        inputBuffer.rewind();

        // END PREPROCESSING TIMING (now includes all preprocessing steps)
        totalPreprocessingTimeNs += (System.nanoTime() - prepStartTime);

        // ============================================================
        // INFERENCE
        // ============================================================
        Object[] inputs = {inputBuffer};
        Map<Integer, Object> outputsMap = new HashMap<>();

        if (outputDataTypes[0] == DataType.FLOAT32) {
            outputsMap.put(0, new float[outputShapes[0][0]][outputShapes[0][1]][outputShapes[0][2]]);
        } else {
            outputsMap.put(0, new byte[outputShapes[0][0]][outputShapes[0][1]][outputShapes[0][2]]);
        }

        long inferenceStartTime = System.nanoTime();
        tflite.runForMultipleInputsOutputs(inputs, outputsMap);
        totalInferenceTimeNs += (System.nanoTime() - inferenceStartTime);

        // ============================================================
        // POSTPROCESSING
        // ============================================================
        long postStartTime = System.nanoTime();
        List<Detection> detections = decodeOutput(outputsMap, originalW, originalH, modelW, modelH, padX, padY, scale);
        totalPostprocessingTimeNs += (System.nanoTime() - postStartTime);

        return detections;
    }

    private List<Detection> decodeOutput(Map<Integer, Object> outputsMap, int originalW, int originalH,
                                         int modelW, int modelH, int padX, int padY, float scale) {
        Tensor outputTensor = tflite.getOutputTensor(0);
        int[] outShape = outputTensor.shape();
        DataType outType = outputTensor.dataType();

        int numDetails = outShape[1];
        int numPredictions = outShape[2];

        if (numDetails < 5) {
            Log.e(TAG, "Model output is too small. Expected numDetails >= 5, but got " + numDetails);
            return Collections.emptyList();
        }

        float[][] dequantized = new float[numPredictions][numDetails];
        if (outType == DataType.FLOAT32) {
            float[][][] rawFloat = (float[][][]) outputsMap.get(0);
            for (int i = 0; i < numPredictions; i++) {
                for (int j = 0; j < numDetails; j++) {
                    dequantized[i][j] = rawFloat[0][j][i];
                }
            }
        } else {
            byte[][][] rawByte = (byte[][][]) outputsMap.get(0);
            Tensor.QuantizationParams qp = outputTensor.quantizationParams();
            float scale_q = qp.getScale();
            int zp = qp.getZeroPoint();
            boolean isUint8 = (outType == DataType.UINT8);

            for (int i = 0; i < numPredictions; i++) {
                for (int j = 0; j < numDetails; j++) {
                    int val = isUint8 ? (rawByte[0][j][i] & 0xFF) : rawByte[0][j][i];
                    dequantized[i][j] = (val - zp) * scale_q;
                }
            }
        }

        ArrayList<RectF> boxes = new ArrayList<>();
        ArrayList<Float> confidences = new ArrayList<>();
        ArrayList<Integer> classIds = new ArrayList<>();

        for (int i = 0; i < numPredictions; i++) {
            float[] row = dequantized[i];
            float maxScore = -1.0f;
            int classId = -1;

            for (int j = 4; j < numDetails; j++) {
                float score = row[j];
                if (score > maxScore) {
                    maxScore = score;
                    classId = j - 4;
                }
            }

            if (maxScore >= CONFIDENCE_THRESHOLD) {
                float cx = row[0];
                float cy = row[1];
                float w_norm = row[2];
                float h_norm = row[3];

                float pixel_w = w_norm * modelW;
                float pixel_h = h_norm * modelH;
                float left = (cx * modelW) - (pixel_w / 2f);
                float top = (cy * modelH) - (pixel_h / 2f);

                boxes.add(new RectF(left, top, left + pixel_w, top + pixel_h));
                confidences.add(maxScore);
                classIds.add(classId);
            }
        }

        List<Integer> indices = nonMaxSuppression(boxes, confidences);

        List<Detection> finalDetections = new ArrayList<>();
        float padXf = (float) padX;
        float padYf = (float) padY;

        for (int index : indices) {
            RectF box = boxes.get(index);

            float left_unpadded = box.left - padXf;
            float top_unpadded = box.top - padYf;
            float width_unpadded = box.width();
            float height_unpadded = box.height();

            float left_orig = left_unpadded / scale;
            float top_orig = top_unpadded / scale;
            float width_orig = width_unpadded / scale;
            float height_orig = height_unpadded / scale;

            float left_clipped = Math.max(0f, Math.min(left_orig, originalW));
            float top_clipped = Math.max(0f, Math.min(top_orig, originalH));
            float right_clipped = Math.max(0f, Math.min(left_orig + width_orig, originalW));
            float bottom_clipped = Math.max(0f, Math.min(top_orig + height_orig, originalH));
            float final_w = right_clipped - left_clipped;
            float final_h = bottom_clipped - top_clipped;

            float norm_cx = (left_clipped + final_w / 2f) / (float) originalW;
            float norm_cy = (top_clipped + final_h / 2f) / (float) originalH;
            float norm_w = final_w / (float) originalW;
            float norm_h = final_h / (float) originalH;

            if (final_w > 1 && final_h > 1) {
                finalDetections.add(
                        new Detection(norm_cx, norm_cy, norm_w, norm_h, classIds.get(index), confidences.get(index), -1)
                );
            }
        }

        return finalDetections;
    }

    private static List<Integer> nonMaxSuppression(ArrayList<RectF> boxes, ArrayList<Float> confidences) {
        List<Integer> selectedIndices = new ArrayList<>();
        if (boxes.isEmpty()) {
            return selectedIndices;
        }

        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < confidences.size(); i++) {
            indices.add(i);
        }
        indices.sort((a, b) -> Float.compare(confidences.get(b), confidences.get(a)));

        while (!indices.isEmpty()) {
            int current_index = indices.get(0);
            selectedIndices.add(current_index);
            indices.remove(0);

            RectF current_box = boxes.get(current_index);
            List<Integer> indices_to_remove = new ArrayList<>();

            for (int i = 0; i < indices.size(); i++) {
                int comparing_index = indices.get(i);
                RectF comparing_box = boxes.get(comparing_index);

                float interArea = Math.max(0, Math.min(current_box.right, comparing_box.right) -
                        Math.max(current_box.left, comparing_box.left)) *
                        Math.max(0, Math.min(current_box.bottom, comparing_box.bottom) -
                                Math.max(current_box.top, comparing_box.top));
                float unionArea = current_box.width() * current_box.height() +
                        comparing_box.width() * comparing_box.height() - interArea;
                float iou = (unionArea > 0f) ? (interArea / unionArea) : 0f;

                if (iou > NMS_THRESHOLD) {
                    indices_to_remove.add(i);
                }
            }

            for (int i = indices_to_remove.size() - 1; i >= 0; i--) {
                indices.remove((int) indices_to_remove.get(i));
            }
        }

        return selectedIndices;
    }

    private List<Detection> parseTrackerOutput(float[] trackerOutput) {
        List<Detection> result = new ArrayList<>();
        if (trackerOutput == null || trackerOutput.length == 0) {
            return result;
        }

        int numTracks = trackerOutput.length / 7;
        for (int i = 0; i < numTracks; i++) {
            float cx = trackerOutput[i * 7];
            float cy = trackerOutput[i * 7 + 1];
            float w = trackerOutput[i * 7 + 2];
            float h = trackerOutput[i * 7 + 3];
            int classId = (int) trackerOutput[i * 7 + 4];
            float conf = trackerOutput[i * 7 + 5];
            int trackId = (int) trackerOutput[i * 7 + 6];

            result.add(new Detection(cx, cy, w, h, classId, conf, trackId));
        }

        return result;
    }

    private byte[] bitmapToGrayscale(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        byte[] grayscale = new byte[width * height];

        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            int r = (pixel >> 16) & 0xFF;
            int g = (pixel >> 8) & 0xFF;
            int b = pixel & 0xFF;
            grayscale[i] = (byte) ((r + g + b) / 3);
        }

        return grayscale;
    }

    private boolean loadTFLiteModel(String assetFilename) {
        try {
            AssetFileDescriptor afd = getAssets().openFd(assetFilename);
            FileInputStream fis = new FileInputStream(afd.getFileDescriptor());
            FileChannel fc = fis.getChannel();
            MappedByteBuffer mb = fc.map(FileChannel.MapMode.READ_ONLY, afd.getStartOffset(), afd.getLength());

            Interpreter.Options opts = new Interpreter.Options();

            String deviceModel = Build.MODEL;
            String deviceManufacturer = Build.MANUFACTURER;
            boolean isPixelWithTensor = "Google".equalsIgnoreCase(deviceManufacturer) &&
                    (deviceModel.startsWith("Pixel 6") || deviceModel.startsWith("Pixel 7") ||
                            deviceModel.startsWith("Pixel 8") || deviceModel.startsWith("Pixel 9"));

            if (isPixelWithTensor) {
                try {
                    NnApiDelegate.Options nnApiOptions = new NnApiDelegate.Options();
                    nnApiOptions.setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED);
                    nnApiOptions.setAllowFp16(true);
                    nnApiOptions.setUseNnapiCpu(false);

                    nnApiDelegate = new NnApiDelegate(nnApiOptions);
                    opts.addDelegate(nnApiDelegate);

                    Log.i(TAG, "✓ NNAPI/TPU delegate enabled (ESSENTIAL)");
                } catch (Exception e) {
                    Log.w(TAG, "⚠ TPU initialization failed, falling back to CPU", e);
                }
            } else {
                Log.i(TAG, "Device: " + deviceManufacturer + " " + deviceModel + " (CPU only)");
            }

            opts.setNumThreads(4);
            opts.setUseXNNPACK(true);

            tflite = new Interpreter(mb, opts);

            Tensor inTensor = tflite.getInputTensor(0);
            inputShape = inTensor.shape();
            inputDataType = inTensor.dataType();

            outputCount = tflite.getOutputTensorCount();
            outputShapes = new int[outputCount][];
            outputDataTypes = new DataType[outputCount];
            for (int i = 0; i < outputCount; i++) {
                Tensor t = tflite.getOutputTensor(i);
                outputShapes[i] = t.shape();
                outputDataTypes[i] = t.dataType();
            }

            Log.i(TAG, "Input shape: " + Arrays.toString(inputShape));
            Log.i(TAG, "Input dtype: " + inputDataType.name());
            Log.i(TAG, "Output shape: " + Arrays.toString(outputShapes[0]));
            Log.i(TAG, "Output dtype: " + outputDataTypes[0].name());

            return true;

        } catch (Exception e) {
            Log.e(TAG, "Failed to load model", e);
            return false;
        }
    }

    private void cleanup() {
        Log.i(TAG, "");
        Log.i(TAG, "Cleaning up...");

        if (hybridTrackerHandle != 0) {
            nativeReleaseHybridTracker(hybridTrackerHandle);
            hybridTrackerHandle = 0;
        }

        if (tflite != null) {
            tflite.close();
            tflite = null;
        }

        if (nnApiDelegate != null) {
            nnApiDelegate.close();
            nnApiDelegate = null;
        }

        if (reusableLetterboxBmp != null && !reusableLetterboxBmp.isRecycled()) {
            reusableLetterboxBmp.recycle();
            reusableLetterboxBmp = null;
        }

        Log.i(TAG, "✓ Cleanup complete");
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cleanup();
    }

    private void requestStoragePermission() {
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_IMAGES) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.READ_MEDIA_IMAGES}, REQUEST_STORAGE_PERMISSION);
            } else {
                new Thread(this::runMeasurement).start();
            }
        } else {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, REQUEST_STORAGE_PERMISSION);
            } else {
                new Thread(this::runMeasurement).start();
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_STORAGE_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                new Thread(this::runMeasurement).start();
            } else {
                Toast.makeText(this, "Storage permission is required for benchmark", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }
}