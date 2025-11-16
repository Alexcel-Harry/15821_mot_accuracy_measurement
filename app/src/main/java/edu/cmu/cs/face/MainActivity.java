package edu.cmu.cs.face;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.os.Bundle;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.RectF;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.widget.Toast;

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

    // ============================================================================
    // ⭐️ START: NEW GLOBAL TIMERS
    // ============================================================================
    private long totalProcessingNanos = 0;
    private long totalPreprocessingNanos = 0;
    private long totalFormatConversionNanos = 0;
    private long totalInferenceNanos = 0;
    private long totalPostprocessingNanos = 0;
    private long totalGrayscaleNanos = 0;
    private long totalJniTrackingNanos = 0;
    // ============================================================================
    // ⭐️ END: NEW GLOBAL TIMERS
    // ============================================================================


    static {
        System.loadLibrary("bytetrack_jni");
    }

    // ============================================================================
    // CONFIGURATION - MODIFY THESE VALUES
    // ============================================================================

    /**
     * Path to the sequence folder (containing img1/ subdirectory)
     *
     * Example: "/sdcard/MOT17/train/MOT17-02-DPM"
     *
     * Structure expected:
     * /sdcard/MOT17/train/MOT17-02-DPM/
     * img1/
     * 000001.jpg
     * 000002.jpg
     * ...
     */
    private static final String SEQUENCE_PATH = "/sdcard/Pictures/MOT17_resized_1280x720/train/MOT17-02-DPM";

    /**
     * FPS for Kalman filter
     * TODO: Set this to your video's actual frame rate!
     *
     * This affects:
     * - Kalman filter time step (dt = 1.0 / FPS)c
     * - Track buffer timeout
     * - Motion prediction
     */
    private static final int VIDEO_FPS = 30;  // TODO: CHANGE THIS!

    /**
     * Track buffer (frames to keep lost tracks)
     * Recommended: Set equal to VIDEO_FPS (1 second buffer)
     */
    private static final int TRACK_BUFFER = 30;  // TODO: Usually same as VIDEO_FPS

    /**
     * Keyframe interval (run YOLO every N frames)
     * Higher = faster, Lower = more accurate
     */
    private static final int KEYFRAME_INTERVAL = 3;

    /**
     * Detection confidence threshold
     */
    private static final float CONFIDENCE_THRESHOLD = 0.01f; // REVERTED from 0.4f. ByteTrack needs low-confidence detections.
    private static final int REQUEST_STORAGE_PERMISSION = 1002;
    /**
     * NMS IoU threshold
     */
    private static final float NMS_THRESHOLD = 0.4f;

    // ============================================================================
    // END CONFIGURATION
    // ============================================================================

    // Model file name
    private static final String MODEL_FILE = "yolo11n_finetune_full_integer_quant.tflite";

    // TFLite interpreter
    private Interpreter tflite = null;
    private NnApiDelegate nnApiDelegate = null;
    private int[] inputShape = null;
    private DataType inputDataType = null;
    private int outputCount = 0;
    private int[][] outputShapes = null;
    private DataType[] outputDataTypes = null;

    // Tracker
    private long hybridTrackerHandle = 0;

    // Native methods
    public native long nativeInitHybridTracker(int frameRate, int trackBuffer, int keyframeInterval);
    public native void nativeReleaseHybridTracker(long trackerPtr);
    public native boolean nativeIsKeyframe(long trackerPtr);
    public native float[] nativeUpdateWithDetections(long trackerPtr, float[] detections, byte[] imageData, int w, int h);
    public native float[] nativeUpdateWithoutDetections(long trackerPtr, byte[] imageData, int w, int h);

    // Detection class
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
        // Load model
        Log.i(TAG, "");
        Log.i(TAG, "Loading TFLite model...");
        if (!loadTFLiteModel(MODEL_FILE)) {
            Log.e(TAG, "ERROR: Failed to load model");
            return;
        }
        Log.i(TAG, "✓ Model loaded");

        // Initialize tracker
        Log.i(TAG, "Initializing tracker...");
        hybridTrackerHandle = nativeInitHybridTracker(VIDEO_FPS, TRACK_BUFFER, KEYFRAME_INTERVAL);
        if (hybridTrackerHandle == 0) {
            Log.e(TAG, "ERROR: Failed to initialize tracker");
            return;
        }
        Log.i(TAG, "✓ Tracker initialized (FPS=" + VIDEO_FPS + ")");

        // Get sequence info
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

        // 2. Check for null (This is the most common permissions error)
        if (allFiles == null) {
            Log.e(TAG, "FATAL ERROR: allFiles array is NULL. Directory is inaccessible.");
            Log.e(TAG, ">>> LIKELY FIX: Grant READ_EXTERNAL_STORAGE permission in App Settings.");
            return;
        }

        // 4. Now, filter the list in Java
        List<File> imageFileList = new ArrayList<>();
        for (File f : allFiles) {
            String name = f.getName().toLowerCase();
            if (name.endsWith(".jpg") || name.endsWith(".png")) {
                imageFileList.add(f);
            }
        }
        File[] imageFiles = imageFileList.toArray(new File[0]); // Convert List to Array

        // ====================================================================
        // END DEBUGGING BLOCK
        // ====================================================================


        // 5. Your original check, now much more informative
        if (imageFiles.length == 0) {
            Log.e(TAG, "ERROR: No images found *after filtering*.");
            Log.e(TAG, "Found " + allFiles.length + " total items, but 0 matched the .jpg/.png filter.");
            return;
        }

        Arrays.sort(imageFiles, Comparator.comparing(File::getName));
        Log.i(TAG, "✓ Found " + imageFiles.length + " images");

        // Output file
        File appSpecificDir = getExternalFilesDir("results");
        File outputFile = new File(appSpecificDir, sequenceName + "_results.txt");
        // This will save the file to a path like:
        // /sdcard/Android/data/edu.cmu.cs.face/files/MOT17-02-DPM_results.txt

        Log.i(TAG, "Output: " + outputFile.getAbsolutePath());

        Log.i(TAG, "");
        Log.i(TAG, "=".repeat(60));
        Log.i(TAG, "Starting processing...");
        Log.i(TAG, "=".repeat(60));

        // Process sequence
        try (FileWriter writer = new FileWriter(outputFile)) {
            long startTime = System.currentTimeMillis();
            int processedFrames = 0;
            int totalDetections = 0;

            for (int frameIdx = 0; frameIdx < imageFiles.length; frameIdx++) {
                File imageFile = imageFiles[frameIdx];
                int frameNumber = frameIdx + 1;

//                // Log progress every 100 frames
//                if (frameIdx % 100 == 0 && frameIdx > 0) {
//                    long elapsed = System.currentTimeMillis() - startTime;
//                    float fps = (frameIdx * 1000.0f / elapsed);
//                    int progress = (frameIdx * 100) / imageFiles.length;
//
//                    Log.i(TAG, String.format("Progress: %d%% (%d/%d) - %.1f FPS",
//                            progress, frameNumber, imageFiles.length, fps));
//                }

                // Load frame

                Bitmap frame = BitmapFactory.decodeFile(imageFile.getAbsolutePath());
                long frameStartTime = System.nanoTime();
                if (frame == null) {
                    Log.w(TAG, "WARNING: Failed to load " + imageFile.getName());
                    continue;
                }

                // Process frame
                List<Detection> trackedObjects = processFrame(frame, frameIdx);

                // Write detections
                for (Detection det : trackedObjects) {
                    // Filter for person (classId == 0) HERE, after tracking
                    if (det.trackId > 0 && det.classId == 0) {
                        int imgW = frame.getWidth();
                        int imgH = frame.getHeight();

                        float centerX = det.cx * imgW;
                        float centerY = det.cy * imgH;
                        float width = det.w * imgW;
                        float height = det.h * imgH;

                        float left = centerX - width / 2f;
                        float top = centerY - height / 2f;

                        String line = String.format(Locale.US, "%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,-1,-1,-1\n",
                                frameNumber,
                                det.trackId,
                                left,
                                top,
                                width,
                                height,
                                det.confidence);
                        writer.write(line);
                        totalDetections++;
                    }
                }

                frame.recycle();
                processedFrames++;
                long frameTime = System.nanoTime() - frameStartTime;
                totalProcessingNanos += frameTime;
            }

//            long endTime = System.currentTimeMillis();
            float totalSeconds = totalProcessingNanos / 1000000000f;
            float fps = processedFrames / totalSeconds;

            Log.i(TAG, "");
            Log.i(TAG, "=".repeat(60));
            Log.i(TAG, "MEASUREMENT COMPLETE");
            Log.i(TAG, "=".repeat(60));
            Log.i(TAG, "Sequence: " + sequenceName);
            Log.i(TAG, "Frames: " + processedFrames);
            Log.i(TAG, "Detections: " + totalDetections);
            Log.i(TAG, "Time: " + String.format("%.1f", totalSeconds) + "s");
            Log.i(TAG, "FPS (Total): " + String.format("%.2f", fps));
            Log.i(TAG, "Output: " + outputFile.getAbsolutePath());


            // ============================================================================
            // ⭐️ START: NEW TIMING REPORT BLOCK
            // ============================================================================
            Log.i(TAG, "--- Average Amortized Timings (ms) ---");
            if (processedFrames > 0) {
                double N = (double) processedFrames;
                double NANO_TO_MS = 1_000_000.0;

                // These timings are amortized over ALL frames (N),
                // as requested.
                Log.i(TAG, String.format(Locale.US, "1. Preprocessing:   %.3f ms", (totalPreprocessingNanos / N / NANO_TO_MS)));
                Log.i(TAG, String.format(Locale.US, "2. Format Convert:  %.3f ms", (totalFormatConversionNanos / N / NANO_TO_MS)));
                Log.i(TAG, String.format(Locale.US, "3. Inference:       %.3f ms", (totalInferenceNanos / N / NANO_TO_MS)));
                Log.i(TAG, String.format(Locale.US, "4. Postprocessing:  %.3f ms", (totalPostprocessingNanos / N / NANO_TO_MS)));
                Log.i(TAG, String.format(Locale.US, "5. Grayscale (OF):  %.3f ms", (totalGrayscaleNanos / N / NANO_TO_MS)));
                Log.i(TAG, String.format(Locale.US, "6. JNI (Track/OF):  %.3f ms", (totalJniTrackingNanos / N / NANO_TO_MS)));

                double sumOfPartsMs = (totalPreprocessingNanos + totalFormatConversionNanos + totalInferenceNanos +
                        totalPostprocessingNanos + totalGrayscaleNanos + totalJniTrackingNanos) / N / NANO_TO_MS;
                double totalMs = (totalProcessingNanos / N / NANO_TO_MS);

                Log.i(TAG, "-------------------------------------");
                Log.i(TAG, String.format(Locale.US, "Sum of Parts:     %.3f ms", sumOfPartsMs));
                Log.i(TAG, String.format(Locale.US, "Total Per-Frame:  %.3f ms (1000/%.2f)", totalMs, fps));
                Log.i(TAG, String.format(Locale.US, "Other (Load/etc): %.3f ms", (totalMs - sumOfPartsMs)));
            }
            // ============================================================================
            // ⭐️ END: NEW TIMING REPORT BLOCK
            // ============================================================================


            Log.i(TAG, "=".repeat(60));

        } catch (IOException e) {
            Log.e(TAG, "ERROR: Failed to write results", e);
        }
    }

    /**
     * ⭐️ MODIFIED: This function now times Grayscale and JNI calls.
     */
    private List<Detection> processFrame(Bitmap frame, int frameIdx) {
        int originalW = frame.getWidth();
        int originalH = frame.getHeight();

        boolean isKeyframe = nativeIsKeyframe(hybridTrackerHandle);
        float[] trackerOutput;
        float[] detectArray = null; // Declare outside

        if (isKeyframe) {
            // Detection (which includes Pre, Convert, Infer, Post) + tracking
            // Timers for these are inside runYOLODetection()
            List<Detection> detections = runYOLODetection(frame);

            detectArray = new float[detections.size() * 6];
            for (int i = 0; i < detections.size(); i++) {
                Detection d = detections.get(i);
                detectArray[i * 6] = d.cx;
                detectArray[i * 6 + 1] = d.cy;
                detectArray[i * 6 + 2] = d.w;
                detectArray[i * 6 + 3] = d.h;
                detectArray[i * 6 + 4] = (float)d.classId; // SWAPPED
                detectArray[i * 6 + 5] = d.confidence;   // SWAPPED
            }
        }

        // --- [5. Grayscale (Optical Flow Prep)] ---
        long startGray = System.nanoTime();
        byte[] imageData = bitmapToGrayscale(frame);
        long endGray = System.nanoTime();
        totalGrayscaleNanos += (endGray - startGray);
        // --- [END Grayscale] ---


        // --- [6. JNI (Tracking / Optical Flow)] ---
        long startJNI = System.nanoTime();
        if (isKeyframe) {
            trackerOutput = nativeUpdateWithDetections(
                    hybridTrackerHandle, detectArray, imageData, originalW, originalH);
        } else {
            // Tracking only
            trackerOutput = nativeUpdateWithoutDetections(
                    hybridTrackerHandle, imageData, originalW, originalH);
        }
        long endJNI = System.nanoTime();
        totalJniTrackingNanos += (endJNI - startJNI);
        // --- [END JNI] ---

        return parseTrackerOutput(trackerOutput);
    }


    /**
     * ⭐️ MODIFIED: This function now times Preprocessing, Conversion, Inference, and Postprocessing.
     */
    private List<Detection> runYOLODetection(Bitmap originalBitmap) {
        if (tflite == null || inputShape == null) {
            return Collections.emptyList();
        }

        int originalW = originalBitmap.getWidth();
        int originalH = originalBitmap.getHeight();
        int modelW = inputShape[2]; // Assumes [1, H, W, C]
        int modelH = inputShape[1];


        // --- [1. Preprocessing] ---
        long startPre = System.nanoTime();
        // Letterbox resize
        float scale = Math.min((float) modelW / originalW, (float) modelH / originalH);
        int newW = Math.round(originalW * scale);
        int newH = Math.round(originalH * scale);
        int padX = (modelW - newW) / 2;
        int padY = (modelH - newH) / 2;

        Bitmap resized = Bitmap.createScaledBitmap(originalBitmap, newW, newH, true);
        Bitmap letterbox = Bitmap.createBitmap(modelW, modelH, Bitmap.Config.ARGB_8888);
        android.graphics.Canvas canvas = new android.graphics.Canvas(letterbox);
        canvas.drawColor(Color.rgb(114, 114, 114)); // Same padding color as your code
        canvas.drawBitmap(resized, padX, padY, null);
        resized.recycle();
        long endPre = System.nanoTime();
        totalPreprocessingNanos += (endPre - startPre);
        // --- [END Preprocessing] ---


        // --- [2. Format Conversion] ---
        long startConvert = System.nanoTime();
        // --- [START] FIXED CONVERSION BLOCK ---

        // Get the input tensor details (loaded in loadTFLiteModel)
        Tensor inputTensor = tflite.getInputTensor(0);
        DataType modelInputType = inputTensor.dataType(); // Get the *actual* model type

        // Get model quantization parameters (if they exist)
        Tensor.QuantizationParams qp = inputTensor.quantizationParams();
        final int zeroPoint = qp.getZeroPoint();

        // Allocate the input buffer with the exact size the model expects
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(inputTensor.numBytes());
        inputBuffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[modelW * modelH];
        letterbox.getPixels(pixels, 0, modelW, 0, 0, modelW, modelH);
        letterbox.recycle(); // Recycle the letterbox bitmap

        inputBuffer.rewind(); // Ensure buffer is at position 0

        // --- This is the core logic ---
        if (modelInputType == DataType.FLOAT32) {
            // Model is FLOAT32: Convert 0-255 int to 0.0f-1.0f float
            // Log.d(TAG, "Converting to FLOAT32"); // Removed for performance
            for (int pixel : pixels) {
                float r = ((pixel >> 16) & 0xFF) / 255.0f;
                float g = ((pixel >> 8) & 0xFF) / 255.0f;
                float b = (pixel & 0xFF) / 255.0f;
                inputBuffer.putFloat(r);
                inputBuffer.putFloat(g);
                inputBuffer.putFloat(b);
            }
        } else if (modelInputType == DataType.UINT8) {
            // Model is UINT8: Use raw 0-255 byte values, subtract zeroPoint (usually 0)
            // Log.d(TAG, "Converting to UINT8 (zeroPoint=" + zeroPoint + ")"); // Removed for performance
            for (int pixel : pixels) {
                byte r = (byte) (((pixel >> 16) & 0xFF) - zeroPoint);
                byte g = (byte) (((pixel >> 8) & 0xFF) - zeroPoint);
                byte b = (byte) ((pixel & 0xFF) - zeroPoint);
                inputBuffer.put(r);
                inputBuffer.put(g);
                inputBuffer.put(b);
            }
        } else if (modelInputType == DataType.INT8) {
            // Model is INT8: Subtract the zero-point (often 128)
            // Log.d(TAG, "Converting to INT8 (zeroPoint=" + zeroPoint + ")"); // Removed for performance
            for (int pixel : pixels) {
                // This converts the 0-255 pixel value to the model's required -128 to 127 range
                byte r = (byte) (((pixel >> 16) & 0xFF) - zeroPoint);
                byte g = (byte) (((pixel >> 8) & 0xFF) - zeroPoint);
                byte b = (byte) ((pixel & 0xFF) - zeroPoint);
                inputBuffer.put(r);
                inputBuffer.put(g);
                inputBuffer.put(b);
            }
        } else {
            Log.e(TAG, "Unsupported input data type: " + modelInputType);
            return Collections.emptyList();
        }

        inputBuffer.rewind(); // Rewind again before passing to TFLite
        long endConvert = System.nanoTime();
        totalFormatConversionNanos += (endConvert - startConvert);
        // --- [END] FIXED CONVERSION BLOCK ---
        // --- [END Format Conversion] ---


        // Run inference
        Object[] inputs = {inputBuffer}; // Pass the correctly formatted buffer
        Map<Integer, Object> outputsMap = new HashMap<>();

        // This part remains the same
        if (outputDataTypes[0] == DataType.FLOAT32) {
            outputsMap.put(0, new float[outputShapes[0][0]][outputShapes[0][1]][outputShapes[0][2]]);
        } else {
            outputsMap.put(0, new byte[outputShapes[0][0]][outputShapes[0][1]][outputShapes[0][2]]);
        }

        // --- [3. Inference] ---
        long startInfer = System.nanoTime();
        tflite.runForMultipleInputsOutputs(inputs, outputsMap);
        long endInfer = System.nanoTime();
        totalInferenceNanos += (endInfer - startInfer);
        // --- [END Inference] ---


        // --- [4. Postprocessing] ---
        long startPost = System.nanoTime();
        List<Detection> results = decodeOutput(outputsMap, originalW, originalH, modelW, modelH, padX, padY, scale);
        long endPost = System.nanoTime();
        totalPostprocessingNanos += (endPost - startPost);
        // --- [END Postprocessing] ---

        return results;
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

        // Dequantize output
        float[][] dequantized = new float[numPredictions][numDetails];
        if (outType == DataType.FLOAT32) {
            float[][][] rawFloat = (float[][][]) outputsMap.get(0);
            for (int i = 0; i < numPredictions; i++) {
                for (int j = 0; j < numDetails; j++) {
                    dequantized[i][j] = rawFloat[0][j][i];
                }
            }
        } else {
            // Quantized model (INT8/UINT8)
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

        // --- MODIFIED BLOCK START ---
        // Based on user feedback to fix classId mismatch and filter for people
        //
        // This logic assumes a multi-class YOLO output format like:
        // [cx, cy, w, h, class0_score, class1_score, ..., classN_score]
        //
        // Fix 1: Find max score and correct classId (j - 4)
        // Fix 2: Filter for 'person' (classId == 0) and confidence

        ArrayList<RectF> boxes = new ArrayList<>();
        ArrayList<Float> confidences = new ArrayList<>();
        ArrayList<Integer> classIds = new ArrayList<>(); // ADDED: We need to pass the classId to the tracker

        for (int i = 0; i < numPredictions; i++) {
            float[] row = dequantized[i];

            // Find the max class score and its ID
            // (Assumes class scores start at index 4)
            float maxScore = -1.0f;
            int classId = -1;

            // Loop starts at 4, as 0-3 are box coords
            for (int j = 4; j < numDetails; j++) {
                float score = row[j];
                if (score > maxScore) {
                    maxScore = score;
                    classId = j - 4; // User's specified fix for classId mismatch
                }
            }

            // Apply confidence threshold. We pass ALL classes to the native tracker.
            if (maxScore >= CONFIDENCE_THRESHOLD) { // REMOVED classId == 0 filter
                // Bounding box coordinates are at indices 0-3
                float cx = row[0];
                float cy = row[1];
                float w_norm = row[2];
                float h_norm = row[3];

                // Convert from normalized [0,1] to pixel coordinates [0, modelW]
                float pixel_w = w_norm * modelW;
                float pixel_h = h_norm * modelH;
                float left = (cx * modelW) - (pixel_w / 2f);
                float top = (cy * modelH) - (pixel_h / 2f);

                boxes.add(new RectF(left, top, left + pixel_w, top + pixel_h));
                confidences.add(maxScore);
                classIds.add(classId); // ADDED: Store the classId
            }
        }
        // --- MODIFIED BLOCK END ---


        // NMS
        List<Integer> indices = nonMaxSuppression(boxes, confidences);

        // Map to original coordinates
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
                        // Pass the actual classId to the tracker
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

            // --- FIX ---
            // The JNI returns [..., classId, conf, trackId]
            // We had them swapped. This matches the reference file.
            int classId = (int) trackerOutput[i * 7 + 4];
            float conf = trackerOutput[i * 7 + 5];
            int trackId = (int) trackerOutput[i * 7 + 6];
            // --- END FIX ---

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

            // Check for Pixel TPU (ESSENTIAL - DO NOT REMOVE)
            String deviceModel = Build.MODEL;
            String deviceManufacturer = Build.MANUFACTURER;
            boolean isPixelWithTensor = "Google".equalsIgnoreCase(deviceManufacturer) &&
                    (deviceModel.startsWith("Pixel 6") || deviceModel.startsWith("Pixel 7") ||
                            deviceModel.startsWith("Pixel 8") || deviceModel.startsWith("Pixel 9"));

            if (isPixelWithTensor) {
                try {
                    NnApiDelegate.Options nnApiOptions = new NnApiDelegate.Options();
                    nnApiOptions.setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED);
                    nnApiOptions.setAllowFp16(false);
                    nnApiOptions.setUseNnapiCpu(true);

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

        Log.i(TAG, "✓ Cleanup complete");
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cleanup();
    }
    private void requestStoragePermission() {
        // Running heavy processing on main thread causes ANR (App Not Responding)
        // We MUST move runMeasurement() to a background thread.

        // 检查 Android 13+ (API 33+)
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_IMAGES) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.READ_MEDIA_IMAGES}, REQUEST_STORAGE_PERMISSION);
            } else {
                // Start on background thread
                new Thread(this::runMeasurement).start(); // 权限已存在
            }
        } else {
            // Android 12 及以下
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, REQUEST_STORAGE_PERMISSION);
            } else {
                // Start on background thread
                new Thread(this::runMeasurement).start(); // 权限已存在
            }
        }
    }

    /**
     * [MODIFIED] 处理权限结果
     */
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_STORAGE_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // [GOOD] 权限被授予, Start on background thread to prevent ANR
                new Thread(this::runMeasurement).start();
            } else {
                // [BAD] 权限被拒绝
                Toast.makeText(this, "Storage permission is required for benchmark", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }
}