package edu.cmu.cs.face;

import android.util.Log;
import android.os.Handler;
import android.os.Looper;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.view.PreviewView;

import java.lang.ref.WeakReference;


public class CameraManager {
    private static final String TAG = "CameraManager";

    public interface CameraStateListener {
        void onStarted(CameraSelector selector);
        void onStopped();
        void onError(Exception e);
    }

    private final WeakReference<AppCompatActivity> activityRef;
    private final PreviewView previewView;
    private final int width;
    private final int height;

    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private final Object lock = new Object();

    private volatile boolean useBackCamera = true;
    private volatile boolean started = false;
    private CameraSelector currentSelector = CameraSelector.DEFAULT_BACK_CAMERA;

    private edu.cmu.cs.mycamera.MyCameraCapture cameraCapture;
    private androidx.camera.core.ImageAnalysis.Analyzer analyzer;

    private CameraStateListener listener;

    public CameraManager(AppCompatActivity activity,
                         PreviewView previewView,
                         int width, int height,
                         androidx.camera.core.ImageAnalysis.Analyzer analyzer) {
        this.activityRef = new WeakReference<>(activity);
        this.previewView = previewView;
        this.width = width;
        this.height = height;
        this.analyzer = analyzer;
    }

    public void setCameraStateListener(CameraStateListener listener) {
        this.listener = listener;
    }

    public void startBackCamera() {
        startCamera(CameraSelector.DEFAULT_BACK_CAMERA, true);
    }

    public void startFrontCamera() {
        startCamera(CameraSelector.DEFAULT_FRONT_CAMERA, false);
    }

    public void toggleCamera() {
        if (useBackCamera) startFrontCamera();
        else startBackCamera();
    }

    public boolean isUsingBackCamera() {
        return useBackCamera;
    }

    public boolean isStarted() {
        return started;
    }

    public void setAnalyzer(androidx.camera.core.ImageAnalysis.Analyzer newAnalyzer) {
        synchronized (lock) {
            this.analyzer = newAnalyzer;
            if (started) {
                mainHandler.post(() -> {
                    synchronized (lock) {
                        try {
                            recreateCameraCapture();
                        } catch (Exception e) {
                            Log.w(TAG, "Failed to recreate camera with new analyzer", e);
                            if (listener != null) listener.onError(e);
                        }
                    }
                });
            }
        }
    }

    private void recreateCameraCapture() {
        synchronized (lock) {
            // shutdown old instance
            if (cameraCapture != null) {
                try {
                    cameraCapture.shutdown();
                } catch (Exception e) {
                    Log.w(TAG, "Error shutting down existing CameraCapture", e);
                }
                cameraCapture = null;
            }

            AppCompatActivity activity = activityRef.get();
            if (activity == null) {
                started = false;
                if (listener != null) listener.onError(new IllegalStateException("Activity was GC'd"));
                return;
            }

            try {
                // Construct new CameraCapture.
                cameraCapture = new edu.cmu.cs.mycamera.MyCameraCapture(
                        activity, analyzer, width, height, previewView, currentSelector, false);
                started = true;
                if (listener != null) listener.onStarted(currentSelector);
            } catch (Exception e) {
                started = false;
                Log.e(TAG, "Failed to recreate CameraCapture", e);
                if (listener != null) listener.onError(e);
            }
        }
    }

    private void startCamera(final CameraSelector selector, final boolean backCamera) {
        synchronized (lock) {
            currentSelector = selector;
            useBackCamera = backCamera;
            AppCompatActivity activity = activityRef.get();
            if (activity == null) {
                Log.w(TAG, "Activity gone, cannot start camera");
                if (listener != null) listener.onError(new IllegalStateException("Activity is null"));
                return;
            }
            mainHandler.post(() -> {
                synchronized (lock) {
                    try {
                        if (cameraCapture != null) {
                            try {
                                cameraCapture.shutdown();
                            } catch (Exception e) {
                                Log.w(TAG, "Error shutting down previous CameraCapture", e);
                            }
                            cameraCapture = null;
                        }
                        cameraCapture = new edu.cmu.cs.mycamera.MyCameraCapture(
                                activity, analyzer, width, height, previewView, selector, false);

                        started = true;
                        if (listener != null) listener.onStarted(selector);
                    } catch (Exception e) {
                        started = false;
                        Log.e(TAG, "Failed to start CameraCapture", e);
                        if (listener != null) listener.onError(e);
                    }
                }
            });
        }
    }

    public void release() {
        synchronized (lock) {
            mainHandler.post(() -> {
                synchronized (lock) {
                    try {
                        if (cameraCapture != null) {
                            try {
                                cameraCapture.shutdown();
                            } catch (Exception e) {
                                Log.w(TAG, "Error shutting down CameraCapture during release", e);
                            }
                            cameraCapture = null;
                        }
                        started = false;
                        if (listener != null) listener.onStopped();
                    } catch (Exception e) {
                        Log.w(TAG, "Unexpected error during release", e);
                        if (listener != null) listener.onError(e);
                    }
                }
            });
        }
    }
}
