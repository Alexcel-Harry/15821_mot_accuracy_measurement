package edu.cmu.cs.face;

import android.app.Application;
import android.util.Log;

import java.lang.reflect.Method;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import java.util.function.Consumer;

import edu.cmu.cs.gabriel.client.comm.ServerComm;
import edu.cmu.cs.gabriel.client.results.ErrorType;
import edu.cmu.cs.gabriel.protocol.Protos.InputFrame;
import com.google.protobuf.ByteString;


public class ServerClientWrapper {
    private static final String TAG = "ServerClientWrapper";
    private final ExecutorService senderExecutor = Executors.newSingleThreadExecutor();
    private ServerComm serverComm;
    private final Consumer<edu.cmu.cs.gabriel.protocol.Protos.ResultWrapper> resultConsumer;
    private final Application app;
    private final int port;
    private final String host;
    private final Consumer<ErrorType> onDisconnectInternal;
    private volatile boolean started = false;

    public ServerClientWrapper(Application app,
                               String host,
                               int port,
                               Consumer<edu.cmu.cs.gabriel.protocol.Protos.ResultWrapper> resultConsumer,
                               Consumer<ErrorType> onDisconnect) {
        this.app = app;
        this.host = host;
        this.port = port;
        this.resultConsumer = resultConsumer;
        this.onDisconnectInternal = onDisconnect;
    }

    public synchronized void start() {
        if (serverComm != null) return;
        try {
            serverComm = ServerComm.createServerComm(
                    resultConsumer,
                    host,
                    port,
                    app,
                    (err) -> {
                        Log.e(TAG, "Server disconnected: " + err);
                        if (onDisconnectInternal != null) onDisconnectInternal.accept(err);
                    });
            started = true;
        } catch (Exception e) {
            Log.e(TAG, "Failed to create ServerComm", e);
            started = false;
        }
    }

    /**
     * Try to stop/close the ServerComm in a defensive way.
     * Tries common method names first, then falls back to reflection attempts.
     */
    public synchronized void stop() {
        if (serverComm == null) return;
        try {
            // First try common explicit methods (if available)
            tryInvokeNoArg(serverComm, "close");
            tryInvokeNoArg(serverComm, "shutdown");
            tryInvokeNoArg(serverComm, "stop");
            tryInvokeNoArg(serverComm, "disconnect");
        } catch (Exception e) {
            Log.w(TAG, "Exception while trying to stop ServerComm (ignored)", e);
        } finally {
            serverComm = null;
            started = false;
        }
    }

    /**
     * Send image asynchronously. No-op if server not started.
     */
    public void sendImageAsync(final String source, final ByteString jpegByteString) {
        if (serverComm == null) return;
        senderExecutor.execute(() -> {
            try {
                // Prefer sendSupplier if available; fallback to send(InputFrame,...)
                boolean usedSupplier = false;
                try {
                    // try call sendSupplier directly (most likely available)
                    Method m = serverComm.getClass().getMethod("sendSupplier", java.util.function.Supplier.class, String.class, boolean.class);
                    // invoke with a Supplier<InputFrame>
                    java.util.function.Supplier<InputFrame> supplier = () -> InputFrame.newBuilder()
                            .setPayloadType(edu.cmu.cs.gabriel.protocol.Protos.PayloadType.IMAGE)
                            .addPayloads(jpegByteString)
                            .build();
                    m.invoke(serverComm, supplier, source, Boolean.FALSE);
                    usedSupplier = true;
                } catch (NoSuchMethodException nsme) {
                    // fallback to send(InputFrame, String, boolean)
                }
                if (!usedSupplier) {
                    // fallback: try send(...)
                    try {
                        Method m2 = serverComm.getClass().getMethod("send", InputFrame.class, String.class, boolean.class);
                        InputFrame f = InputFrame.newBuilder()
                                .setPayloadType(edu.cmu.cs.gabriel.protocol.Protos.PayloadType.IMAGE)
                                .addPayloads(jpegByteString)
                                .build();
                        m2.invoke(serverComm, f, source, Boolean.FALSE);
                    } catch (NoSuchMethodException nsme2) {
                        // As a last resort, try to call a method named 'sendSupplier' via reflection with raw Runnable style.
                        Log.w(TAG, "ServerComm does not expose expected send/sendSupplier methods; image not sent.");
                    }
                }
            } catch (Exception e) {
                Log.w(TAG, "Failed to send image", e);
            }
        });
    }

    /**
     * Send an empty frame asynchronously (non-blocking).
     */
    public void sendEmptyFrameAsync(final String source) {
        if (serverComm == null) return;
        senderExecutor.execute(() -> {
            try {
                // try direct send(InputFrame, String, boolean)
                try {
                    Method m = serverComm.getClass().getMethod("send", InputFrame.class, String.class, boolean.class);
                    m.invoke(serverComm, InputFrame.newBuilder().build(), source, Boolean.FALSE);
                } catch (NoSuchMethodException nsme) {
                    // last fallback: try invoking a generic 'send' with 2 args if exist
                    try {
                        Method m2 = serverComm.getClass().getMethod("send", InputFrame.class, String.class);
                        m2.invoke(serverComm, InputFrame.newBuilder().build(), source);
                    } catch (NoSuchMethodException nsme2) {
                        Log.w(TAG, "ServerComm does not expose expected send methods; empty frame not sent.");
                    }
                }
            } catch (Exception e) {
                Log.w(TAG, "Failed to send empty frame", e);
            }
        });
    }

    /**
     * Shutdown wrapper: stop serverComm and shutdown executor.
     */
    public synchronized void shutdown() {
        stop();
        // Gracefully shutdown senderExecutor
        senderExecutor.shutdown();
        try {
            if (!senderExecutor.awaitTermination(800, TimeUnit.MILLISECONDS)) {
                senderExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            senderExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

    // ---------------- helpers ----------------

    private void tryInvokeNoArg(Object target, String methodName) {
        if (target == null) return;
        try {
            Method m = target.getClass().getMethod(methodName);
            if (m != null) {
                m.invoke(target);
                Log.i(TAG, "Invoked ServerComm." + methodName + "()");
            }
        } catch (NoSuchMethodException nsme) {
            // method does not exist - ignore
        } catch (Exception e) {
            Log.w(TAG, "Error invoking " + methodName + " on ServerComm", e);
        }
    }
}
