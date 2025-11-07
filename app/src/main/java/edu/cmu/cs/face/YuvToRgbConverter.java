package edu.cmu.cs.face;


import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;
import androidx.camera.core.ImageProxy;
import java.nio.ByteBuffer;


public class YuvToRgbConverter {
    private final RenderScript rs;
    private final ScriptIntrinsicYuvToRGB script;
    private Allocation allocationIn;
    private Allocation allocationOut;
    private int width = 0;
    private int height = 0;
    private byte[] yuvBuffer;

    public YuvToRgbConverter(Context context) {
        rs = RenderScript.create(context);
        script = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs));
    }

    public synchronized void yuvToRgb(ImageProxy image, Bitmap outputBitmap) {
        if (image.getFormat() != ImageFormat.YUV_420_888) {
            throw new IllegalArgumentException("Invalid image format");
        }

        int imageWidth = image.getWidth();
        int imageHeight = image.getHeight();

        // Ensure allocations are sized correctly
        if (allocationIn == null || imageWidth != width || imageHeight != height) {
            width = imageWidth;
            height = imageHeight;
            int yuvSize = width * height * 3 / 2;

            Type.Builder yuvType = new Type.Builder(rs, Element.U8(rs)).setX(yuvSize);
            allocationIn = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT);

            Type.Builder rgbaType = new Type.Builder(rs, Element.RGBA_8888(rs)).setX(width).setY(height);
            allocationOut = Allocation.createTyped(rs, rgbaType.create(), Allocation.USAGE_SCRIPT);

            script.setInput(allocationIn);
            yuvBuffer = new byte[yuvSize];
        }

        // Convert ImageProxy to a single NV21 byte array
        imageProxyToNv21(image.getPlanes(), width, height, yuvBuffer);

        // Run the RenderScript
        allocationIn.copyFrom(yuvBuffer);
        script.forEach(allocationOut);
        allocationOut.copyTo(outputBitmap);
    }

    // Converts YUV_420_888 ImageProxy planes to a single NV21 byte array.
    private void imageProxyToNv21(ImageProxy.PlaneProxy[] planes, int width, int height, byte[] output) {
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int yRowStride = planes[0].getRowStride();
        int uRowStride = planes[1].getRowStride();
        int vRowStride = planes[2].getRowStride();
        int uPixelStride = planes[1].getPixelStride();
        int vPixelStride = planes[2].getPixelStride();

        // Copy the Y plane.
        int yLen = width * height;
        yBuffer.get(output, 0, yLen);

        int uvOffset = yLen;
        // Interleaved VU plane for NV21 format.
        for (int row = 0; row < height / 2; row++) {
            for (int col = 0; col < width / 2; col++) {
                int vPos = row * vRowStride + col * vPixelStride;
                int uPos = row * uRowStride + col * uPixelStride;

                // V is first in NV21
                output[uvOffset++] = vBuffer.get(vPos);
                output[uvOffset++] = uBuffer.get(uPos);
            }
        }
    }
}
