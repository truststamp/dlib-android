// Copyright (c) 2017-present boyw165
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
//    The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <jni.h>
#include <android/log.h>
#include <android/bitmap.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <exception.h>
#include <profiler.h>

#define LOGI(...) \
  ((void)__android_log_print(ANDROID_LOG_INFO, "dlib-jni:", __VA_ARGS__))

#define JNI_METHOD(NAME) \
    Java_net_truststamp_dlib_Face_##NAME


// FIXME: Create a class inheriting from dlib::array2d<dlib::rgb_pixel>.
void convertBitmapToArray2d(JNIEnv* env,
                            jobject bitmap,
                            dlib::array2d<dlib::rgb_pixel>& out) {
    AndroidBitmapInfo bitmapInfo;
    void* pixels;
    int state;

    if (0 > (state = AndroidBitmap_getInfo(env, bitmap, &bitmapInfo))) {
        LOGI("L%d: AndroidBitmap_getInfo() failed! error=%d", __LINE__, state);
        throwException(env, "AndroidBitmap_getInfo() failed!");
        return;
    } else if (bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        LOGI("L%d: Bitmap format is not RGB_565!", __LINE__);
        throwException(env, "Bitmap format is not RGB_565!");
    }

    // Lock the bitmap for copying the pixels safely.
    if (0 > (state = AndroidBitmap_lockPixels(env, bitmap, &pixels))) {
        LOGI("L%d: AndroidBitmap_lockPixels() failed! error=%d", __LINE__, state);
        throwException(env, "AndroidBitmap_lockPixels() failed!");
        return;
    }

    LOGI("L%d: info.width=%d, info.height=%d", __LINE__, bitmapInfo.width, bitmapInfo.height);
    out.set_size((long) bitmapInfo.height, (long) bitmapInfo.width);

    char* line = (char*) pixels;
    for (int h = 0; h < bitmapInfo.height; ++h) {
        for (int w = 0; w < bitmapInfo.width; ++w) {
            uint32_t* color = (uint32_t*) (line + 4 * w);

            out[h][w].red = (unsigned char) (0xFF & ((*color) >> 24));
            out[h][w].green = (unsigned char) (0xFF & ((*color) >> 16));
            out[h][w].blue = (unsigned char) (0xFF & ((*color) >> 8));
        }

        line = line + bitmapInfo.stride;
    }

    // Unlock the bitmap.
    AndroidBitmap_unlockPixels(env, bitmap);
}

// JNI ////////////////////////////////////////////////////////////////////////

dlib::shape_predictor sFaceLandmarksDetector;
dlib::frontal_face_detector sFaceDetector;


extern "C" JNIEXPORT void JNICALL
JNI_METHOD(prepareFaceDetector)(JNIEnv *env,
                                jobject thiz) {
    // Profiler.
    Profiler profiler;
    profiler.start();

    // Prepare the detector.
    sFaceDetector = dlib::get_frontal_face_detector();

    double interval = profiler.stopAndGetInterval();

    LOGI("L%d: sFaceDetector is initialized (took %.3f ms)", __LINE__, interval);
    LOGI("L%d: sFaceDetector.num_detectors()=%lu", __LINE__, sFaceDetector.num_detectors());
}

extern "C" JNIEXPORT void JNICALL
JNI_METHOD(prepareFaceLandmarksDetector)(JNIEnv *env,
                                         jobject thiz,
                                         jstring detectorPath) {
    const char *path = env->GetStringUTFChars(detectorPath, JNI_FALSE);

    // Profiler.
    Profiler profiler;
    profiler.start();

    // We need a shape_predictor. This is the tool that will predict face
    // landmark positions given an image and face bounding box.  Here we are just
    // loading the model from the shape_predictor_68_face_landmarks.dat file you gave
    // as a command line argument.
    // Deserialize the shape detector.
    dlib::deserialize(path) >> sFaceLandmarksDetector;

    double interval = profiler.stopAndGetInterval();

    LOGI("L%d: sFaceLandmarksDetector is initialized (took %.3f ms)", __LINE__, interval);
    LOGI("L%d: sFaceLandmarksDetector.num_parts()=%lu", __LINE__, sFaceLandmarksDetector.num_parts());

    env->ReleaseStringUTFChars(detectorPath, path);

    if (sFaceLandmarksDetector.num_parts() != 68) {
        throwException(env, "It's not a 68 landmarks detector!");
    }
}

extern "C" JNIEXPORT jbyteArray JNICALL
JNI_METHOD(detectFace)(JNIEnv *env,
                        jobject thiz,
                        jobject bitmap) {
    // Profiler.
    Profiler profiler;
    profiler.start();

    // Convert bitmap to dlib::array2d.
    dlib::array2d<dlib::rgb_pixel> img;
    convertBitmapToArray2d(env, bitmap, img);

    double interval = profiler.stopAndGetInterval();

    const long width = img.nc();
    const long height = img.nr();
    LOGI("L%d: input image (w=%ld, h=%ld) is read (took %.3f ms)",
         __LINE__, width, height, interval);

    profiler.start();

    // Now tell the face detector to give us a list of bounding boxes
    // around all the faces in the image.
    std::vector<dlib::rectangle> dets = sFaceDetector(img);
    interval = profiler.stopAndGetInterval();
    LOGI("L%d: Number of faces detected: %u (took %.3f ms)",
         __LINE__, (unsigned int) dets.size(), interval);

    // To protobuf message.
    FaceList faces;
    for (unsigned long i = 0; i < dets.size(); ++i) {
        // Profiler.
        profiler.start();

        dlib::rectangle& det = dets.at(i);

        Face* face = faces.add_faces();
        RectF* bound = face->mutable_bound();

        bound->set_left((float) det.left() / width);
        bound->set_top((float) det.top() / height);
        bound->set_right((float) det.right() / width);
        bound->set_bottom((float) det.bottom() / height);

        interval = profiler.stopAndGetInterval();
        LOGI("L%d: Convert face #%lu to protobuf message (took %.3f ms)",
             __LINE__, i, interval);
    }

    // Profiler.
    profiler.start();

    // Prepare the return message.
    int outSize = faces.ByteSize();
    jbyteArray out = env->NewByteArray(outSize);
    jbyte* buffer = new jbyte[outSize];

    faces.SerializeToArray(buffer, outSize);
    env->SetByteArrayRegion(out, 0, outSize, buffer);
    delete[] buffer;

    interval = profiler.stopAndGetInterval();
    LOGI("L%d: Convert faces to protobuf message (took %.3f ms)",
         __LINE__, interval);

    return out;
}

extern "C" JNIEXPORT jbyteArray JNICALL
JNI_METHOD(detectLandmarksFromFace)(JNIEnv *env,
                                    jobject thiz,
                                    jobject bitmap,
                                    jlong left,
                                    jlong top,
                                    jlong right,
                                    jlong bottom) {
    // Profiler.
    Profiler profiler;
    profiler.start();

    // Convert bitmap to dlib::array2d.
    dlib::array2d<dlib::rgb_pixel> img;
    convertBitmapToArray2d(env, bitmap, img);

    double interval = profiler.stopAndGetInterval();

    const long width = img.nc();
    const long height = img.nr();
    LOGI("L%d: input image (w=%ld, h=%ld) is read (took %.3f ms)",
         __LINE__, width, height, interval);

    profiler.start();

    // Detect landmarks.
    dlib::rectangle bound(left, top, right, bottom);
    dlib::full_object_detection shape = sFaceLandmarksDetector(img, bound);
    interval = profiler.stopAndGetInterval();
    LOGI("L%d: %lu landmarks detected (took %.3f ms)",
         __LINE__, shape.num_parts(), interval);

    profiler.start();
    // Protobuf message.
    LandmarkList landmarks;
    // You get the idea, you can get all the face part locations if
    // you want them.  Here we just store them in shapes so we can
    // put them on the screen.
    for (unsigned long i = 0 ; i < shape.num_parts(); ++i) {
        dlib::point& pt = shape.part(i);

        Landmark* landmark = landmarks.add_landmarks();
        landmark->set_x((float) pt.x() / width);
        landmark->set_y((float) pt.y() / height);
    }
    interval = profiler.stopAndGetInterval();
    LOGI("L%d: Convert #%lu landmarks to protobuf message (took %.3f ms)",
         __LINE__, shape.num_parts(), interval);

    // Profiler.
    profiler.start();

    // TODO: Make a JNI function to convert a message to byte[] living in
    // TODO: lib-protobuf project.
    // Prepare the return message.
    int outSize = landmarks.ByteSize();
    jbyteArray out = env->NewByteArray(outSize);
    jbyte* buffer = new jbyte[outSize];

    landmarks.SerializeToArray(buffer, outSize);
    env->SetByteArrayRegion(out, 0, outSize, buffer);
    delete[] buffer;

    interval = profiler.stopAndGetInterval();
    LOGI("L%d: Convert faces to protobuf message (took %.3f ms)",
         __LINE__, interval);

    return out;
}
