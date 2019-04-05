#include <jni.h>
#include <android/log.h>
#include <android/bitmap.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <exception.h>
#include <profiler.h>
#include <recognition.h>


#define JAVA_NULL 0

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
anet_type sFaceEmbedder;


extern "C" JNIEXPORT void JNICALL
JNI_METHOD(prepare)(JNIEnv *env,
                     jobject thiz,
                     jstring landmarkModel,
                     jstring recognitionModel) {
    const char *landmarkPath = env->GetStringUTFChars(landmarkModel, JNI_FALSE);
    const char *recognitionPath = env->GetStringUTFChars(recognitionModel, JNI_FALSE);

    // Profiler.
    Profiler profiler;
    profiler.start();

    sFaceDetector = dlib::get_frontal_face_detector();
    dlib::deserialize(landmarkPath) >> sFaceLandmarksDetector;
    dlib::deserialize(recognitionPath) >> sFaceEmbedder;

    double interval = profiler.stopAndGetInterval();

    LOGI("L%d: models are initialized (took %.3f ms)", __LINE__, interval);

    env->ReleaseStringUTFChars(landmarkModel, landmarkPath);
    env->ReleaseStringUTFChars(recognitionModel, recognitionPath);
}

extern "C" JNIEXPORT jfloatArray JNICALL
JNI_METHOD(embedFace)(JNIEnv *env,
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

    if (dets.size() > 0) {
        dlib::rectangle best = dets[0];
        full_object_detection shape = sFaceLandmarksDetector(img, dets[0]);

        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip);

        matrix<float,0,1> face_descriptor = sFaceEmbedder(face_chip);
        std::vector<float> fdVec(face_descriptor.begin(), face_descriptor.end());

        int size = 128;
        jfloat hashArray[size];
        for (int i = 0; i < size; i++) {
            hashArray[i] = fdVec[i];
        }

        jfloatArray result;
        result = env->NewFloatArray(size);
        env->SetFloatArrayRegion(result, 0, size, hashArray);
        return result;
    }

    return JAVA_NULL;
}
