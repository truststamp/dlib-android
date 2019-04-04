/*
 * detector.h using google-style
 *
 *  Created on: May 24, 2016
 *      Author: Tzutalin
 *
 *  Copyright (c) 2016 Tzutalin. All rights reserved.
 */

#pragma once

#include <jni_common/jni_fileutils.h>
#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_loader/load_image.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/opencv/cv_image.h>
#include <glog/logging.h>
#include <jni.h>
#include <memory>
#include <stdio.h>
#include <android/log.h>

using namespace std;
using namespace dlib;

#define LOGI(...) \
((void)__android_log_print(ANDROID_LOG_INFO, "dlib-jni:", __VA_ARGS__))

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;


void throwException(JNIEnv* env,
                    const char* message) {
    jclass Exception = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(Exception, message);
}

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


class DLibHOGFaceDetector {
 private:
  string mLandMarkModel;
  string mRecognitionModel;
  shape_predictor msp;
  anet_type mr;
  unordered_map<int, full_object_detection> mFaceShapeMap;
  frontal_face_detector mFaceDetector;
  std::vector<rectangle> mRets;

  inline void init() {
    LOG(INFO) << "Init mFaceDetector";
    mFaceDetector = get_frontal_face_detector();
  }

 public:
  DLibHOGFaceDetector() { init(); }

  DLibHOGFaceDetector(const string& landmarkmodel, const string& recognitionmodel)
      : mLandMarkModel(landmarkmodel), mRecognitionModel(recognitionmodel) {
    init();
    if (!mLandMarkModel.empty() && jniutils::fileExists(mLandMarkModel)) {
      deserialize(mLandMarkModel) >> msp;
      LOG(INFO) << "Load landmarkmodel from " << mLandMarkModel;
    }
    if (mRecognitionModel.empty() && jniutils::fileExists(mRecognitionModel)) {
      deserialize(mRecognitionModel) >> mr;
      LOG(INFO) << "Load recognition model from " << mRecognitionModel;
    }
  }

  virtual inline int det(const string& path) {
    LOG(INFO) << "Read path from " << path;
    cv::Mat src_img = cv::imread(path, cv::IMREAD_COLOR);
    return det(src_img);
  }

  inline std::vector<rectangle> getResult() { return mRets; }

  // The format of mat should be BGR or Gray
  // If converting 4 channels to 3 channls because the format could be BGRA or
  // ARGB
  virtual inline int det(const cv::Mat& image) {
    if (image.empty())
      return 0;
    LOG(INFO) << "com_tzutalin_dlib_PeopleDet go to det(mat)";
    if (image.channels() == 1) {
      cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    }
    CHECK(image.channels() == 3);
    // TODO : Convert to gray image to speed up detection
    // It's unnecessary to use color image for face/landmark detection
    cv_image<bgr_pixel> img(image);
    mRets = mFaceDetector(img);
    LOG(INFO) << "Dlib HOG face det size : " << mRets.size();
    mFaceShapeMap.clear();
    // Process shape
    if (mRets.size() != 0 && mLandMarkModel.empty() == false) {
      for (unsigned long j = 0; j < mRets.size(); ++j) {
        full_object_detection shape = msp(img, mRets[j]);
        LOG(INFO) << "face index:" << j
                  << "number of parts: " << shape.num_parts();
        mFaceShapeMap[j] = shape;
      }
    }
    return mRets.size();
  }

  virtual inline std::vector<float> embed(JNIEnv* env, jobject bitmap) {
    LOG(INFO) << "start embedding";

    array2d<rgb_pixel> arr2d;
    convertBitmapToArray2d(env, bitmap, arr2d);
    matrix<rgb_pixel> img;
    img = mat(arr2d);

    LOG(INFO) << "preparation finished";

    mRets = mFaceDetector(img);
    LOG(INFO) << "face detected";

    if (mRets.size() != 0 && mLandMarkModel.empty() == false) {
        // Best score detection should be first
        full_object_detection shape = msp(img, mRets[0]);
        LOG(INFO) << "landmarks detected";
        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip);
        LOG(INFO) << "image chip extracted";
        matrix<float,0,1> face_descriptor = mr(face_chip);
        LOG(INFO) << "embedding finished";
        LOG(INFO) << face_descriptor;
        return std::vector<float>(face_descriptor.begin(), face_descriptor.end());
    }
    return std::vector<float>();
  }

  unordered_map<int, full_object_detection>& getFaceShapeMap() {
    return mFaceShapeMap;
  }
};

