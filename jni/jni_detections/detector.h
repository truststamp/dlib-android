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

using namespace std;
using namespace dlib;

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

class DLibHOGDetector {
 private:
  typedef scan_fhog_pyramid<pyramid_down<6>> image_scanner_type;
  object_detector<image_scanner_type> mObjectDetector;

  inline void init() {
    LOG(INFO) << "Model Path: " << mModelPath;
    if (jniutils::fileExists(mModelPath)) {
      deserialize(mModelPath) >> mObjectDetector;
    } else {
      LOG(INFO) << "Not exist " << mModelPath;
    }
  }

 public:
  DLibHOGDetector(const string& modelPath = "/sdcard/person.svm")
      : mModelPath(modelPath) {
    init();
  }

  virtual inline int det(const string& path) {
    using namespace jniutils;
    if (!fileExists(mModelPath) || !fileExists(path)) {
      LOG(WARNING) << "No modle path or input file path";
      return 0;
    }
    cv::Mat src_img = cv::imread(path, cv::IMREAD_COLOR);
    if (src_img.empty())
      return 0;
    int img_width = src_img.cols;
    int img_height = src_img.rows;
    int im_size_min = MIN(img_width, img_height);
    int im_size_max = MAX(img_width, img_height);

    float scale = float(INPUT_IMG_MIN_SIZE) / float(im_size_min);
    if (scale * im_size_max > INPUT_IMG_MAX_SIZE) {
      scale = (float)INPUT_IMG_MAX_SIZE / (float)im_size_max;
    }

    if (scale != 1.0) {
      cv::Mat outputMat;
      cv::resize(src_img, outputMat,
                 cv::Size(img_width * scale, img_height * scale));
      src_img = outputMat;
    }

    // cv::resize(src_img, src_img, cv::Size(320, 240));
    cv_image<bgr_pixel> cimg(src_img);

    double thresh = 0.5;
    mRets = mObjectDetector(cimg, thresh);
    return mRets.size();
  }

  inline std::vector<rectangle> getResult() { return mRets; }

  virtual ~DLibHOGDetector() {}

 protected:
  std::vector<rectangle> mRets;
  string mModelPath;
  const int INPUT_IMG_MAX_SIZE = 800;
  const int INPUT_IMG_MIN_SIZE = 600;
};

/*
 * DLib face detect and face feature extractor
 */
class DLibHOGFaceDetector : public DLibHOGDetector {
 private:
  string mLandMarkModel;
  string mRecognitionModel;
  shape_predictor msp;
  anet_type mr;
  unordered_map<int, full_object_detection> mFaceShapeMap;
  frontal_face_detector mFaceDetector;

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

  virtual inline std::vector<float> embed(const cv::Mat& image) {
    LOG(INFO) << "start embedding";

    if (image.empty())
      return std::vector<float>();

    if (image.channels() == 1) {
      cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    }
    CHECK(image.channels() == 3);

    cv_image<bgr_pixel> img(image);
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
        return std::vector<float>(face_descriptor.begin(), face_descriptor.end());
    }
    return std::vector<float>();
  }

  unordered_map<int, full_object_detection>& getFaceShapeMap() {
    return mFaceShapeMap;
  }
};

