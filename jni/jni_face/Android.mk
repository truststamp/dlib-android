LOCAL_PATH := $(call my-dir)

# =======================================================
LOCAL_MODULE := dlib_face_jni

LOCAL_SRC_FILES += \
           exceptuon.cpp \
           profiler.cpp \
           face.cpp

LOCAL_LDLIBS += -lm -llog -ldl -lz -ljnigraphics
LOCAL_CPPFLAGS += -fexceptions -frtti -std=c++11

# import dlib
LOCAL_SHARED_LIBRARIES := dlib

ifeq ($(TARGET_ARCH_ABI), armeabi-v7a)
    LOCAL_ARM_MODE := arm
    LOCAL_ARM_NEON := true
endif

include $(BUILD_SHARED_LIBRARY)
#-----------------------------------------------------------------------------
