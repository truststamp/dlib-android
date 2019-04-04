LOCAL_PATH := $(call my-dir)
SUB_MK_FILES := $(call all-subdir-makefiles)

## Build dlib to static library
include $(CLEAR_VARS)
LOCAL_MODULE := dlib
LOCAL_C_INCLUDES := $(LOCAL_PATH)/../dlib
LOCAL_SRC_FILES += \
                ../$(LOCAL_PATH)/../dlib/dlib/threads/threads_kernel_shared.cpp \
                ../$(LOCAL_PATH)/../dlib/dlib/entropy_decoder/entropy_decoder_kernel_2.cpp \
                ../$(LOCAL_PATH)/../dlib/dlib/base64/base64_kernel_1.cpp \
                ../$(LOCAL_PATH)/../dlib/dlib/threads/threads_kernel_1.cpp \
                ../$(LOCAL_PATH)/../dlib/dlib/threads/threads_kernel_2.cpp \
                ../$(LOCAL_PATH)/../dlib/dlib/threads/thread_pool_extension.cpp \
                ../$(LOCAL_PATH)/../dlib/dlib/threads/async.cpp \
                ../$(LOCAL_PATH)/../dlib/dlib/cuda/cpu_dlib.cpp \
                ../$(LOCAL_PATH)/../dlib/dlib/cuda/tensor_tools.cpp
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_C_INCLUDES)
LOCAL_CFLAGS += -O3 -DNDEBUG -DDLIB_NO_GUI_SUPPORT=on

ifeq ($(TARGET_ARCH_ABI), armeabi-v7a)
    LOCAL_ARM_MODE := arm
	LOCAL_ARM_NEON := true
endif

include $(BUILD_SHARED_LIBRARY)

include $(SUB_MK_FILES)
