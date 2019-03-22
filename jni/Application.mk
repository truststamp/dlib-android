NDK_TOOLCHAIN_VERSION := clang
APP_ABI := armeabi-v7a arm64-v8a 
APP_CPPFLAGS := -std=c++11 -frtti -fexceptions
APP_PLATFORM := android-28
APP_STL := c++_static
APP_CFLAGS+=-DDLIB_JPEG_SUPPORT=on
APP_CFLAGS+=-DDLIB_JPEG_STATIC=on
