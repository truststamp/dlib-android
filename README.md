# dlib-android

[![Build Status](https://travis-ci.org/tzutalin/dlib-android.png)](https://travis-ci.org/tzutalin/dlib-android)

### Purpose
* Port [dlib](http://dlib.net/) to Android platform

* You can build it to dynamic or static library for Android. You can also build dlib's sample to Android executable file.

* You can refer to [dlib-android-app](https://github.com/tzutalin/dlib-android-app) which demonstrates dlib-android features

### Grab the source

    $ git clone --recursive https://github.com/tzutalin/dlib-android.git
    $ cd dlib-android
    $ ./envsetup

### Prerequisites
* Download OpenCV android sdk https://sourceforge.net/projects/opencvlibrary/files/latest/download. Put contents of sdk `native` folder to `third_party/opencv`

* Download Android-NDK from [Android website](https://developer.android.com/ndk/downloads/index.html).

	 After downloading, go to the directory to which you downloaded the package to extract it

	 Export ANDROID_NDK_HOME in ~/.bashrc
     `$ vim ~/.bashrc`

	`export ANDROID_NDK_HOME=[NDK_PATH]/android-ndk-[version]`

    `export PATH=$PATH:$ANDROID_NDK_HOME`

* Install Android Debug Bride (ADB). You can download it via [Android SDK Manager](https://developer.android.com/sdk/installing/index.html) or $ sudo apt-get install android-tools-adb

* Prepare an Android device for test

### Build JNI code and shared library for Android application
* You can change the compiler architecture in dlib-android/jni/Application.mk

* The way to build the shared library for Android application

```sh
    $ cd [dlib-android]
    $ python build.py
```

Alternative way to build native code and copy to the Android Studio's project manually:
```sh
    $ cd [dlib-android]
    $ ndk-build -j 2
    $ cp -r libs/* androidstudio-examples/dlib-android-app/dlib/src/main/jniLibs
```

Build dlib manually for arm in `Release` mode
```
Build for armh
cmake -DBUILD_SHARED_LIBS=1 -DDLIB_NO_GUI_SUPPORT=on -DCMAKE_C_FLAGS="-O3 -mfpu=neon -fprofile-use -DENABLE_NEON" -DCMAKE_C_COMPILER=/usr/bin/arm-linux-gnueabihf-gcc -DCMAKE_CXX_COMPILER=/usr/bin/arm-linux-gnueabihf-g++ -DCMAKE_CXX_FLAGS="-std=c++11" --config Release --build ../dlib

Build for arm64
cmake -DBUILD_SHARED_LIBS=1 -DDLIB_NO_GUI_SUPPORT=on -DCMAKE_C_FLAGS="-O3 -fprofile-use -DENABLE_NEON" -DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++ -DCMAKE_CXX_FLAGS="-std=c++11" --config Release --build ../dlib
```

### Run Android application
* Open Android Studio's projects in androidstudio-examples/dlib-android-app to run face detection, face landmark, and so on

### Folder structure

```
├── data                    # Test data or the models for detection and landmarks
├── dlib                    # Source files of dlib. It is a submodule
├── jni                     # Source files of JNI codes and their make files
├── androidstudio-examples  # Android Studio's projects use the shared library built by this repo
├── tools                   # Tools and utilities
├── third_party             # Like OpenCV and [miniglog](https://github.com/tzutalin/miniglog)
├── CMakeLists.txt          # Use CMake to build instead of using Android.mk
├── LICENSE
└── README.md
```

### Do you want to contribute
 * If you have any improvement or you've found any bug, send a pull request with the code.
 * Give me a star on this repository
 * <a href='https://ko-fi.com/A4263TV2' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://az743702.vo.msecnd.net/cdn/kofi1.png?v=0' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>

### Future tasks
* Add more examples to [dlib-android-app](https://github.com/tzutalin/dlib-android-app)

