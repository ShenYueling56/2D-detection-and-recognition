# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /face/ArcSoft_ArcFace_Linux_x64_V3.0/samplecode/ASFTestDemo

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /face/ArcSoft_ArcFace_Linux_x64_V3.0/samplecode/ASFTestDemo/build

# Include any dependencies generated for this target.
include CMakeFiles/arcsoft_face_engine_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/arcsoft_face_engine_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/arcsoft_face_engine_test.dir/flags.make

CMakeFiles/arcsoft_face_engine_test.dir/samplecode.cpp.o: CMakeFiles/arcsoft_face_engine_test.dir/flags.make
CMakeFiles/arcsoft_face_engine_test.dir/samplecode.cpp.o: ../samplecode.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/face/ArcSoft_ArcFace_Linux_x64_V3.0/samplecode/ASFTestDemo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/arcsoft_face_engine_test.dir/samplecode.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/arcsoft_face_engine_test.dir/samplecode.cpp.o -c /face/ArcSoft_ArcFace_Linux_x64_V3.0/samplecode/ASFTestDemo/samplecode.cpp

CMakeFiles/arcsoft_face_engine_test.dir/samplecode.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/arcsoft_face_engine_test.dir/samplecode.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /face/ArcSoft_ArcFace_Linux_x64_V3.0/samplecode/ASFTestDemo/samplecode.cpp > CMakeFiles/arcsoft_face_engine_test.dir/samplecode.cpp.i

CMakeFiles/arcsoft_face_engine_test.dir/samplecode.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/arcsoft_face_engine_test.dir/samplecode.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /face/ArcSoft_ArcFace_Linux_x64_V3.0/samplecode/ASFTestDemo/samplecode.cpp -o CMakeFiles/arcsoft_face_engine_test.dir/samplecode.cpp.s

CMakeFiles/arcsoft_face_engine_test.dir/samplecode.cpp.o.requires:

.PHONY : CMakeFiles/arcsoft_face_engine_test.dir/samplecode.cpp.o.requires

CMakeFiles/arcsoft_face_engine_test.dir/samplecode.cpp.o.provides: CMakeFiles/arcsoft_face_engine_test.dir/samplecode.cpp.o.requires
	$(MAKE) -f CMakeFiles/arcsoft_face_engine_test.dir/build.make CMakeFiles/arcsoft_face_engine_test.dir/samplecode.cpp.o.provides.build
.PHONY : CMakeFiles/arcsoft_face_engine_test.dir/samplecode.cpp.o.provides

CMakeFiles/arcsoft_face_engine_test.dir/samplecode.cpp.o.provides.build: CMakeFiles/arcsoft_face_engine_test.dir/samplecode.cpp.o


# Object files for target arcsoft_face_engine_test
arcsoft_face_engine_test_OBJECTS = \
"CMakeFiles/arcsoft_face_engine_test.dir/samplecode.cpp.o"

# External object files for target arcsoft_face_engine_test
arcsoft_face_engine_test_EXTERNAL_OBJECTS =

arcsoft_face_engine_test: CMakeFiles/arcsoft_face_engine_test.dir/samplecode.cpp.o
arcsoft_face_engine_test: CMakeFiles/arcsoft_face_engine_test.dir/build.make
arcsoft_face_engine_test: /usr/local/lib/libopencv_calib3d.a
arcsoft_face_engine_test: /usr/local/lib/libopencv_photo.a
arcsoft_face_engine_test: /usr/local/lib/libopencv_features2d.a
arcsoft_face_engine_test: /usr/local/lib/libopencv_highgui.a
arcsoft_face_engine_test: /usr/local/lib/libopencv_imgproc.a
arcsoft_face_engine_test: /usr/local/lib/libopencv_videoio.a
arcsoft_face_engine_test: /usr/local/lib/libopencv_videostab.a
arcsoft_face_engine_test: /usr/local/lib/libopencv_shape.a
arcsoft_face_engine_test: /usr/local/lib/libopencv_objdetect.a
arcsoft_face_engine_test: /usr/local/lib/libopencv_flann.a
arcsoft_face_engine_test: /usr/local/lib/libopencv_core.a
arcsoft_face_engine_test: /usr/local/lib/libopencv_video.a
arcsoft_face_engine_test: /usr/local/lib/libopencv_ml.a
arcsoft_face_engine_test: /usr/local/lib/libopencv_imgcodecs.a
arcsoft_face_engine_test: /usr/local/lib/libopencv_superres.a
arcsoft_face_engine_test: /usr/local/lib/libopencv_stitching.a
arcsoft_face_engine_test: /usr/local/lib/libopencv_dnn.a
arcsoft_face_engine_test: /usr/local/lib/libopencv_photo.a
arcsoft_face_engine_test: /usr/local/share/OpenCV/3rdparty/lib/libquirc.a
arcsoft_face_engine_test: /usr/local/lib/libopencv_video.a
arcsoft_face_engine_test: /usr/local/lib/libopencv_calib3d.a
arcsoft_face_engine_test: /usr/local/lib/libopencv_features2d.a
arcsoft_face_engine_test: /usr/local/lib/libopencv_highgui.a
arcsoft_face_engine_test: /usr/local/lib/libopencv_videoio.a
arcsoft_face_engine_test: /usr/local/lib/libopencv_flann.a
arcsoft_face_engine_test: /usr/local/lib/libopencv_imgcodecs.a
arcsoft_face_engine_test: /usr/local/share/OpenCV/3rdparty/lib/liblibwebp.a
arcsoft_face_engine_test: /home/xuzz27/anaconda3/envs/qdtrack/lib/libjpeg.so
arcsoft_face_engine_test: /home/xuzz27/anaconda3/envs/qdtrack/lib/libpng.so
arcsoft_face_engine_test: /home/xuzz27/anaconda3/envs/qdtrack/lib/libtiff.so
arcsoft_face_engine_test: /usr/lib/x86_64-linux-gnu/libjasper.so
arcsoft_face_engine_test: /home/xuzz27/anaconda3/envs/qdtrack/lib/libjpeg.so
arcsoft_face_engine_test: /home/xuzz27/anaconda3/envs/qdtrack/lib/libpng.so
arcsoft_face_engine_test: /home/xuzz27/anaconda3/envs/qdtrack/lib/libtiff.so
arcsoft_face_engine_test: /usr/lib/x86_64-linux-gnu/libjasper.so
arcsoft_face_engine_test: /usr/lib/x86_64-linux-gnu/libImath.so
arcsoft_face_engine_test: /usr/lib/x86_64-linux-gnu/libIlmImf.so
arcsoft_face_engine_test: /usr/lib/x86_64-linux-gnu/libIex.so
arcsoft_face_engine_test: /usr/lib/x86_64-linux-gnu/libHalf.so
arcsoft_face_engine_test: /usr/lib/x86_64-linux-gnu/libIlmThread.so
arcsoft_face_engine_test: /usr/local/lib/libopencv_imgproc.a
arcsoft_face_engine_test: /usr/local/lib/libopencv_core.a
arcsoft_face_engine_test: /home/xuzz27/anaconda3/envs/qdtrack/lib/libz.so
arcsoft_face_engine_test: /usr/local/share/OpenCV/3rdparty/lib/libittnotify.a
arcsoft_face_engine_test: /usr/local/share/OpenCV/3rdparty/lib/libippiw.a
arcsoft_face_engine_test: /usr/local/share/OpenCV/3rdparty/lib/libippicv.a
arcsoft_face_engine_test: /usr/local/share/OpenCV/3rdparty/lib/liblibprotobuf.a
arcsoft_face_engine_test: CMakeFiles/arcsoft_face_engine_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/face/ArcSoft_ArcFace_Linux_x64_V3.0/samplecode/ASFTestDemo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable arcsoft_face_engine_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/arcsoft_face_engine_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/arcsoft_face_engine_test.dir/build: arcsoft_face_engine_test

.PHONY : CMakeFiles/arcsoft_face_engine_test.dir/build

CMakeFiles/arcsoft_face_engine_test.dir/requires: CMakeFiles/arcsoft_face_engine_test.dir/samplecode.cpp.o.requires

.PHONY : CMakeFiles/arcsoft_face_engine_test.dir/requires

CMakeFiles/arcsoft_face_engine_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/arcsoft_face_engine_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/arcsoft_face_engine_test.dir/clean

CMakeFiles/arcsoft_face_engine_test.dir/depend:
	cd /face/ArcSoft_ArcFace_Linux_x64_V3.0/samplecode/ASFTestDemo/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /face/ArcSoft_ArcFace_Linux_x64_V3.0/samplecode/ASFTestDemo /face/ArcSoft_ArcFace_Linux_x64_V3.0/samplecode/ASFTestDemo /face/ArcSoft_ArcFace_Linux_x64_V3.0/samplecode/ASFTestDemo/build /face/ArcSoft_ArcFace_Linux_x64_V3.0/samplecode/ASFTestDemo/build /face/ArcSoft_ArcFace_Linux_x64_V3.0/samplecode/ASFTestDemo/build/CMakeFiles/arcsoft_face_engine_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/arcsoft_face_engine_test.dir/depend

