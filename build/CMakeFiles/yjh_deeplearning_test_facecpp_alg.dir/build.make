# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /wangjiadong/dlalg_new

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /wangjiadong/dlalg_new/build

# Include any dependencies generated for this target.
include CMakeFiles/yjh_deeplearning_test_facecpp_alg.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/yjh_deeplearning_test_facecpp_alg.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/yjh_deeplearning_test_facecpp_alg.dir/flags.make

CMakeFiles/yjh_deeplearning_test_facecpp_alg.dir/examples/test_facecpp_alg.cpp.o: CMakeFiles/yjh_deeplearning_test_facecpp_alg.dir/flags.make
CMakeFiles/yjh_deeplearning_test_facecpp_alg.dir/examples/test_facecpp_alg.cpp.o: ../examples/test_facecpp_alg.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/wangjiadong/dlalg_new/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/yjh_deeplearning_test_facecpp_alg.dir/examples/test_facecpp_alg.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yjh_deeplearning_test_facecpp_alg.dir/examples/test_facecpp_alg.cpp.o -c /wangjiadong/dlalg_new/examples/test_facecpp_alg.cpp

CMakeFiles/yjh_deeplearning_test_facecpp_alg.dir/examples/test_facecpp_alg.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yjh_deeplearning_test_facecpp_alg.dir/examples/test_facecpp_alg.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /wangjiadong/dlalg_new/examples/test_facecpp_alg.cpp > CMakeFiles/yjh_deeplearning_test_facecpp_alg.dir/examples/test_facecpp_alg.cpp.i

CMakeFiles/yjh_deeplearning_test_facecpp_alg.dir/examples/test_facecpp_alg.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yjh_deeplearning_test_facecpp_alg.dir/examples/test_facecpp_alg.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /wangjiadong/dlalg_new/examples/test_facecpp_alg.cpp -o CMakeFiles/yjh_deeplearning_test_facecpp_alg.dir/examples/test_facecpp_alg.cpp.s

# Object files for target yjh_deeplearning_test_facecpp_alg
yjh_deeplearning_test_facecpp_alg_OBJECTS = \
"CMakeFiles/yjh_deeplearning_test_facecpp_alg.dir/examples/test_facecpp_alg.cpp.o"

# External object files for target yjh_deeplearning_test_facecpp_alg
yjh_deeplearning_test_facecpp_alg_EXTERNAL_OBJECTS =

yjh_deeplearning_test_facecpp_alg: CMakeFiles/yjh_deeplearning_test_facecpp_alg.dir/examples/test_facecpp_alg.cpp.o
yjh_deeplearning_test_facecpp_alg: CMakeFiles/yjh_deeplearning_test_facecpp_alg.dir/build.make
yjh_deeplearning_test_facecpp_alg: ../lib/libyjh_deeplearning.so
yjh_deeplearning_test_facecpp_alg: /usr/local/lib/libopencv_gapi.so.4.5.5
yjh_deeplearning_test_facecpp_alg: /usr/local/lib/libopencv_highgui.so.4.5.5
yjh_deeplearning_test_facecpp_alg: /usr/local/lib/libopencv_ml.so.4.5.5
yjh_deeplearning_test_facecpp_alg: /usr/local/lib/libopencv_objdetect.so.4.5.5
yjh_deeplearning_test_facecpp_alg: /usr/local/lib/libopencv_photo.so.4.5.5
yjh_deeplearning_test_facecpp_alg: /usr/local/lib/libopencv_stitching.so.4.5.5
yjh_deeplearning_test_facecpp_alg: /usr/local/lib/libopencv_video.so.4.5.5
yjh_deeplearning_test_facecpp_alg: /usr/local/lib/libopencv_videoio.so.4.5.5
yjh_deeplearning_test_facecpp_alg: /usr/local/lib/libopencv_calib3d.so.4.5.5
yjh_deeplearning_test_facecpp_alg: /usr/local/lib/libopencv_dnn.so.4.5.5
yjh_deeplearning_test_facecpp_alg: /usr/local/lib/libopencv_features2d.so.4.5.5
yjh_deeplearning_test_facecpp_alg: /usr/local/lib/libopencv_flann.so.4.5.5
yjh_deeplearning_test_facecpp_alg: /usr/local/lib/libopencv_imgcodecs.so.4.5.5
yjh_deeplearning_test_facecpp_alg: /usr/local/lib/libopencv_imgproc.so.4.5.5
yjh_deeplearning_test_facecpp_alg: /usr/local/lib/libopencv_core.so.4.5.5
yjh_deeplearning_test_facecpp_alg: CMakeFiles/yjh_deeplearning_test_facecpp_alg.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/wangjiadong/dlalg_new/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable yjh_deeplearning_test_facecpp_alg"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/yjh_deeplearning_test_facecpp_alg.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/yjh_deeplearning_test_facecpp_alg.dir/build: yjh_deeplearning_test_facecpp_alg

.PHONY : CMakeFiles/yjh_deeplearning_test_facecpp_alg.dir/build

CMakeFiles/yjh_deeplearning_test_facecpp_alg.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/yjh_deeplearning_test_facecpp_alg.dir/cmake_clean.cmake
.PHONY : CMakeFiles/yjh_deeplearning_test_facecpp_alg.dir/clean

CMakeFiles/yjh_deeplearning_test_facecpp_alg.dir/depend:
	cd /wangjiadong/dlalg_new/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /wangjiadong/dlalg_new /wangjiadong/dlalg_new /wangjiadong/dlalg_new/build /wangjiadong/dlalg_new/build /wangjiadong/dlalg_new/build/CMakeFiles/yjh_deeplearning_test_facecpp_alg.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/yjh_deeplearning_test_facecpp_alg.dir/depend

