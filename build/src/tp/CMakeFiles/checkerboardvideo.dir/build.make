# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_SOURCE_DIR = /home/hbakri/3A-SN/AR/TP1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hbakri/3A-SN/AR/TP1/build

# Include any dependencies generated for this target.
include src/tp/CMakeFiles/checkerboardvideo.dir/depend.make

# Include the progress variables for this target.
include src/tp/CMakeFiles/checkerboardvideo.dir/progress.make

# Include the compile flags for this target's objects.
include src/tp/CMakeFiles/checkerboardvideo.dir/flags.make

src/tp/CMakeFiles/checkerboardvideo.dir/checkerboardvideo.cpp.o: src/tp/CMakeFiles/checkerboardvideo.dir/flags.make
src/tp/CMakeFiles/checkerboardvideo.dir/checkerboardvideo.cpp.o: ../src/tp/checkerboardvideo.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hbakri/3A-SN/AR/TP1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/tp/CMakeFiles/checkerboardvideo.dir/checkerboardvideo.cpp.o"
	cd /home/hbakri/3A-SN/AR/TP1/build/src/tp && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/checkerboardvideo.dir/checkerboardvideo.cpp.o -c /home/hbakri/3A-SN/AR/TP1/src/tp/checkerboardvideo.cpp

src/tp/CMakeFiles/checkerboardvideo.dir/checkerboardvideo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/checkerboardvideo.dir/checkerboardvideo.cpp.i"
	cd /home/hbakri/3A-SN/AR/TP1/build/src/tp && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hbakri/3A-SN/AR/TP1/src/tp/checkerboardvideo.cpp > CMakeFiles/checkerboardvideo.dir/checkerboardvideo.cpp.i

src/tp/CMakeFiles/checkerboardvideo.dir/checkerboardvideo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/checkerboardvideo.dir/checkerboardvideo.cpp.s"
	cd /home/hbakri/3A-SN/AR/TP1/build/src/tp && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hbakri/3A-SN/AR/TP1/src/tp/checkerboardvideo.cpp -o CMakeFiles/checkerboardvideo.dir/checkerboardvideo.cpp.s

src/tp/CMakeFiles/checkerboardvideo.dir/checkerboardvideo.cpp.o.requires:

.PHONY : src/tp/CMakeFiles/checkerboardvideo.dir/checkerboardvideo.cpp.o.requires

src/tp/CMakeFiles/checkerboardvideo.dir/checkerboardvideo.cpp.o.provides: src/tp/CMakeFiles/checkerboardvideo.dir/checkerboardvideo.cpp.o.requires
	$(MAKE) -f src/tp/CMakeFiles/checkerboardvideo.dir/build.make src/tp/CMakeFiles/checkerboardvideo.dir/checkerboardvideo.cpp.o.provides.build
.PHONY : src/tp/CMakeFiles/checkerboardvideo.dir/checkerboardvideo.cpp.o.provides

src/tp/CMakeFiles/checkerboardvideo.dir/checkerboardvideo.cpp.o.provides.build: src/tp/CMakeFiles/checkerboardvideo.dir/checkerboardvideo.cpp.o


# Object files for target checkerboardvideo
checkerboardvideo_OBJECTS = \
"CMakeFiles/checkerboardvideo.dir/checkerboardvideo.cpp.o"

# External object files for target checkerboardvideo
checkerboardvideo_EXTERNAL_OBJECTS =

bin/checkerboardvideo: src/tp/CMakeFiles/checkerboardvideo.dir/checkerboardvideo.cpp.o
bin/checkerboardvideo: src/tp/CMakeFiles/checkerboardvideo.dir/build.make
bin/checkerboardvideo: lib/libtracker.a
bin/checkerboardvideo: /mnt/n7fs/ens/tp_gasparini/opencv2.4.13.4/lib/libopencv_videostab.so.2.4.13
bin/checkerboardvideo: /mnt/n7fs/ens/tp_gasparini/opencv2.4.13.4/lib/libopencv_superres.so.2.4.13
bin/checkerboardvideo: /mnt/n7fs/ens/tp_gasparini/opencv2.4.13.4/lib/libopencv_video.so.2.4.13
bin/checkerboardvideo: /mnt/n7fs/ens/tp_gasparini/opencv2.4.13.4/lib/libopencv_photo.so.2.4.13
bin/checkerboardvideo: /mnt/n7fs/ens/tp_gasparini/opencv2.4.13.4/lib/libopencv_objdetect.so.2.4.13
bin/checkerboardvideo: /mnt/n7fs/ens/tp_gasparini/opencv2.4.13.4/lib/libopencv_nonfree.so.2.4.13
bin/checkerboardvideo: /mnt/n7fs/ens/tp_gasparini/opencv2.4.13.4/lib/libopencv_calib3d.so.2.4.13
bin/checkerboardvideo: /mnt/n7fs/ens/tp_gasparini/opencv2.4.13.4/lib/libopencv_features2d.so.2.4.13
bin/checkerboardvideo: /mnt/n7fs/ens/tp_gasparini/opencv2.4.13.4/lib/libopencv_highgui.so.2.4.13
bin/checkerboardvideo: /mnt/n7fs/ens/tp_gasparini/opencv2.4.13.4/lib/libopencv_imgproc.so.2.4.13
bin/checkerboardvideo: /mnt/n7fs/ens/tp_gasparini/opencv2.4.13.4/lib/libopencv_flann.so.2.4.13
bin/checkerboardvideo: /mnt/n7fs/ens/tp_gasparini/opencv2.4.13.4/lib/libopencv_core.so.2.4.13
bin/checkerboardvideo: src/tp/CMakeFiles/checkerboardvideo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hbakri/3A-SN/AR/TP1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/checkerboardvideo"
	cd /home/hbakri/3A-SN/AR/TP1/build/src/tp && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/checkerboardvideo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/tp/CMakeFiles/checkerboardvideo.dir/build: bin/checkerboardvideo

.PHONY : src/tp/CMakeFiles/checkerboardvideo.dir/build

src/tp/CMakeFiles/checkerboardvideo.dir/requires: src/tp/CMakeFiles/checkerboardvideo.dir/checkerboardvideo.cpp.o.requires

.PHONY : src/tp/CMakeFiles/checkerboardvideo.dir/requires

src/tp/CMakeFiles/checkerboardvideo.dir/clean:
	cd /home/hbakri/3A-SN/AR/TP1/build/src/tp && $(CMAKE_COMMAND) -P CMakeFiles/checkerboardvideo.dir/cmake_clean.cmake
.PHONY : src/tp/CMakeFiles/checkerboardvideo.dir/clean

src/tp/CMakeFiles/checkerboardvideo.dir/depend:
	cd /home/hbakri/3A-SN/AR/TP1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hbakri/3A-SN/AR/TP1 /home/hbakri/3A-SN/AR/TP1/src/tp /home/hbakri/3A-SN/AR/TP1/build /home/hbakri/3A-SN/AR/TP1/build/src/tp /home/hbakri/3A-SN/AR/TP1/build/src/tp/CMakeFiles/checkerboardvideo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/tp/CMakeFiles/checkerboardvideo.dir/depend
