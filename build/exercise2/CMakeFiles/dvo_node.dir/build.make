# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/cmake-gui

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /work/catkin_ws/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /work/catkin_ws/build

# Include any dependencies generated for this target.
include exercise2/CMakeFiles/dvo_node.dir/depend.make

# Include the progress variables for this target.
include exercise2/CMakeFiles/dvo_node.dir/progress.make

# Include the compile flags for this target's objects.
include exercise2/CMakeFiles/dvo_node.dir/flags.make

exercise2/CMakeFiles/dvo_node.dir/src/dvo_node.cpp.o: exercise2/CMakeFiles/dvo_node.dir/flags.make
exercise2/CMakeFiles/dvo_node.dir/src/dvo_node.cpp.o: /work/catkin_ws/src/exercise2/src/dvo_node.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /work/catkin_ws/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object exercise2/CMakeFiles/dvo_node.dir/src/dvo_node.cpp.o"
	cd /work/catkin_ws/build/exercise2 && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/dvo_node.dir/src/dvo_node.cpp.o -c /work/catkin_ws/src/exercise2/src/dvo_node.cpp

exercise2/CMakeFiles/dvo_node.dir/src/dvo_node.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dvo_node.dir/src/dvo_node.cpp.i"
	cd /work/catkin_ws/build/exercise2 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /work/catkin_ws/src/exercise2/src/dvo_node.cpp > CMakeFiles/dvo_node.dir/src/dvo_node.cpp.i

exercise2/CMakeFiles/dvo_node.dir/src/dvo_node.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dvo_node.dir/src/dvo_node.cpp.s"
	cd /work/catkin_ws/build/exercise2 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /work/catkin_ws/src/exercise2/src/dvo_node.cpp -o CMakeFiles/dvo_node.dir/src/dvo_node.cpp.s

exercise2/CMakeFiles/dvo_node.dir/src/dvo_node.cpp.o.requires:
.PHONY : exercise2/CMakeFiles/dvo_node.dir/src/dvo_node.cpp.o.requires

exercise2/CMakeFiles/dvo_node.dir/src/dvo_node.cpp.o.provides: exercise2/CMakeFiles/dvo_node.dir/src/dvo_node.cpp.o.requires
	$(MAKE) -f exercise2/CMakeFiles/dvo_node.dir/build.make exercise2/CMakeFiles/dvo_node.dir/src/dvo_node.cpp.o.provides.build
.PHONY : exercise2/CMakeFiles/dvo_node.dir/src/dvo_node.cpp.o.provides

exercise2/CMakeFiles/dvo_node.dir/src/dvo_node.cpp.o.provides.build: exercise2/CMakeFiles/dvo_node.dir/src/dvo_node.cpp.o

# Object files for target dvo_node
dvo_node_OBJECTS = \
"CMakeFiles/dvo_node.dir/src/dvo_node.cpp.o"

# External object files for target dvo_node
dvo_node_EXTERNAL_OBJECTS =

/work/catkin_ws/devel/lib/exercise2/dvo_node: exercise2/CMakeFiles/dvo_node.dir/src/dvo_node.cpp.o
/work/catkin_ws/devel/lib/exercise2/dvo_node: exercise2/CMakeFiles/dvo_node.dir/build.make
/work/catkin_ws/devel/lib/exercise2/dvo_node: /opt/ros/indigo/lib/libimage_geometry.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /opt/ros/indigo/lib/libcv_bridge.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.8
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.8
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.8
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.8
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.8
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.8
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
/work/catkin_ws/devel/lib/exercise2/dvo_node: /opt/ros/indigo/lib/libpcl_ros_filters.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /opt/ros/indigo/lib/libpcl_ros_io.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /opt/ros/indigo/lib/libpcl_ros_tf.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/libpcl_common.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/libpcl_kdtree.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/libpcl_octree.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/libpcl_search.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/libpcl_surface.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/libpcl_sample_consensus.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/libpcl_filters.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/libpcl_features.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/libpcl_segmentation.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/libpcl_io.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/libpcl_registration.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/libpcl_keypoints.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/libpcl_recognition.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/libpcl_visualization.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/libpcl_people.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/libpcl_outofcore.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/libpcl_tracking.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/libpcl_apps.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libqhull.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/libOpenNI.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/libvtkCommon.so.5.8.0
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/libvtkRendering.so.5.8.0
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/libvtkHybrid.so.5.8.0
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/libvtkCharts.so.5.8.0
/work/catkin_ws/devel/lib/exercise2/dvo_node: /opt/ros/indigo/lib/libdynamic_reconfigure_config_init_mutex.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /opt/ros/indigo/lib/libnodeletlib.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /opt/ros/indigo/lib/libbondcpp.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libuuid.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libtinyxml.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /opt/ros/indigo/lib/libclass_loader.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/libPocoFoundation.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libdl.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /opt/ros/indigo/lib/libroslib.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /opt/ros/indigo/lib/librosbag.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /opt/ros/indigo/lib/librosbag_storage.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /opt/ros/indigo/lib/libroslz4.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/liblz4.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /opt/ros/indigo/lib/libtopic_tools.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /opt/ros/indigo/lib/libtf.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /opt/ros/indigo/lib/libtf2_ros.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /opt/ros/indigo/lib/libactionlib.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /opt/ros/indigo/lib/libmessage_filters.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /opt/ros/indigo/lib/libroscpp.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libboost_signals.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /opt/ros/indigo/lib/libxmlrpcpp.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /opt/ros/indigo/lib/libtf2.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /opt/ros/indigo/lib/libroscpp_serialization.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /opt/ros/indigo/lib/librosconsole.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /opt/ros/indigo/lib/librosconsole_log4cxx.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /opt/ros/indigo/lib/librosconsole_backend_interface.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/liblog4cxx.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libboost_regex.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /opt/ros/indigo/lib/librostime.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /opt/ros/indigo/lib/libcpp_common.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libboost_system.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libboost_thread.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libpthread.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: /work/catkin_ws/devel/lib/libdvo.so
/work/catkin_ws/devel/lib/exercise2/dvo_node: exercise2/CMakeFiles/dvo_node.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable /work/catkin_ws/devel/lib/exercise2/dvo_node"
	cd /work/catkin_ws/build/exercise2 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dvo_node.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
exercise2/CMakeFiles/dvo_node.dir/build: /work/catkin_ws/devel/lib/exercise2/dvo_node
.PHONY : exercise2/CMakeFiles/dvo_node.dir/build

exercise2/CMakeFiles/dvo_node.dir/requires: exercise2/CMakeFiles/dvo_node.dir/src/dvo_node.cpp.o.requires
.PHONY : exercise2/CMakeFiles/dvo_node.dir/requires

exercise2/CMakeFiles/dvo_node.dir/clean:
	cd /work/catkin_ws/build/exercise2 && $(CMAKE_COMMAND) -P CMakeFiles/dvo_node.dir/cmake_clean.cmake
.PHONY : exercise2/CMakeFiles/dvo_node.dir/clean

exercise2/CMakeFiles/dvo_node.dir/depend:
	cd /work/catkin_ws/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /work/catkin_ws/src /work/catkin_ws/src/exercise2 /work/catkin_ws/build /work/catkin_ws/build/exercise2 /work/catkin_ws/build/exercise2/CMakeFiles/dvo_node.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : exercise2/CMakeFiles/dvo_node.dir/depend

