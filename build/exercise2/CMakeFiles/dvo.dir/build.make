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
include exercise2/CMakeFiles/dvo.dir/depend.make

# Include the progress variables for this target.
include exercise2/CMakeFiles/dvo.dir/progress.make

# Include the compile flags for this target's objects.
include exercise2/CMakeFiles/dvo.dir/flags.make

exercise2/CMakeFiles/dvo.dir/src/dvo.cpp.o: exercise2/CMakeFiles/dvo.dir/flags.make
exercise2/CMakeFiles/dvo.dir/src/dvo.cpp.o: /work/catkin_ws/src/exercise2/src/dvo.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /work/catkin_ws/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object exercise2/CMakeFiles/dvo.dir/src/dvo.cpp.o"
	cd /work/catkin_ws/build/exercise2 && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/dvo.dir/src/dvo.cpp.o -c /work/catkin_ws/src/exercise2/src/dvo.cpp

exercise2/CMakeFiles/dvo.dir/src/dvo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dvo.dir/src/dvo.cpp.i"
	cd /work/catkin_ws/build/exercise2 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /work/catkin_ws/src/exercise2/src/dvo.cpp > CMakeFiles/dvo.dir/src/dvo.cpp.i

exercise2/CMakeFiles/dvo.dir/src/dvo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dvo.dir/src/dvo.cpp.s"
	cd /work/catkin_ws/build/exercise2 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /work/catkin_ws/src/exercise2/src/dvo.cpp -o CMakeFiles/dvo.dir/src/dvo.cpp.s

exercise2/CMakeFiles/dvo.dir/src/dvo.cpp.o.requires:
.PHONY : exercise2/CMakeFiles/dvo.dir/src/dvo.cpp.o.requires

exercise2/CMakeFiles/dvo.dir/src/dvo.cpp.o.provides: exercise2/CMakeFiles/dvo.dir/src/dvo.cpp.o.requires
	$(MAKE) -f exercise2/CMakeFiles/dvo.dir/build.make exercise2/CMakeFiles/dvo.dir/src/dvo.cpp.o.provides.build
.PHONY : exercise2/CMakeFiles/dvo.dir/src/dvo.cpp.o.provides

exercise2/CMakeFiles/dvo.dir/src/dvo.cpp.o.provides.build: exercise2/CMakeFiles/dvo.dir/src/dvo.cpp.o

# Object files for target dvo
dvo_OBJECTS = \
"CMakeFiles/dvo.dir/src/dvo.cpp.o"

# External object files for target dvo
dvo_EXTERNAL_OBJECTS =

/work/catkin_ws/devel/lib/libdvo.so: exercise2/CMakeFiles/dvo.dir/src/dvo.cpp.o
/work/catkin_ws/devel/lib/libdvo.so: exercise2/CMakeFiles/dvo.dir/build.make
/work/catkin_ws/devel/lib/libdvo.so: exercise2/CMakeFiles/dvo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX shared library /work/catkin_ws/devel/lib/libdvo.so"
	cd /work/catkin_ws/build/exercise2 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dvo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
exercise2/CMakeFiles/dvo.dir/build: /work/catkin_ws/devel/lib/libdvo.so
.PHONY : exercise2/CMakeFiles/dvo.dir/build

exercise2/CMakeFiles/dvo.dir/requires: exercise2/CMakeFiles/dvo.dir/src/dvo.cpp.o.requires
.PHONY : exercise2/CMakeFiles/dvo.dir/requires

exercise2/CMakeFiles/dvo.dir/clean:
	cd /work/catkin_ws/build/exercise2 && $(CMAKE_COMMAND) -P CMakeFiles/dvo.dir/cmake_clean.cmake
.PHONY : exercise2/CMakeFiles/dvo.dir/clean

exercise2/CMakeFiles/dvo.dir/depend:
	cd /work/catkin_ws/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /work/catkin_ws/src /work/catkin_ws/src/exercise2 /work/catkin_ws/build /work/catkin_ws/build/exercise2 /work/catkin_ws/build/exercise2/CMakeFiles/dvo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : exercise2/CMakeFiles/dvo.dir/depend
