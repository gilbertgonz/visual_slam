# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/gilberto/projects/pangolin

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/gilberto/projects/pangolin/build

# Include any dependencies generated for this target.
include examples/HelloPangolin/CMakeFiles/HelloPangolin.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/HelloPangolin/CMakeFiles/HelloPangolin.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/HelloPangolin/CMakeFiles/HelloPangolin.dir/progress.make

# Include the compile flags for this target's objects.
include examples/HelloPangolin/CMakeFiles/HelloPangolin.dir/flags.make

examples/HelloPangolin/CMakeFiles/HelloPangolin.dir/main.cpp.o: examples/HelloPangolin/CMakeFiles/HelloPangolin.dir/flags.make
examples/HelloPangolin/CMakeFiles/HelloPangolin.dir/main.cpp.o: ../examples/HelloPangolin/main.cpp
examples/HelloPangolin/CMakeFiles/HelloPangolin.dir/main.cpp.o: examples/HelloPangolin/CMakeFiles/HelloPangolin.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gilberto/projects/pangolin/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/HelloPangolin/CMakeFiles/HelloPangolin.dir/main.cpp.o"
	cd /home/gilberto/projects/pangolin/build/examples/HelloPangolin && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT examples/HelloPangolin/CMakeFiles/HelloPangolin.dir/main.cpp.o -MF CMakeFiles/HelloPangolin.dir/main.cpp.o.d -o CMakeFiles/HelloPangolin.dir/main.cpp.o -c /home/gilberto/projects/pangolin/examples/HelloPangolin/main.cpp

examples/HelloPangolin/CMakeFiles/HelloPangolin.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/HelloPangolin.dir/main.cpp.i"
	cd /home/gilberto/projects/pangolin/build/examples/HelloPangolin && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gilberto/projects/pangolin/examples/HelloPangolin/main.cpp > CMakeFiles/HelloPangolin.dir/main.cpp.i

examples/HelloPangolin/CMakeFiles/HelloPangolin.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/HelloPangolin.dir/main.cpp.s"
	cd /home/gilberto/projects/pangolin/build/examples/HelloPangolin && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gilberto/projects/pangolin/examples/HelloPangolin/main.cpp -o CMakeFiles/HelloPangolin.dir/main.cpp.s

# Object files for target HelloPangolin
HelloPangolin_OBJECTS = \
"CMakeFiles/HelloPangolin.dir/main.cpp.o"

# External object files for target HelloPangolin
HelloPangolin_EXTERNAL_OBJECTS =

examples/HelloPangolin/HelloPangolin: examples/HelloPangolin/CMakeFiles/HelloPangolin.dir/main.cpp.o
examples/HelloPangolin/HelloPangolin: examples/HelloPangolin/CMakeFiles/HelloPangolin.dir/build.make
examples/HelloPangolin/HelloPangolin: src/lib_pangolin.a
examples/HelloPangolin/HelloPangolin: /usr/lib/x86_64-linux-gnu/libGL.so
examples/HelloPangolin/HelloPangolin: /usr/lib/x86_64-linux-gnu/libGLU.so
examples/HelloPangolin/HelloPangolin: /usr/lib/x86_64-linux-gnu/libGLEW.so
examples/HelloPangolin/HelloPangolin: /usr/lib/x86_64-linux-gnu/libX11.so
examples/HelloPangolin/HelloPangolin: /usr/lib/x86_64-linux-gnu/libpython3.10.so
examples/HelloPangolin/HelloPangolin: examples/HelloPangolin/CMakeFiles/HelloPangolin.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/gilberto/projects/pangolin/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable HelloPangolin"
	cd /home/gilberto/projects/pangolin/build/examples/HelloPangolin && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/HelloPangolin.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/HelloPangolin/CMakeFiles/HelloPangolin.dir/build: examples/HelloPangolin/HelloPangolin
.PHONY : examples/HelloPangolin/CMakeFiles/HelloPangolin.dir/build

examples/HelloPangolin/CMakeFiles/HelloPangolin.dir/clean:
	cd /home/gilberto/projects/pangolin/build/examples/HelloPangolin && $(CMAKE_COMMAND) -P CMakeFiles/HelloPangolin.dir/cmake_clean.cmake
.PHONY : examples/HelloPangolin/CMakeFiles/HelloPangolin.dir/clean

examples/HelloPangolin/CMakeFiles/HelloPangolin.dir/depend:
	cd /home/gilberto/projects/pangolin/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/gilberto/projects/pangolin /home/gilberto/projects/pangolin/examples/HelloPangolin /home/gilberto/projects/pangolin/build /home/gilberto/projects/pangolin/build/examples/HelloPangolin /home/gilberto/projects/pangolin/build/examples/HelloPangolin/CMakeFiles/HelloPangolin.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/HelloPangolin/CMakeFiles/HelloPangolin.dir/depend

