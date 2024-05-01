# Compute paths
get_filename_component( PROJECT_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH )
SET( Pangolin_INCLUDE_DIRS "/home/gilberto/projects/pangolin/include;/home/gilberto/projects/pangolin/build/src/include;/usr/include;/usr/include;/usr/include;/usr/include/eigen3" )
SET( Pangolin_INCLUDE_DIR  "/home/gilberto/projects/pangolin/include;/home/gilberto/projects/pangolin/build/src/include;/usr/include;/usr/include;/usr/include;/usr/include/eigen3" )

# Library dependencies (contains definitions for IMPORTED targets)
if( NOT TARGET _pangolin AND NOT Pangolin_BINARY_DIR )
  include( "${PROJECT_CMAKE_DIR}/PangolinTargets.cmake" )
  
endif()

SET( Pangolin_LIBRARIES    _pangolin )
SET( Pangolin_LIBRARY      _pangolin )
SET( Pangolin_CMAKEMODULES /home/gilberto/projects/pangolin/src/../CMakeModules )
