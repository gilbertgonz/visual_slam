#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "_pangolin" for configuration "Release"
set_property(TARGET _pangolin APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(_pangolin PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C;CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "/usr/lib/x86_64-linux-gnu/libGL.so;/usr/lib/x86_64-linux-gnu/libGLU.so;/usr/lib/x86_64-linux-gnu/libGLEW.so;/usr/lib/x86_64-linux-gnu/libX11.so;rt;pthread;/usr/lib/x86_64-linux-gnu/libpython3.10.so"
  IMPORTED_LOCATION_RELEASE "/usr/local/lib/lib_pangolin.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS _pangolin )
list(APPEND _IMPORT_CHECK_FILES_FOR__pangolin "/usr/local/lib/lib_pangolin.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
