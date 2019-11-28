cmake_minimum_required(VERSION 3.0)
include(ExternalProject)

project( TP_Interface_AR )

# guard against in-source builds
IF(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds not allowed.")
ENDIF()

# set the path where we can find the findXXX.cmake
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/.cmake)

# add a search path for the glm libraries
# it will work if they are already compiled in /3rdparty
if(NOT DEFINED GLM_DIR)
    set(GLM_DIR ${PROJECT_SOURCE_DIR}/3rdparty/glm/build)
endif() 
list(APPEND CMAKE_LIBRARY_PATH "${GLM_DIR}/lib")
list(APPEND CMAKE_INCLUDE_PATH "${GLM_DIR}/include")


# set the output path for the generated files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib )

#########################################################
# SET COMPILATION FLAGS FOR C++11
#########################################################

if (CMAKE_VERSION VERSION_LESS "3.1")
    set(CMAKE_CXX_FLAGS "-std=gnu++11 ${CMAKE_CXX_FLAGS}")
else ()
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif ()

#########################################################
#
# EXTERNAL LIBRARIES
#
#########################################################

#########################################################
# LIB GLM
#########################################################

#try to find the libraries
find_package(GLM REQUIRED)

if(NOT LIBGLM_FOUND)

    #just to be sure
    set(GLM_DIR ${PROJECT_SOURCE_DIR}/3rdparty/glm)

    # set(EP_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/3rdpary)
    # set_property(DIRECTORY PROPERTY EP_BASE ${PROJECT_SOURCE_DIR}/3rdparty)
    ExternalProject_Add( glm
                        URL http://devernay.free.fr/hacks/glm/glm-0.3.2.tar.gz
                        PREFIX ${GLM_DIR}
                        BUILD_IN_SOURCE 1
                        INSTALL_DIR ${GLM_DIR}/build
                        CONFIGURE_COMMAND ./configure --prefix=${GLM_DIR}/build --enable-shared
                        BUILD_COMMAND make -j 4
                        INSTALL_COMMAND make install)

    ExternalProject_Get_Property(glm install_dir)
    message( "lib glm has not be found, it will be installed in: ${install_dir}")
    set(LIBGLM_INCLUDE_DIRS ${install_dir}/include)   
    # message( "lib glm has not be found, it will be installed in: ${LIBGLM_INCLUDE_DIRS}")

    add_library(glm_lib SHARED IMPORTED)
    set_property(TARGET glm_lib PROPERTY IMPORTED_LOCATION ${install_dir}/lib/libglm.so)
    add_dependencies(glm_lib glm)

    set(LIBGLM_LIBRARIES glm_lib)

else()
    message( "-- Found glm" )
    message( ${LIBGLM_INCLUDE_DIRS} )
    message( ${LIBGLM_LIBRARIES} )
endif(NOT LIBGLM_FOUND)



#########################################################
# FIND OPENCV
#########################################################
set(OpenCV_FOUND OFF)
#if(UNIX)
#    include(FindPkgConfig)
#    pkg_search_module(OpenCV QUIET opencv)
#endif(UNIX)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIBRARY_DIRS})
    set(OpenCV_LIBS ${OpenCV_LIBRARIES})    
else(OpenCV_FOUND)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIB_DIR})
    add_definitions(${OpenCV_DEFINITIONS})
    # MESSAGE( "${OpenCV_LIB_DIR}" )
    # MESSAGE( "${OpenCV_VERSION}" )
    # MESSAGE( "${OpenCV_FOUND}" )
endif(OpenCV_FOUND)

if(NOT OpenCV_FOUND)
    message(ERROR " OpenCV not found!")
else()
    message( "-- Found OpenCV version: ${OpenCV_VERSION}" )
endif(NOT OpenCV_FOUND)

#########################################################
# FIND OPENGL
#########################################################
find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
# MESSAGE( "${OPENGL_LIBRARIES}" )
# MESSAGE( "${OPENGL_LIBRARIES_DIRS}" )
# MESSAGE( "${OPENGL_DEFINITIONS}" )
if(NOT OPENGL_FOUND)
    message(ERROR " OPENGL not found!")
endif(NOT OPENGL_FOUND)

#########################################################
# FIND GLUT
#########################################################
find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIR})
link_directories(${GLUT_LIBRARY_DIRS})
add_definitions(${GLUT_DEFINITIONS})
# message( "glut ${GLUT_INCLUDE_DIR}" )
if(NOT GLUT_FOUND)
    message(ERROR " GLUT not found!")
else()
    message( "-- Found GLUT" )
endif(NOT GLUT_FOUND)


## FreeGLUT ?
# find_package(FreeGLUT REQUIRED)
# include_directories(${FreeGLUT_INCLUDE_DIRS})
# link_directories(${FreeGLUT_LIBRARY_DIRS})
# add_definitions(${FreeGLUT_DEFINITIONS})
# MESSAGE( "${FREEGLUT_INCLUDE_PATH}" )
# MESSAGE( "${FREEGLUT_LIBRARY}" )
# MESSAGE( "${FreeGLUT_DEFINITIONS}" )
# if(NOT FREEGLUT_FOUND)
#     message(ERROR " FreeGLUT not found!")
# endif(NOT FREEGLUT_FOUND)

#########################################################
# Doxygen
#########################################################

# add a target to generate API documentation with Doxygen
find_package(Doxygen QUIET)
# message( "${DOXYGEN_EXECUTABLE}" )
if(DOXYGEN_FOUND)
    set(CMAKE_DOC_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in ${CMAKE_DOC_OUTPUT_DIRECTORY}/Doxyfile @ONLY)
    add_custom_target(doc
            ${DOXYGEN_EXECUTABLE} ${CMAKE_DOC_OUTPUT_DIRECTORY}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_DOC_OUTPUT_DIRECTORY}
            COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif(DOXYGEN_FOUND)



#########################################################
#
# PROJECT LIBRARIES
#
#########################################################

#########################################################
# TRACKER LIBRARY
#########################################################

add_subdirectory(src)

# add_library( tracker STATIC utility.cpp ChessboardCameraTracker.cpp ChessboardCameraTrackerKLT.cpp Camera.cpp)
# target_link_libraries( tracker ${OpenCV_LIBS} )

# install(TARGETS tracker DESTINATION lib)
# install(FILES tracker DESTINATION include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/tracker)

get_property(inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
# message("inc_dirs = ${inc_dirs}")


#########################################################
#
# OPENCV TUTORIALS 
#
#########################################################

#########################################################
#
# PROJECT APPLICATIONS
#
#########################################################

# opencv part


# add_executable( tracking tracking.cpp )
# target_link_libraries( tracking ${OpenCV_LIBS} tracker )





# ## opengl part

# add_executable( helloteapot helloteapot.cpp )
# target_link_libraries( helloteapot  ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )



# ## opengl + OCV part

# add_executable( videoOGL videoOGL.cpp )
# target_link_libraries( videoOGL  ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${OpenCV_LIBS} )
