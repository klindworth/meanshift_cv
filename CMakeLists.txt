cmake_minimum_required(VERSION 2.8)

project(meanshift_cv)

find_package(OpenCV REQUIRED)

set(SOURCES ${SOURCES}
    rlist.cpp
    ms.cpp
    ms_cv.cpp
    msImageProcessor.cpp
    msSysPrompt.cpp
    RAList.cpp
)

set(HEADERS ${HEADERS}
    rlist.h
    tdef.h
    ms.h
    ms_cv.h
    msImageProcessor.h
    msSys.h
    RAList.h
)

add_library(${PROJECT_NAME} SHARED
    ${HEADERS}
    ${SOURCES}
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -O3")
endif()

