cmake_minimum_required(VERSION 3.20)
project(model_improvement_lab_capture LANGUAGES C CXX)

if(NOT LLAMA_CPP_DIR)
  message(FATAL_ERROR "Set LLAMA_CPP_DIR to the compatible llama.cpp fork checkout")
endif()

set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
add_subdirectory("${LLAMA_CPP_DIR}" llama.cpp EXCLUDE_FROM_ALL)

add_executable(dflash-capture
  capture.cpp
  capture_io.cpp
  feature_capture.cpp
)
add_executable(dflash-tokenize tokenize_prompts.cpp)

foreach(target IN ITEMS dflash-capture dflash-tokenize)
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_include_directories(${target} PRIVATE "${LLAMA_CPP_DIR}/src" "${LLAMA_CPP_DIR}/vendor")
  target_link_libraries(${target} PRIVATE llama-common llama)
endforeach()
