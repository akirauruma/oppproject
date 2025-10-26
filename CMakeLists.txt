cmake_minimum_required(VERSION 3.10)
project(MyCppProject)

set(CMAKE_CXX_STANDARD 17)

add_executable(main main.cpp)
target_compile_definitions(main PRIVATE DEMO=1)

enable_testing()
add_executable(tests test.cpp)
add_test(NAME SimpleTest COMMAND tests)
