find_package(nlohmann_json 3.9.0 REQUIRED)

add_library(nlohmann_json INTERFACE )
target_link_libraries(nlohmann_json INTERFACE "nlohmann_json::nlohmann_json")