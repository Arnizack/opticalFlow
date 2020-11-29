#add_subdirectory(Hypodermic)
add_library(hypodermic INTERFACE)
target_include_directories(hypodermic INTERFACE "Hypodermic")

set(
    TESTSRC
    Hypodermic/Hypodermic.Tests/main.cpp
    Hypodermic/Hypodermic.Tests/CircularDependencyTests.cpp
    Hypodermic/Hypodermic.Tests/ContainerBuilderTests.cpp
    Hypodermic/Hypodermic.Tests/ContainerTests.cpp
    Hypodermic/Hypodermic.Tests/DefaultConstructibleTests.cpp
    Hypodermic/Hypodermic.Tests/FactoryTests.cpp
    Hypodermic/Hypodermic.Tests/IsCompleteTests.cpp
    Hypodermic/Hypodermic.Tests/MemoryTests.cpp
    Hypodermic/Hypodermic.Tests/NamedTests.cpp
    Hypodermic/Hypodermic.Tests/NestedContainerTests.cpp
    Hypodermic/Hypodermic.Tests/PerformanceTests.cpp
    Hypodermic/Hypodermic.Tests/PersistentInstanceRegistrationTests.cpp
    Hypodermic/Hypodermic.Tests/ProvidedDependenciesTests.cpp
    Hypodermic/Hypodermic.Tests/ProvidedInstanceFactoryRegistrationTests.cpp
    Hypodermic/Hypodermic.Tests/ProvidedInstanceRegistrationTests.cpp
    Hypodermic/Hypodermic.Tests/RegistrationTests.cpp
    Hypodermic/Hypodermic.Tests/RuntimeRegistrationTests.cpp
    Hypodermic/Hypodermic.Tests/UseIfNoneTests.cpp
)

add_executable(hypodermic.tests ${TESTSRC})
target_link_libraries(hypodermic.tests boost hypodermic)
target_include_directories(hypodermic.tests PRIVATE "Hypodermic/Hypodermic.Tests")

set_target_properties(hypodermic.tests PROPERTIES FOLDER "/libs/tests")
