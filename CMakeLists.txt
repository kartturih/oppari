cmake_minimum_required(VERSION 3.25)

project(NeatCarSimulation
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG 6.0
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(raylib)

# Stage 20: Box2D provides the rigid-body integration and force/impulse API
# that simulation::Car's vehicle model is built on (see Car.h/.cpp) -- track
# collision itself stays entirely on Track's CPU pixel mask (see Track.h),
# never on Box2D geometry. Box2D v3 is a from-scratch C rewrite (different
# API from the older, still-widely-referenced v2.x C++ line) with no
# samples/tests/benchmarks pulled in here: those default to ON only when
# Box2D is configured as the top-level CMake project (see its own
# CMakeLists.txt's `if(PROJECT_IS_TOP_LEVEL)` guard), which is never the
# case when it is added via FetchContent as a dependency the way raylib is
# above -- so this stays a minimal, single-library dependency with no extra
# transitive fetches (e.g. enkiTS, which only the samples/tests need).
FetchContent_Declare(
    box2d
    GIT_REPOSITORY https://github.com/erincatto/box2d.git
    GIT_TAG v3.1.1
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(box2d)

# Stage 21: a short git commit hash, baked in at configure time, so a
# training run's metadata file can record exactly which build produced it
# (see training::RunMetadata::buildVersion / TrainingLogger.cpp). Falls back
# to "unknown" if git is not available or CMAKE_SOURCE_DIR is not inside a
# git repository -- never fails the configure step.
find_package(Git QUIET)
set(OPPARI_BUILD_VERSION "unknown")
if(Git_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE OPPARI_BUILD_VERSION_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE OPPARI_GIT_RESULT
    )
    if(OPPARI_GIT_RESULT EQUAL 0 AND NOT OPPARI_BUILD_VERSION_OUTPUT STREQUAL "")
        set(OPPARI_BUILD_VERSION ${OPPARI_BUILD_VERSION_OUTPUT})
    endif()
endif()

# Box2D's own CMakeLists.txt unconditionally points
# CMAKE_LIBRARY_OUTPUT_DIRECTORY/CMAKE_RUNTIME_OUTPUT_DIRECTORY at its own
# build tree (so its samples executable can find its DLL) -- reset both
# back to this project's build directory root so neat_car_simulation.exe
# keeps landing exactly where it always has (build/neat_car_simulation.exe),
# unaffected by which dependency happens to be fetched first.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

add_executable(neat_car_simulation
    src/main.cpp
    src/AppConfig.cpp
    src/ui/SimulationRenderer.cpp
    src/ui/DebugRenderer.cpp
    src/ui/HudRenderer.cpp
    src/input/ManualInput.cpp
    src/verification/VerificationRunner.cpp
    src/verification/TrackVerification.cpp
    src/verification/VehicleVerification.cpp
    src/verification/NeuralNetworkVerification.cpp
    src/verification/NeatGenomeVerification.cpp
    src/verification/NeatMutationVerification.cpp
    src/verification/NeatCrossoverVerification.cpp
    src/verification/NeatSpeciationVerification.cpp
    src/verification/PopulationVerification.cpp
    src/verification/FitnessVerification.cpp
    src/verification/TrainingVerification.cpp
    src/simulation/Track.cpp
    src/simulation/TrackVisual.cpp
    src/simulation/Car.cpp
    src/simulation/TrackProgress.cpp
    src/ai/Observation.cpp
    src/ai/NeuralNetwork.cpp
    src/ai/AIController.cpp
    src/ai/FitnessEvaluator.cpp
    src/ai/neat/NodeGene.cpp
    src/ai/neat/ConnectionGene.cpp
    src/ai/neat/Genome.cpp
    src/ai/neat/PhenotypeBuilder.cpp
    src/ai/neat/GenomeMutator.cpp
    src/ai/neat/InnovationTracker.cpp
    src/ai/neat/GenomeCrossover.cpp
    src/ai/neat/CompatibilityDistance.cpp
    src/ai/neat/Species.cpp
    src/ai/neat/Speciator.cpp
    src/ai/neat/Individual.cpp
    src/ai/neat/Population.cpp
    src/training/GenerationMetrics.cpp
    src/training/TrainingLogger.cpp
)

target_include_directories(neat_car_simulation PRIVATE src)

# Stage 18: track image assets (assets/tracks/...) are loaded via an
# absolute, source-tree-relative path baked in at compile time, so asset
# loading does not depend on the executable's current working directory
# (which differs between running from build/ directly vs. from an IDE).
target_compile_definitions(neat_car_simulation PRIVATE
    OPPARI_ASSETS_DIR="${CMAKE_SOURCE_DIR}/assets"
    OPPARI_BUILD_VERSION="${OPPARI_BUILD_VERSION}"
)

target_link_libraries(neat_car_simulation PRIVATE raylib box2d::box2d)

# Dev-only tool: regenerates assets/tracks/test/fixtures/*.png (see
# tools/generate_test_fixtures.cpp). Not built by default, not a dependency
# of neat_car_simulation -- build/run it explicitly only when a fixture
# needs to be regenerated.
add_executable(generate_test_fixtures EXCLUDE_FROM_ALL tools/generate_test_fixtures.cpp)
target_link_libraries(generate_test_fixtures PRIVATE raylib)
target_compile_definitions(generate_test_fixtures PRIVATE
    OPPARI_ASSETS_DIR="${CMAKE_SOURCE_DIR}/assets"
)