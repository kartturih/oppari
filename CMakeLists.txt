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

add_executable(neat_car_simulation
    src/main.cpp
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
)

target_include_directories(neat_car_simulation PRIVATE src)

# Stage 18: track image assets (assets/tracks/...) are loaded via an
# absolute, source-tree-relative path baked in at compile time, so asset
# loading does not depend on the executable's current working directory
# (which differs between running from build/ directly vs. from an IDE).
target_compile_definitions(neat_car_simulation PRIVATE
    OPPARI_ASSETS_DIR="${CMAKE_SOURCE_DIR}/assets"
)

target_link_libraries(neat_car_simulation PRIVATE raylib)