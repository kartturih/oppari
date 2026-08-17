#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>

#include "ai/neat/CompatibilityConfig.h"
#include "ai/neat/CrossoverConfig.h"
#include "ai/neat/MutationConfig.h"
#include "ai/neat/PopulationConfig.h"
#include "ai/neat/SpeciationConfig.h"
#include "simulation/Car.h"
#include "training/GenerationMetrics.h"

namespace training
{

// Everything needed to tell, months later, exactly what configuration
// produced one CSV experiment log -- written once per run as a small
// companion text file (see TrainingLogger::TrainingLogger()). Deliberately a
// flat struct of the fields that already exist elsewhere in the codebase
// (PopulationConfig, MutationConfig, CrossoverConfig, CompatibilityConfig,
// SpeciationConfig, simulation::CarParams) plus a handful of run-identity
// fields main.cpp fills in -- not a new configuration framework, and it
// changes none of those existing configs' own behavior or defaults.
struct RunMetadata
{
    // Wall-clock run identity. runId is also the CSV/metadata filename stem
    // (see TrainingLogger) -- stored here too so the metadata file is
    // self-describing even if renamed.
    std::string runId;
    std::string startedAtLocal; // human-readable local timestamp, e.g. "2026-08-17 14:03:21"

    // Build/version identifier -- the short git commit hash the running
    // executable was built from (OPPARI_BUILD_VERSION, set by
    // CMakeLists.txt from `git rev-parse --short HEAD` at configure time),
    // or "unknown" if that could not be determined (e.g. no git available,
    // or building from a source tree with no .git directory).
    std::string buildVersion;

    // Active track -- main.cpp hardcodes exactly one active track at compile
    // time (see makeTrackDefinition()); there is no runtime track-switching
    // mechanism to read this from, so main.cpp passes the matching literal
    // name directly.
    std::string trackName;

    // Mirrors FitnessEvaluator.cpp's own kMaxEvaluationTime constant (30.0f
    // seconds, unchanged in this stage -- see FitnessEvaluator.h/.cpp,
    // deliberately not exposed as a public accessor so this stage touches
    // zero bytes of fitness-evaluation code). If that constant is ever
    // changed by a future stage, this literal must be updated to match by
    // hand -- out of scope here.
    float maxEvaluationTimeSeconds = 30.0f;

    ai::neat::PopulationConfig populationConfig;
    ai::neat::MutationConfig mutationConfig;
    ai::neat::CrossoverConfig crossoverConfig;
    ai::neat::CompatibilityConfig compatibilityConfig;
    ai::neat::SpeciationConfig speciationConfig;
    simulation::CarParams carParams;
};

// The CSV header line (no trailing newline), in the exact column order every
// row from generationMetricsToCsvRow() below follows. Exposed as a pure
// string-building function (no file I/O) so it -- and its agreement with
// generationMetricsToCsvRow()'s column count/order -- can be checked
// directly by main.cpp's verify*() suite without touching the filesystem.
std::string csvHeaderLine();

// One CSV data row (no trailing newline) for metrics, in the exact column
// order csvHeaderLine() declares. Every float field is formatted with
// formatFloat() below (locale-independent, '.' decimal separator, shortest
// round-trippable representation -- see its own doc comment); every
// std::size_t field with plain decimal digits (std::to_string(), which is
// itself locale-independent for integers). Pure string building -- no file
// I/O.
std::string generationMetricsToCsvRow(const GenerationMetrics& metrics);

// Formats value as a locale-independent decimal string with a '.' separator
// -- always, regardless of the process's global C/C++ locale (std::to_chars
// is defined by the C++17 standard to never consult locale at all, unlike
// snprintf("%f", ...) or std::ostream with the default/"C" locale, both of
// which follow whatever locale happens to be globally active). Uses the
// shortest representation that round-trips back to exactly value (no fixed
// decimal-place count, no trailing zeros) -- e.g. 812.4f formats as
// "812.4", not "812.400000". Falls back to the literal strings "nan"/
// "inf"/"-inf" for a non-finite value (std::to_chars itself rejects
// NaN/Inf) so a logging call can never throw or crash even if an upstream
// computation somehow produced one.
std::string formatFloat(float value);

// Persists one training run's per-generation metrics to a CSV file, plus a
// companion metadata text file -- both under resultsDir (created, including
// any missing parent directories, if it does not already exist). Neither
// file is ever overwritten by a later run: the run name is derived from the
// construction-time local timestamp (training_run_YYYYMMDD_HHMMSS), and if a
// file with that exact stem already exists (e.g. two runs started within the
// same second) a numeric suffix (-2, -3, ...) is appended until an unused
// name is found -- see the .cpp for the exact algorithm.
//
// Every logGeneration() call appends exactly one row and flushes the
// underlying file stream immediately, so a long training run that is closed
// (or crashes) mid-run never loses any generation already logged -- at most
// the in-progress one is missing.
class TrainingLogger
{
public:
    // Throws std::runtime_error if the CSV file cannot be created/opened for
    // writing (e.g. resultsDir cannot be created, or the filesystem
    // otherwise refuses the write) -- metadata is only written after the CSV
    // file itself is confirmed open, so a failure here never leaves an
    // orphaned metadata file with no matching CSV.
    TrainingLogger(const std::string& resultsDir, const RunMetadata& metadata);

    // Appends generationMetricsToCsvRow(metrics) as one line and flushes.
    void logGeneration(const GenerationMetrics& metrics);

    const std::string& getCsvPath() const { return m_csvPath; }
    const std::string& getMetadataPath() const { return m_metadataPath; }
    const std::string& getRunId() const { return m_runId; }

private:
    std::ofstream m_csv;
    std::string m_csvPath;
    std::string m_metadataPath;
    std::string m_runId;
};

} // namespace training
