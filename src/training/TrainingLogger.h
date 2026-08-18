#pragma once

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

// Everything needed to tell, months later, what configuration produced one
// CSV log -- a flat struct of fields that already exist elsewhere, plus a
// few run-identity fields main.cpp fills in.
struct RunMetadata
{
    std::string runId; // also the CSV/metadata filename stem
    std::string startedAtLocal; // e.g. "2026-08-17 14:03:21"

    // Short git commit hash (OPPARI_BUILD_VERSION, set by CMakeLists.txt), or "unknown".
    std::string buildVersion;

    // main.cpp hardcodes one active track at compile time; passed as a literal.
    std::string trackName;

    // Mirrors FitnessEvaluator.cpp's kMaxEvaluationTime -- update by hand if that ever changes.
    float maxEvaluationTimeSeconds = 60.0f;

    ai::neat::PopulationConfig populationConfig;
    ai::neat::MutationConfig mutationConfig;
    ai::neat::CrossoverConfig crossoverConfig;
    ai::neat::CompatibilityConfig compatibilityConfig;
    ai::neat::SpeciationConfig speciationConfig;
    simulation::CarParams carParams;
};

// CSV header line (no trailing newline), matching generationMetricsToCsvRow()'s
// column order. Pure string building -- no file I/O. New columns are always
// appended at the end so earlier ones keep their index/meaning.
std::string csvHeaderLine();

// One CSV data row (no trailing newline), same column order as csvHeaderLine().
std::string generationMetricsToCsvRow(const GenerationMetrics& metrics);

// Locale-independent decimal string with a '.' separator (std::to_chars
// never consults locale, unlike snprintf/ostream), shortest round-trippable
// representation (812.4f -> "812.4", not "812.400000"). Non-finite values
// format as "nan"/"inf"/"-inf" so this never throws.
std::string formatFloat(float value);

// Persists one run's per-generation metrics to a CSV file plus a companion
// metadata text file, both under resultsDir (created if missing). Never
// overwrites: the run name is a local timestamp
// (training_run_YYYYMMDD_HHMMSS), with a numeric suffix appended if that
// name is already taken (e.g. two runs in the same second).
//
// Each logGeneration() call appends one row and flushes immediately, so a
// crash mid-run loses at most the in-progress generation.
class TrainingLogger
{
public:
    // Throws std::runtime_error if the CSV file can't be created/opened.
    // Metadata is only written after the CSV is confirmed open.
    TrainingLogger(const std::string& resultsDir, const RunMetadata& metadata);

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
