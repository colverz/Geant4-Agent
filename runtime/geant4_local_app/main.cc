#include "G4Box.hh"
#include "G4Colour.hh"
#include "G4Gamma.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4Orb.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4PhysListFactory.hh"
#include "G4PVPlacement.hh"
#include "G4RunManagerFactory.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4TrackingManager.hh"
#include "G4ThreeVector.hh"
#include "G4Tubs.hh"
#include "G4UIExecutive.hh"
#include "G4UImanager.hh"
#include "G4UserEventAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4UserTrackingAction.hh"
#include "G4VisAttributes.hh"
#include "G4VisExecutive.hh"
#include "G4VSolid.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "globals.hh"

#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct RuntimeConfig {
  std::string geometry_structure = "single_box";
  std::string material = "G4_Cu";
  std::string particle = "gamma";
  std::string source_type = "point";
  double energy_mev = 1.0;
  double source_x_mm = 0.0;
  double source_y_mm = 0.0;
  double source_z_mm = -100.0;
  double direction_x = 0.0;
  double direction_y = 0.0;
  double direction_z = 1.0;
  double size_x_mm = 50.0;
  double size_y_mm = 50.0;
  double size_z_mm = 50.0;
  double radius_mm = 25.0;
  double half_length_mm = 50.0;
  std::string physics_list = "FTFP_BERT";
  int events = 1;
  bool score_target_edep = true;
  std::vector<std::string> scoring_volume_names = {"Target"};
  std::string artifact_dir;
  std::string mode = "batch";
};

std::string read_text(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Failed to open config file: " + path.string());
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

std::string extract_top_level_string(const std::string& text, const std::string& key, const std::string& fallback) {
  const std::regex pattern(
      "(?:^|[\\r\\n])\\s*\"" + key + "\"\\s*:\\s*\"([^\"]+)\"",
      std::regex::icase);
  std::smatch match;
  if (std::regex_search(text, match, pattern) && match.size() >= 2) {
    return match[1].str();
  }
  return fallback;
}

double extract_top_level_number(const std::string& text, const std::string& key, double fallback) {
  const std::regex pattern(
      "(?:^|[\\r\\n])\\s*\"" + key + "\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?)",
      std::regex::icase);
  std::smatch match;
  if (std::regex_search(text, match, pattern) && match.size() >= 2) {
    return std::stod(match[1].str());
  }
  return fallback;
}

double extract_top_level_object_number(
    const std::string& text,
    const std::string& object_key,
    const std::string& key,
    double fallback) {
  const std::regex pattern(
      "(?:^|[\\r\\n])\\s*\"" + object_key +
          "\"\\s*:\\s*\\{[^\\}]*?\"" + key + "\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?)",
      std::regex::icase);
  std::smatch match;
  if (std::regex_search(text, match, pattern) && match.size() >= 2) {
    return std::stod(match[1].str());
  }
  return fallback;
}

bool extract_nested_bool(
    const std::string& text,
    const std::string& object_key,
    const std::string& key,
    bool fallback) {
  const std::regex pattern(
      "(?:^|[\\r\\n])\\s*\"" + object_key +
          "\"\\s*:\\s*\\{[^\\}]*?\"" + key + "\"\\s*:\\s*(true|false)",
      std::regex::icase);
  std::smatch match;
  if (std::regex_search(text, match, pattern) && match.size() >= 2) {
    const auto value = match[1].str();
    return value == "true" || value == "TRUE";
  }
  return fallback;
}

std::vector<std::string> extract_nested_string_list(
    const std::string& text,
    const std::string& object_key,
    const std::string& key,
    const std::vector<std::string>& fallback) {
  const std::regex pattern(
      "(?:^|[\\r\\n])\\s*\"" + object_key +
          "\"\\s*:\\s*\\{[^\\}]*?\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]",
      std::regex::icase);
  std::smatch match;
  if (!(std::regex_search(text, match, pattern) && match.size() >= 2)) {
    return fallback;
  }
  const auto body = match[1].str();
  const std::regex item_pattern("\"([^\"]+)\"");
  std::sregex_iterator begin(body.begin(), body.end(), item_pattern);
  std::sregex_iterator end;
  std::vector<std::string> values;
  for (auto it = begin; it != end; ++it) {
    if (it->size() >= 2 && !(*it)[1].str().empty()) {
      values.push_back((*it)[1].str());
    }
  }
  return values.empty() ? fallback : values;
}

struct VolumeScoringMetrics {
  double edep_mev = 0.0;
  double current_event_edep_mev = 0.0;
  int hit_events = 0;
  int step_count = 0;
  int track_entries = 0;
};

struct RuntimeScoringState {
  double target_edep_mev = 0.0;
  double current_event_target_edep_mev = 0.0;
  int target_hit_events = 0;
  int target_step_count = 0;
  int target_track_entries = 0;
  std::map<std::string, VolumeScoringMetrics> volume_stats;
};

RuntimeConfig load_runtime_config(const fs::path& config_path, int events, const fs::path& artifact_dir) {
  RuntimeConfig cfg;
  cfg.events = events;
  cfg.artifact_dir = artifact_dir.string();

  const auto text = read_text(config_path);
  cfg.geometry_structure = extract_top_level_string(text, "structure", cfg.geometry_structure);
  cfg.material = extract_top_level_string(text, "material", cfg.material);
  cfg.particle = extract_top_level_string(text, "particle", cfg.particle);
  cfg.source_type = extract_top_level_string(text, "source_type", cfg.source_type);
  cfg.physics_list = extract_top_level_string(text, "physics_list", cfg.physics_list);
  cfg.energy_mev = extract_top_level_number(text, "energy", cfg.energy_mev);
  cfg.source_x_mm = extract_top_level_object_number(text, "position", "x", cfg.source_x_mm);
  cfg.source_y_mm = extract_top_level_object_number(text, "position", "y", cfg.source_y_mm);
  cfg.source_z_mm = extract_top_level_object_number(text, "position", "z", cfg.source_z_mm);
  cfg.direction_x = extract_top_level_object_number(text, "direction", "x", cfg.direction_x);
  cfg.direction_y = extract_top_level_object_number(text, "direction", "y", cfg.direction_y);
  cfg.direction_z = extract_top_level_object_number(text, "direction", "z", cfg.direction_z);
  cfg.size_x_mm = extract_top_level_number(text, "size_x", cfg.size_x_mm);
  cfg.size_y_mm = extract_top_level_number(text, "size_y", cfg.size_y_mm);
  cfg.size_z_mm = extract_top_level_number(text, "size_z", cfg.size_z_mm);
  cfg.radius_mm = extract_top_level_number(text, "radius", cfg.radius_mm);
  cfg.half_length_mm = extract_top_level_number(text, "half_length", cfg.half_length_mm);
  cfg.score_target_edep = extract_nested_bool(text, "scoring", "target_edep", cfg.score_target_edep);
  cfg.scoring_volume_names =
      extract_nested_string_list(text, "scoring", "volume_names", cfg.scoring_volume_names);
  return cfg;
}

class RuntimeDetectorConstruction : public G4VUserDetectorConstruction {
 public:
  explicit RuntimeDetectorConstruction(RuntimeConfig config) : config_(std::move(config)) {}

  G4VPhysicalVolume* Construct() override {
    auto* nist = G4NistManager::Instance();
    auto* air = nist->FindOrBuildMaterial("G4_AIR");
    auto* target_material = nist->FindOrBuildMaterial(config_.material, false);
    if (!target_material) {
      target_material = air;
    }

    auto* solid_world = new G4Box("World", 200 * mm, 200 * mm, 200 * mm);
    auto* logic_world = new G4LogicalVolume(solid_world, air, "World");

    G4VSolid* solid_target = nullptr;
    if (config_.geometry_structure == "single_tubs") {
      solid_target = new G4Tubs(
          "Target",
          0.0,
          config_.radius_mm * mm,
          config_.half_length_mm * mm,
          0.0,
          360.0 * deg);
    } else {
      solid_target = new G4Box(
          "Target",
          (config_.size_x_mm * 0.5) * mm,
          (config_.size_y_mm * 0.5) * mm,
          (config_.size_z_mm * 0.5) * mm);
    }

    auto* logic_target = new G4LogicalVolume(solid_target, target_material, "Target");
    auto* vis = new G4VisAttributes(G4Colour(0.86, 0.42, 0.16));
    vis->SetForceSolid(true);
    logic_target->SetVisAttributes(vis);

    auto* world_vis = new G4VisAttributes(G4Colour(0.92, 0.94, 0.98));
    world_vis->SetVisibility(false);
    logic_world->SetVisAttributes(world_vis);

    new G4PVPlacement(
        nullptr,
        G4ThreeVector(),
        logic_target,
        "Target",
        logic_world,
        false,
        0,
        true);

    const auto marker_radius_mm = 2.0;
    auto* solid_source_marker = new G4Orb("SourceMarker", marker_radius_mm * mm);
    auto* logic_source_marker = new G4LogicalVolume(solid_source_marker, air, "SourceMarker");
    auto* source_vis = new G4VisAttributes(G4Colour(0.10, 0.45, 0.95));
    source_vis->SetForceSolid(true);
    logic_source_marker->SetVisAttributes(source_vis);

    new G4PVPlacement(
        nullptr,
        G4ThreeVector(config_.source_x_mm * mm, config_.source_y_mm * mm, config_.source_z_mm * mm),
        logic_source_marker,
        "SourceMarker",
        logic_world,
        false,
        0,
        false);

    return new G4PVPlacement(
        nullptr,
        G4ThreeVector(),
        logic_world,
        "World",
        nullptr,
        false,
        0,
        true);
  }

 private:
  RuntimeConfig config_;
};

class RuntimePrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction {
 public:
  explicit RuntimePrimaryGeneratorAction(RuntimeConfig config)
      : config_(std::move(config)), particle_gun_(new G4ParticleGun(1)) {
    auto* particle_table = G4ParticleTable::GetParticleTable();
    auto* particle = particle_table->FindParticle(config_.particle);
    if (!particle) {
      particle = G4Gamma::GammaDefinition();
    }
    particle_gun_->SetParticleDefinition(particle);
    particle_gun_->SetParticleEnergy(config_.energy_mev * MeV);
    auto direction = G4ThreeVector(config_.direction_x, config_.direction_y, config_.direction_z);
    if (direction.mag() == 0.0) {
      direction = G4ThreeVector(0., 0., 1.);
    }
    direction = direction.unit();
    particle_gun_->SetParticleMomentumDirection(direction);
    particle_gun_->SetParticlePosition(
        G4ThreeVector(config_.source_x_mm * mm, config_.source_y_mm * mm, config_.source_z_mm * mm));
  }

  ~RuntimePrimaryGeneratorAction() override { delete particle_gun_; }

  void GeneratePrimaries(G4Event* event) override { particle_gun_->GeneratePrimaryVertex(event); }

 private:
  RuntimeConfig config_;
  G4ParticleGun* particle_gun_;
};

class RuntimeTrackingAction : public G4UserTrackingAction {
 public:
  void PreUserTrackingAction(const G4Track* track) override {
    if (!fpTrackingManager || !track) {
      return;
    }
    fpTrackingManager->SetStoreTrajectory(track->GetParentID() == 0 ? 1 : 0);
  }
};

class RuntimeSteppingAction : public G4UserSteppingAction {
 public:
  RuntimeSteppingAction(RuntimeConfig config, RuntimeScoringState* scoring_state)
      : config_(std::move(config)), scoring_state_(scoring_state) {}

  void UserSteppingAction(const G4Step* step) override {
    if (!step || !scoring_state_ || !config_.score_target_edep) {
      return;
    }
    const auto* pre_point = step->GetPreStepPoint();
    if (!pre_point) {
      return;
    }
    const auto touchable = pre_point->GetTouchableHandle();
    if (!touchable) {
      return;
    }
    const auto* volume = touchable->GetVolume();
    if (!volume) {
      return;
    }
    const auto volume_name = volume->GetName();
    auto it = scoring_state_->volume_stats.find(volume_name);
    if (it == scoring_state_->volume_stats.end()) {
      return;
    }
    auto& metrics = it->second;
    metrics.step_count += 1;
    if (pre_point->GetStepStatus() == fGeomBoundary) {
      metrics.track_entries += 1;
    }
    const auto edep = step->GetTotalEnergyDeposit();
    if (edep > 0.0) {
      const auto edep_mev = edep / MeV;
      metrics.edep_mev += edep_mev;
      metrics.current_event_edep_mev += edep_mev;
    }
  }

 private:
  RuntimeConfig config_;
  RuntimeScoringState* scoring_state_ = nullptr;
};

class RuntimeEventAction : public G4UserEventAction {
 public:
  explicit RuntimeEventAction(RuntimeScoringState* scoring_state) : scoring_state_(scoring_state) {}

  void BeginOfEventAction(const G4Event*) override {
    if (!scoring_state_) {
      return;
    }
    scoring_state_->current_event_target_edep_mev = 0.0;
    for (auto& [_, metrics] : scoring_state_->volume_stats) {
      metrics.current_event_edep_mev = 0.0;
    }
  }

  void EndOfEventAction(const G4Event*) override {
    if (!scoring_state_) {
      return;
    }
    for (auto& [volume_name, metrics] : scoring_state_->volume_stats) {
      if (metrics.current_event_edep_mev > 0.0) {
        metrics.hit_events += 1;
      }
      if (volume_name == "Target") {
        scoring_state_->target_edep_mev = metrics.edep_mev;
        scoring_state_->current_event_target_edep_mev = metrics.current_event_edep_mev;
        scoring_state_->target_hit_events = metrics.hit_events;
        scoring_state_->target_step_count = metrics.step_count;
        scoring_state_->target_track_entries = metrics.track_entries;
      }
    }
  }

 private:
  RuntimeScoringState* scoring_state_ = nullptr;
};

void write_vis_macro(const fs::path& macro_path) {
  std::ofstream out(macro_path, std::ios::trunc);
  out << "/vis/open OGL 800x800-0+0\n";
  out << "/vis/viewer/set/style surface\n";
  out << "/vis/viewer/set/background 1 1 1\n";
  out << "/vis/verbose warnings\n";
  out << "/tracking/storeTrajectory 1\n";
  out << "/vis/drawVolume\n";
  out << "/vis/scene/add/trajectories smooth\n";
  out << "/vis/scene/endOfEventAction accumulate\n";
  out << "/vis/viewer/set/viewpointThetaPhi 70 20\n";
  out << "/vis/viewer/zoom 0.9\n";
  out << "/vis/scene/add/axes 0 0 0 100 mm\n";
  out << "/vis/viewer/flush\n";
}

void write_batch_vis_macro(const fs::path& macro_path) {
  std::ofstream out(macro_path, std::ios::trunc);
  out << "/vis/disable\n";
}

int main(int argc, char** argv) {
  fs::path config_path;
  fs::path artifact_dir = fs::current_path() / "runtime_artifacts";
  int events = 1;
  std::string mode = "batch";

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config" && i + 1 < argc) {
      config_path = argv[++i];
    } else if (arg == "--events" && i + 1 < argc) {
      events = std::stoi(argv[++i]);
    } else if (arg == "--artifact-dir" && i + 1 < argc) {
      artifact_dir = argv[++i];
    } else if (arg == "--mode" && i + 1 < argc) {
      std::string value = argv[++i];
      if (value == "viewer" || value == "batch") {
        mode = value;
      }
    }
  }

  if (config_path.empty()) {
    throw std::runtime_error("Missing --config argument");
  }

  fs::create_directories(artifact_dir);
  auto cfg = load_runtime_config(config_path, events, artifact_dir);
  cfg.mode = mode;

  auto* run_manager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::SerialOnly);
  run_manager->SetUserInitialization(new RuntimeDetectorConstruction(cfg));

  G4PhysListFactory phys_factory;
  auto* physics = phys_factory.GetReferencePhysList(cfg.physics_list);
  if (!physics) {
    physics = phys_factory.GetReferencePhysList("FTFP_BERT");
  }
  run_manager->SetUserInitialization(physics);
  RuntimeScoringState scoring_state;
  for (const auto& volume_name : cfg.scoring_volume_names) {
    if (!volume_name.empty()) {
      scoring_state.volume_stats.emplace(volume_name, VolumeScoringMetrics{});
    }
  }
  run_manager->SetUserAction(new RuntimePrimaryGeneratorAction(cfg));
  run_manager->SetUserAction(new RuntimeEventAction(&scoring_state));
  run_manager->SetUserAction(new RuntimeTrackingAction());
  run_manager->SetUserAction(new RuntimeSteppingAction(cfg, &scoring_state));
  run_manager->Initialize();

  auto* vis_manager = new G4VisExecutive();
  vis_manager->Initialize();

  const auto macro_path = artifact_dir / "vis.mac";
  if (cfg.mode == "viewer") {
    write_vis_macro(macro_path);
  } else {
    write_batch_vis_macro(macro_path);
  }

  auto* ui = G4UImanager::GetUIpointer();
  ui->ApplyCommand("/control/execute " + macro_path.string());
  if (cfg.mode == "viewer") {
    ui->ApplyCommand("/run/initialize");
    ui->ApplyCommand("/run/beamOn " + std::to_string(events));
    ui->ApplyCommand("/vis/viewer/update");
    ui->ApplyCommand("/vis/sceneHandler/attach");
    ui->ApplyCommand("/vis/viewer/refresh");
    ui->ApplyCommand("/vis/viewer/flush");
    std::cout << "viewer_ready" << std::endl;
    std::cout << "Close the Geant4 viewer window to exit." << std::endl;
    auto* idle_ui = new G4UIExecutive(argc, argv);
    idle_ui->SessionStart();
    delete idle_ui;
  } else {
    ui->ApplyCommand("/run/verbose 1");
    ui->ApplyCommand("/event/verbose 0");
    ui->ApplyCommand("/tracking/verbose 0");
    ui->ApplyCommand("/run/beamOn " + std::to_string(events));
  }

  std::ofstream summary(artifact_dir / "run_summary.json", std::ios::trunc);
  summary << "{\n"
          << "  \"run_ok\": true,\n"
          << "  \"events_requested\": " << events << ",\n"
          << "  \"events_completed\": " << events << ",\n"
          << "  \"geometry_structure\": \"" << cfg.geometry_structure << "\",\n"
          << "  \"material\": \"" << cfg.material << "\",\n"
          << "  \"particle\": \"" << cfg.particle << "\",\n"
          << "  \"source_type\": \"" << cfg.source_type << "\",\n"
          << "  \"source_position_mm\": [" << cfg.source_x_mm << ", " << cfg.source_y_mm << ", " << cfg.source_z_mm
          << "],\n"
          << "  \"source_direction\": [" << cfg.direction_x << ", " << cfg.direction_y << ", " << cfg.direction_z
          << "],\n"
          << "  \"physics_list\": \"" << cfg.physics_list << "\",\n"
          << "  \"events\": " << events << ",\n"
          << "  \"mode\": \"" << cfg.mode << "\",\n"
          << "  \"scoring\": {\n"
          << "    \"target_edep_enabled\": " << (cfg.score_target_edep ? "true" : "false") << ",\n"
          << "    \"target_edep_total_mev\": " << scoring_state.target_edep_mev << ",\n"
          << "    \"target_edep_mean_mev_per_event\": "
          << (events > 0 ? (scoring_state.target_edep_mev / static_cast<double>(events)) : 0.0) << ",\n"
          << "    \"target_hit_events\": " << scoring_state.target_hit_events << ",\n"
          << "    \"target_step_count\": " << scoring_state.target_step_count << ",\n"
          << "    \"target_track_entries\": " << scoring_state.target_track_entries << ",\n"
          << "    \"volume_stats\": {\n";
  bool first_volume = true;
  for (const auto& [volume_name, metrics] : scoring_state.volume_stats) {
    if (!first_volume) {
      summary << ",\n";
    }
    first_volume = false;
    summary << "      \"" << volume_name << "\": {\n"
            << "        \"edep_total_mev\": " << metrics.edep_mev << ",\n"
            << "        \"edep_mean_mev_per_event\": "
            << (events > 0 ? (metrics.edep_mev / static_cast<double>(events)) : 0.0) << ",\n"
            << "        \"hit_events\": " << metrics.hit_events << ",\n"
            << "        \"step_count\": " << metrics.step_count << ",\n"
            << "        \"track_entries\": " << metrics.track_entries << "\n"
            << "      }";
  }
  summary << "\n"
          << "    }\n"
          << "  }\n"
          << "}\n";

  delete vis_manager;
  delete run_manager;
  return 0;
}
