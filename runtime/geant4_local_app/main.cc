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
#include "G4Version.hh"
#include "G4VisAttributes.hh"
#include "G4VisExecutive.hh"
#include "G4VSolid.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "Randomize.hh"
#include "globals.hh"
#include "third_party/json.hpp"

#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;
constexpr const char* kResultSchemaVersion = "2026-04-14.v1";

struct RuntimeConfig {
  std::string geometry_structure = "single_box";
  std::string material = "G4_Cu";
  std::string root_volume_name = "Target";
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
  bool detector_enabled = false;
  std::string detector_name = "Detector";
  std::string detector_material = "G4_Si";
  double detector_x_mm = 0.0;
  double detector_y_mm = 0.0;
  double detector_z_mm = 100.0;
  double detector_size_x_mm = 20.0;
  double detector_size_y_mm = 20.0;
  double detector_size_z_mm = 2.0;
  std::string physics_list = "FTFP_BERT";
  int events = 1;
  int seed = 1337;
  bool score_target_edep = true;
  bool score_detector_crossings = true;
  bool score_plane_crossings = false;
  std::string scoring_plane_name = "ScoringPlane";
  double scoring_plane_z_mm = 0.0;
  std::vector<std::string> scoring_volume_names = {"Target"};
  std::map<std::string, std::vector<std::string>> scoring_volume_roles = {{"target", {"Target"}}};
  std::string payload_sha256;
  std::string artifact_dir;
  std::string mode = "batch";
};

json read_json(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Failed to open config file: " + path.string());
  }
  json payload;
  input >> payload;
  return payload;
}

template <typename T>
T json_value(const json& node, const char* key, const T& fallback) {
  if (!node.is_object() || !node.contains(key) || node.at(key).is_null()) {
    return fallback;
  }
  try {
    return node.at(key).get<T>();
  } catch (const json::exception&) {
    return fallback;
  }
}

std::vector<std::string> json_string_list(const json& node, const char* key, const std::vector<std::string>& fallback) {
  if (!node.is_object() || !node.contains(key) || !node.at(key).is_array()) {
    return fallback;
  }
  std::vector<std::string> values;
  for (const auto& item : node.at(key)) {
    if (item.is_string()) {
      const auto value = item.get<std::string>();
      if (!value.empty()) {
        values.push_back(value);
      }
    }
  }
  return values.empty() ? fallback : values;
}

struct VolumeScoringMetrics {
  double edep_mev = 0.0;
  double current_event_edep_mev = 0.0;
  int hit_events = 0;
  bool current_event_crossed = false;
  int crossing_events = 0;
  int crossing_count = 0;
  int step_count = 0;
  int track_entries = 0;
};

struct RuntimeScoringState {
  std::string target_volume_name = "Target";
  double target_edep_mev = 0.0;
  double current_event_target_edep_mev = 0.0;
  int target_hit_events = 0;
  int target_step_count = 0;
  int target_track_entries = 0;
  std::string scoring_plane_name = "ScoringPlane";
  double scoring_plane_z_mm = 0.0;
  bool current_event_plane_crossed = false;
  int plane_crossing_events = 0;
  int plane_crossing_count = 0;
  std::map<std::string, VolumeScoringMetrics> volume_stats;
};

std::map<std::string, VolumeScoringMetrics> derive_role_stats(
    const std::map<std::string, VolumeScoringMetrics>& volume_stats,
    const std::map<std::string, std::vector<std::string>>& volume_roles) {
  std::map<std::string, VolumeScoringMetrics> role_stats;
  for (const auto& [role_name, volume_names] : volume_roles) {
    VolumeScoringMetrics aggregate;
    bool matched = false;
    for (const auto& volume_name : volume_names) {
      const auto it = volume_stats.find(volume_name);
      if (it == volume_stats.end()) {
        continue;
      }
      matched = true;
      aggregate.edep_mev += it->second.edep_mev;
      aggregate.current_event_edep_mev += it->second.current_event_edep_mev;
      aggregate.hit_events += it->second.hit_events;
      aggregate.crossing_events += it->second.crossing_events;
      aggregate.crossing_count += it->second.crossing_count;
      aggregate.step_count += it->second.step_count;
      aggregate.track_entries += it->second.track_entries;
    }
    if (matched) {
      role_stats.emplace(role_name, aggregate);
    }
  }
  return role_stats;
}

void write_string_array(std::ostream& out, const std::vector<std::string>& values) {
  out << "[";
  bool first = true;
  for (const auto& value : values) {
    if (!first) {
      out << ", ";
    }
    first = false;
    out << "\"" << value << "\"";
  }
  out << "]";
}

void write_role_map(std::ostream& out, const std::map<std::string, std::vector<std::string>>& roles) {
  out << "{";
  bool first_role = true;
  for (const auto& [role_name, values] : roles) {
    if (!first_role) {
      out << ", ";
    }
    first_role = false;
    out << "\"" << role_name << "\": ";
    write_string_array(out, values);
  }
  out << "}";
}

RuntimeConfig load_runtime_config(const fs::path& config_path, int events, const fs::path& artifact_dir) {
  RuntimeConfig cfg;
  cfg.events = events;
  cfg.artifact_dir = artifact_dir.string();

  const auto payload = read_json(config_path);
  cfg.geometry_structure = json_value<std::string>(payload, "structure", cfg.geometry_structure);
  cfg.material = json_value<std::string>(payload, "material", cfg.material);
  cfg.root_volume_name = json_value<std::string>(payload, "root_volume_name", cfg.root_volume_name);
  cfg.particle = json_value<std::string>(payload, "particle", cfg.particle);
  cfg.source_type = json_value<std::string>(payload, "source_type", cfg.source_type);
  cfg.physics_list = json_value<std::string>(payload, "physics_list", cfg.physics_list);
  cfg.energy_mev = json_value<double>(payload, "energy", cfg.energy_mev);

  if (payload.contains("position") && payload.at("position").is_object()) {
    const auto& position = payload.at("position");
    cfg.source_x_mm = json_value<double>(position, "x", cfg.source_x_mm);
    cfg.source_y_mm = json_value<double>(position, "y", cfg.source_y_mm);
    cfg.source_z_mm = json_value<double>(position, "z", cfg.source_z_mm);
  }
  if (payload.contains("direction") && payload.at("direction").is_object()) {
    const auto& direction = payload.at("direction");
    cfg.direction_x = json_value<double>(direction, "x", cfg.direction_x);
    cfg.direction_y = json_value<double>(direction, "y", cfg.direction_y);
    cfg.direction_z = json_value<double>(direction, "z", cfg.direction_z);
  }

  cfg.size_x_mm = json_value<double>(payload, "size_x", cfg.size_x_mm);
  cfg.size_y_mm = json_value<double>(payload, "size_y", cfg.size_y_mm);
  cfg.size_z_mm = json_value<double>(payload, "size_z", cfg.size_z_mm);
  cfg.radius_mm = json_value<double>(payload, "radius", cfg.radius_mm);
  cfg.half_length_mm = json_value<double>(payload, "half_length", cfg.half_length_mm);

  cfg.detector_enabled = json_value<bool>(payload, "detector_enabled", cfg.detector_enabled);
  cfg.detector_name = json_value<std::string>(payload, "detector_name", cfg.detector_name);
  cfg.detector_material = json_value<std::string>(payload, "detector_material", cfg.detector_material);
  if (payload.contains("detector_position") && payload.at("detector_position").is_object()) {
    const auto& detector_position = payload.at("detector_position");
    cfg.detector_x_mm = json_value<double>(detector_position, "x", cfg.detector_x_mm);
    cfg.detector_y_mm = json_value<double>(detector_position, "y", cfg.detector_y_mm);
    cfg.detector_z_mm = json_value<double>(detector_position, "z", cfg.detector_z_mm);
  }
  cfg.detector_size_x_mm = json_value<double>(payload, "detector_size_x", cfg.detector_size_x_mm);
  cfg.detector_size_y_mm = json_value<double>(payload, "detector_size_y", cfg.detector_size_y_mm);
  cfg.detector_size_z_mm = json_value<double>(payload, "detector_size_z", cfg.detector_size_z_mm);

  if (payload.contains("run") && payload.at("run").is_object()) {
    const auto& run = payload.at("run");
    cfg.seed = json_value<int>(run, "seed", cfg.seed);
    cfg.mode = json_value<std::string>(run, "mode", cfg.mode);
  }

  if (payload.contains("scoring") && payload.at("scoring").is_object()) {
    const auto& scoring = payload.at("scoring");
    cfg.score_target_edep = json_value<bool>(scoring, "target_edep", cfg.score_target_edep);
    cfg.score_detector_crossings = json_value<bool>(scoring, "detector_crossings", cfg.score_detector_crossings);
    cfg.score_plane_crossings = json_value<bool>(scoring, "plane_crossings", cfg.score_plane_crossings);
    if (scoring.contains("plane") && scoring.at("plane").is_object()) {
      const auto& plane = scoring.at("plane");
      cfg.scoring_plane_name = json_value<std::string>(plane, "name", cfg.scoring_plane_name);
      cfg.scoring_plane_z_mm = json_value<double>(plane, "z_mm", cfg.scoring_plane_z_mm);
    }
    cfg.scoring_volume_names = json_string_list(scoring, "volume_names", cfg.scoring_volume_names);
    if (scoring.contains("volume_roles") && scoring.at("volume_roles").is_object()) {
      cfg.scoring_volume_roles.clear();
      for (auto it = scoring.at("volume_roles").begin(); it != scoring.at("volume_roles").end(); ++it) {
        std::vector<std::string> names;
        if (it.value().is_array()) {
          for (const auto& item : it.value()) {
            if (item.is_string() && !item.get<std::string>().empty()) {
              names.push_back(item.get<std::string>());
            }
          }
        } else if (it.value().is_string() && !it.value().get<std::string>().empty()) {
          names.push_back(it.value().get<std::string>());
        }
        if (!names.empty()) {
          cfg.scoring_volume_roles[it.key()] = names;
        }
      }
      if (cfg.scoring_volume_roles.empty()) {
        cfg.scoring_volume_roles["target"] = {cfg.root_volume_name};
      }
    }
  }

  cfg.payload_sha256 = json_value<std::string>(payload, "payload_sha256", cfg.payload_sha256);
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
        config_.root_volume_name,
        logic_world,
        false,
        0,
        true);

    if (config_.detector_enabled) {
      auto* detector_material = nist->FindOrBuildMaterial(config_.detector_material, false);
      if (!detector_material) {
        detector_material = air;
      }
      auto* solid_detector = new G4Box(
          "Detector",
          (config_.detector_size_x_mm * 0.5) * mm,
          (config_.detector_size_y_mm * 0.5) * mm,
          (config_.detector_size_z_mm * 0.5) * mm);
      auto* logic_detector = new G4LogicalVolume(solid_detector, detector_material, "Detector");
      auto* detector_vis = new G4VisAttributes(G4Colour(0.12, 0.65, 0.32));
      detector_vis->SetForceSolid(true);
      logic_detector->SetVisAttributes(detector_vis);
      new G4PVPlacement(
          nullptr,
          G4ThreeVector(config_.detector_x_mm * mm, config_.detector_y_mm * mm, config_.detector_z_mm * mm),
          logic_detector,
          config_.detector_name,
          logic_world,
          false,
          0,
          true);
    }

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
    if (!step || !scoring_state_) {
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
      if (config_.score_detector_crossings) {
        metrics.crossing_count += 1;
        metrics.current_event_crossed = true;
      }
    }
    if (config_.score_plane_crossings) {
      const auto* post_point = step->GetPostStepPoint();
      if (post_point) {
        const auto pre_z = pre_point->GetPosition().z() / mm;
        const auto post_z = post_point->GetPosition().z() / mm;
        const auto plane_z = scoring_state_->scoring_plane_z_mm;
        const auto crosses_plane =
            ((pre_z < plane_z && post_z >= plane_z) || (pre_z > plane_z && post_z <= plane_z));
        if (crosses_plane) {
          scoring_state_->plane_crossing_count += 1;
          scoring_state_->current_event_plane_crossed = true;
        }
      }
    }
    const auto edep = step->GetTotalEnergyDeposit();
    if (config_.score_target_edep && edep > 0.0) {
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
    scoring_state_->current_event_plane_crossed = false;
    for (auto& [_, metrics] : scoring_state_->volume_stats) {
      metrics.current_event_edep_mev = 0.0;
      metrics.current_event_crossed = false;
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
      if (metrics.current_event_crossed) {
        metrics.crossing_events += 1;
      }
      if (volume_name == scoring_state_->target_volume_name) {
        scoring_state_->target_edep_mev = metrics.edep_mev;
        scoring_state_->current_event_target_edep_mev = metrics.current_event_edep_mev;
        scoring_state_->target_hit_events = metrics.hit_events;
        scoring_state_->target_step_count = metrics.step_count;
        scoring_state_->target_track_entries = metrics.track_entries;
      }
    }
    if (scoring_state_->current_event_plane_crossed) {
      scoring_state_->plane_crossing_events += 1;
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
  CLHEP::HepRandom::setTheSeed(static_cast<long>(cfg.seed));

  auto* run_manager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::SerialOnly);
  run_manager->SetUserInitialization(new RuntimeDetectorConstruction(cfg));

  G4PhysListFactory phys_factory;
  auto* physics = phys_factory.GetReferencePhysList(cfg.physics_list);
  if (!physics) {
    physics = phys_factory.GetReferencePhysList("FTFP_BERT");
  }
  run_manager->SetUserInitialization(physics);
  RuntimeScoringState scoring_state;
  scoring_state.target_volume_name = cfg.root_volume_name;
  scoring_state.scoring_plane_name = cfg.scoring_plane_name;
  scoring_state.scoring_plane_z_mm = cfg.scoring_plane_z_mm;
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
  const auto role_stats = derive_role_stats(scoring_state.volume_stats, cfg.scoring_volume_roles);
  summary << "{\n"
          << "  \"schema_version\": \"" << kResultSchemaVersion << "\",\n"
          << "  \"run_ok\": true,\n"
          << "  \"events_requested\": " << events << ",\n"
          << "  \"events_completed\": " << events << ",\n"
          << "  \"run_seed\": " << cfg.seed << ",\n"
          << "  \"geometry_structure\": \"" << cfg.geometry_structure << "\",\n"
          << "  \"material\": \"" << cfg.material << "\",\n"
          << "  \"particle\": \"" << cfg.particle << "\",\n"
          << "  \"source_type\": \"" << cfg.source_type << "\",\n"
          << "  \"payload_sha256\": \"" << cfg.payload_sha256 << "\",\n"
          << "  \"geant4_version\": \"" << G4Version << "\",\n"
          << "  \"source_position_mm\": [" << cfg.source_x_mm << ", " << cfg.source_y_mm << ", " << cfg.source_z_mm
          << "],\n"
          << "  \"source_direction\": [" << cfg.direction_x << ", " << cfg.direction_y << ", " << cfg.direction_z
          << "],\n"
          << "  \"physics_list\": \"" << cfg.physics_list << "\",\n"
          << "  \"events\": " << events << ",\n"
          << "  \"mode\": \"" << cfg.mode << "\",\n"
          << "  \"run_manifest\": {\n"
          << "    \"bridge\": \"simulation_bridge\",\n"
          << "    \"geometry_root_volume\": \"" << cfg.root_volume_name << "\",\n"
          << "    \"detector_enabled\": " << (cfg.detector_enabled ? "true" : "false") << ",\n"
          << "    \"detector_volume_name\": " << (cfg.detector_enabled ? ("\"" + cfg.detector_name + "\"") : "null") << ",\n"
          << "    \"scoring_plane_name\": " << (cfg.score_plane_crossings ? ("\"" + cfg.scoring_plane_name + "\"") : "null") << ",\n"
          << "    \"scoring_plane_z_mm\": " << (cfg.score_plane_crossings ? std::to_string(cfg.scoring_plane_z_mm) : "null") << ",\n"
          << "    \"scoring_volume_names\": ";
  write_string_array(summary, cfg.scoring_volume_names);
  summary << ",\n"
          << "    \"scoring_roles\": ";
  write_role_map(summary, cfg.scoring_volume_roles);
  summary << "\n"
          << "  },\n"
          << "  \"detector\": {\n"
          << "    \"enabled\": " << (cfg.detector_enabled ? "true" : "false") << ",\n"
          << "    \"volume_name\": \"" << cfg.detector_name << "\",\n"
          << "    \"material\": \"" << cfg.detector_material << "\",\n"
          << "    \"position_mm\": [" << cfg.detector_x_mm << ", " << cfg.detector_y_mm << ", " << cfg.detector_z_mm << "],\n"
          << "    \"size_mm\": [" << cfg.detector_size_x_mm << ", " << cfg.detector_size_y_mm << ", " << cfg.detector_size_z_mm << "]\n"
          << "  },\n"
          << "  \"scoring\": {\n"
          << "    \"target_edep_enabled\": " << (cfg.score_target_edep ? "true" : "false") << ",\n"
          << "    \"detector_crossings_enabled\": " << (cfg.score_detector_crossings ? "true" : "false") << ",\n"
          << "    \"plane_crossings_enabled\": " << (cfg.score_plane_crossings ? "true" : "false") << ",\n"
          << "    \"plane_crossing_name\": " << (cfg.score_plane_crossings ? ("\"" + cfg.scoring_plane_name + "\"") : "null") << ",\n"
          << "    \"plane_crossing_z_mm\": " << (cfg.score_plane_crossings ? std::to_string(cfg.scoring_plane_z_mm) : "null") << ",\n"
          << "    \"plane_crossing_count\": " << scoring_state.plane_crossing_count << ",\n"
          << "    \"plane_crossing_events\": " << scoring_state.plane_crossing_events << ",\n"
          << "    \"detector_crossing_count\": "
          << (role_stats.count("detector") ? role_stats.at("detector").crossing_count : 0) << ",\n"
          << "    \"detector_crossing_events\": "
          << (role_stats.count("detector") ? role_stats.at("detector").crossing_events : 0) << ",\n"
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
            << "        \"crossing_events\": " << metrics.crossing_events << ",\n"
            << "        \"crossing_count\": " << metrics.crossing_count << ",\n"
            << "        \"step_count\": " << metrics.step_count << ",\n"
            << "        \"track_entries\": " << metrics.track_entries << "\n"
            << "      }";
  }
  summary << "\n"
          << "    },\n"
          << "    \"role_stats\": {\n";
  bool first_role = true;
  for (const auto& [role_name, metrics] : role_stats) {
    if (!first_role) {
      summary << ",\n";
    }
    first_role = false;
    summary << "      \"" << role_name << "\": {\n"
            << "        \"edep_total_mev\": " << metrics.edep_mev << ",\n"
            << "        \"edep_mean_mev_per_event\": "
            << (events > 0 ? (metrics.edep_mev / static_cast<double>(events)) : 0.0) << ",\n"
            << "        \"hit_events\": " << metrics.hit_events << ",\n"
            << "        \"crossing_events\": " << metrics.crossing_events << ",\n"
            << "        \"crossing_count\": " << metrics.crossing_count << ",\n"
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
