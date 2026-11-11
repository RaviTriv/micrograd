#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "micrograd/NN.h"
#include "micrograd/Tensor.h"

using namespace micrograd;

namespace {

class Pair : public nn::Module {
 public:
  Pair(std::shared_ptr<Tensor> first, std::shared_ptr<Tensor> second) {
    register_parameter("first", std::move(first));
    register_parameter("second", std::move(second));
  }

  std::shared_ptr<Tensor> forward(
      const std::shared_ptr<Tensor> &input) override {
    return input;
  }
};

std::vector<char> raw_bytes(const Tensor &tensor) {
  const auto *begin =
      static_cast<const char *>(tensor.data_storage().host_pointer());
  return {begin, begin + tensor.data_storage().bytes()};
}

std::vector<std::vector<char>> snapshot(const nn::Module &module) {
  std::vector<std::vector<char>> result;
  for (const auto &[name, tensor] : module.named_parameters()) {
    result.push_back(raw_bytes(*tensor));
  }
  return result;
}

std::vector<char> read_file(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}

void write_file(const std::string &path, const std::vector<char> &bytes) {
  std::ofstream file(path, std::ios::binary);
  file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

template <typename T>
void put(std::ofstream &file, T value) {
  file.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void write_version1(const std::string &path, const nn::Module &module) {
  std::ofstream file(path, std::ios::binary);
  const auto named = module.named_parameters();
  put<uint32_t>(file, 0x4d47534e);
  put<uint32_t>(file, 1);
  put<uint32_t>(file, static_cast<uint32_t>(named.size()));
  for (const auto &[name, tensor] : named) {
    put<uint32_t>(file, static_cast<uint32_t>(name.size()));
    file.write(name.data(), static_cast<std::streamsize>(name.size()));
    put<uint32_t>(file, static_cast<uint32_t>(tensor->shape().size()));
    for (auto dim : tensor->shape()) {
      put<uint64_t>(file, dim);
    }
    const auto values = tensor->data();
    file.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size_bytes()));
  }
}

std::string load_error(const std::string &path, nn::Module &module) {
  try {
    load(path, module);
  } catch (const std::runtime_error &error) {
    return error.what();
  }
  return "";
}

}  // namespace

TEST(Serialization, SaveLoadRoundTrip) {
  const std::string path = ::testing::TempDir() + "round_trip.bin";
  Linear saved(3, 2);
  save(path, saved);

  Linear loaded(3, 2);
  load(path, loaded);

  EXPECT_TRUE(
      std::ranges::equal(loaded.weights()->data(), saved.weights()->data()));
  EXPECT_TRUE(std::ranges::equal(loaded.bias()->data(), saved.bias()->data()));
}

TEST(Serialization, LoadsVersion1File) {
  const std::string path = ::testing::TempDir() + "version1.bin";
  Linear saved(3, 2);
  write_version1(path, saved);

  Linear loaded(3, 2);
  load(path, loaded);

  EXPECT_EQ(snapshot(loaded), snapshot(saved));
  EXPECT_TRUE(load_metadata(path).empty());
}

TEST(Serialization, Bfloat16RoundTrip) {
  const std::string path = ::testing::TempDir() + "bf16.bin";
  Pair saved(
      Tensor::from_values({2}, {1.5f, -2.0f}),
      Tensor::from_values({3}, {0.1f, 7.0f, -3.25f})->cast(DType::kBFloat16));
  save(path, saved);

  Pair loaded(Tensor::zeros({2}), Tensor::zeros({3})->cast(DType::kBFloat16));
  load(path, loaded);

  EXPECT_EQ(loaded.named_parameters()[1].second->dtype(), DType::kBFloat16);
  EXPECT_EQ(snapshot(loaded), snapshot(saved));
}

TEST(Serialization, DtypeMismatchNamesParameterAndDtypes) {
  const std::string path = ::testing::TempDir() + "dtype_mismatch.bin";
  Pair saved(Tensor::zeros({2}), Tensor::zeros({3})->cast(DType::kBFloat16));
  save(path, saved);

  Pair loaded(Tensor::from_values({2}, {4, 5}), Tensor::zeros({3}));
  const auto before = snapshot(loaded);
  const std::string error = load_error(path, loaded);

  EXPECT_NE(error.find("dtype mismatch for parameter second"),
            std::string::npos)
      << error;
  EXPECT_NE(error.find("model has float32, file has bfloat16"),
            std::string::npos)
      << error;
  EXPECT_EQ(snapshot(loaded), before);
}

TEST(Serialization, TruncatedFileLeavesParametersUnchanged) {
  const std::string full = ::testing::TempDir() + "full.bin";
  const std::string truncated = ::testing::TempDir() + "truncated.bin";
  Linear saved(3, 2);
  save(full, saved);
  auto bytes = read_file(full);
  bytes.resize(bytes.size() - 4);
  write_file(truncated, bytes);

  Linear loaded(3, 2);
  const auto before = snapshot(loaded);
  EXPECT_THROW(load(truncated, loaded), std::runtime_error);
  EXPECT_EQ(snapshot(loaded), before);
}

TEST(Serialization, ShapeMismatchOnLastTensorLeavesParametersUnchanged) {
  const std::string path = ::testing::TempDir() + "shape_mismatch.bin";
  Pair saved(Tensor::from_values({2}, {1, 2}),
             Tensor::from_values({3}, {3, 4, 5}));
  save(path, saved);

  Pair loaded(Tensor::from_values({2}, {9, 9}),
              Tensor::from_values({4}, {9, 9, 9, 9}));
  const auto before = snapshot(loaded);
  const std::string error = load_error(path, loaded);

  EXPECT_NE(error.find("shape mismatch for parameter second: model has [4], "
                       "file has [3]"),
            std::string::npos)
      << error;
  EXPECT_EQ(snapshot(loaded), before);
}

TEST(Serialization, LoadRejectsTrailingData) {
  const std::string path = ::testing::TempDir() + "trailing.bin";
  Linear saved(3, 2);
  save(path, saved);
  auto bytes = read_file(path);
  bytes.push_back('x');
  write_file(path, bytes);

  Linear loaded(3, 2);
  EXPECT_THROW(load(path, loaded), std::runtime_error);
}

TEST(Serialization, FailedSaveKeepsPreviousFile) {
  namespace fs = std::filesystem;
  const fs::path directory = fs::path(::testing::TempDir()) / "read_only_dir";
  fs::remove_all(directory);
  fs::create_directories(directory);
  const std::string path = (directory / "model.bin").string();

  Linear original(3, 2);
  save(path, original);
  fs::permissions(directory, fs::perms::owner_read | fs::perms::owner_exec);

  Linear replacement(3, 2);
  EXPECT_THROW(save(path, replacement), std::runtime_error);
  fs::permissions(directory, fs::perms::owner_all);

  Linear loaded(3, 2);
  load(path, loaded);
  EXPECT_EQ(snapshot(loaded), snapshot(original));
}

TEST(Serialization, FailedRenameRemovesTemporaryFile) {
  namespace fs = std::filesystem;
  const fs::path target = fs::path(::testing::TempDir()) / "target_is_dir";
  fs::remove_all(target);
  fs::create_directories(target / "child");

  Linear layer(3, 2);
  EXPECT_THROW(save(target.string(), layer), std::runtime_error);
  EXPECT_FALSE(fs::exists(target.string() + ".tmp"));
  EXPECT_TRUE(fs::is_directory(target / "child"));
  fs::remove_all(target);
}

TEST(Serialization, SuccessfulSaveLeavesNoTemporaryFile) {
  const std::string path = ::testing::TempDir() + "no_temp.bin";
  Linear layer(3, 2);
  save(path, layer);
  EXPECT_TRUE(std::filesystem::exists(path));
  EXPECT_FALSE(std::filesystem::exists(path + ".tmp"));
}

TEST(Serialization, MetadataRoundTrip) {
  const std::string path = ::testing::TempDir() + "metadata.bin";
  const StateDictMetadata metadata = {{"n_layer", "12"}, {"arch", "gpt"}};
  Linear saved(3, 2);
  save(path, saved, metadata);

  EXPECT_EQ(load_metadata(path), metadata);
  Linear loaded(3, 2);
  load(path, loaded);
  EXPECT_EQ(snapshot(loaded), snapshot(saved));
}

TEST(Serialization, ReadsMetadataWithoutTensors) {
  const std::string path = ::testing::TempDir() + "metadata_only.bin";
  const StateDictMetadata metadata = {{"arch", "gpt"}};
  Linear saved(3, 2);
  save(path, saved, metadata);
  auto bytes = read_file(path);
  bytes.resize(3 * sizeof(uint32_t) + 2 * sizeof(uint32_t) + 4 + 3);
  write_file(path, bytes);

  EXPECT_EQ(load_metadata(path), metadata);
  Linear loaded(3, 2);
  EXPECT_THROW(load(path, loaded), std::runtime_error);
}

TEST(Serialization, SaveWithoutMetadataHasEmptyMetadata) {
  const std::string path = ::testing::TempDir() + "empty_metadata.bin";
  Linear saved(3, 2);
  save(path, saved);
  EXPECT_TRUE(load_metadata(path).empty());
}

TEST(Serialization, LoadRejectsShapeLargerThanFile) {
  const std::string path = ::testing::TempDir() + "huge_shape.bin";
  {
    std::ofstream file(path, std::ios::binary);
    const std::array<uint32_t, 4> header = {0x4d47534e, 1, 1, 1};
    file.write(reinterpret_cast<const char *>(header.data()), sizeof(header));
    file.write("w", 1);
    const uint32_t ndim = 1;
    file.write(reinterpret_cast<const char *>(&ndim), sizeof(ndim));
    const uint64_t huge_dim = uint64_t{1} << 40;
    file.write(reinterpret_cast<const char *>(&huge_dim), sizeof(huge_dim));
  }

  Linear layer(3, 2);
  EXPECT_THROW(load(path, layer), std::runtime_error);
}
