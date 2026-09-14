#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "examples/gpt/BPE.h"

namespace {

using micrograd::gpt::BPE;

std::vector<std::string> Corpus() {
  return {
      "the quick brown fox jumps over the lazy dog",
      "the quick brown fox jumps over the lazy dog again and again",
      "she sells seashells by the seashore, and the shells she sells "
      "are surely seashells",
      "to be or not to be, that is the question",
  };
}

std::vector<std::string> Samples() {
  return {
      "the quick brown fox",
      "seashells by the seashore",
      "an unseen phrase with punctuation!? and numbers 123 456",
      "",
      "\n\ttabs and newlines\n",
      "unicode: caf\xc3\xa9 na\xc3\xafve",
  };
}

size_t RunAllSamples(const BPE &bpe) {
  size_t failed = 0;
  for (const auto &sample : Samples()) {
    const std::string decoded = bpe.decode(bpe.encode(sample));
    if (decoded == sample) {
      std::printf("ok    roundtrip %s\n", sample.c_str());
      continue;
    }
    failed++;
    std::printf("FAIL  roundtrip %s -> %s\n", sample.c_str(), decoded.c_str());
  }
  return failed;
}

}  // namespace

int main() {
  try {
    BPE bpe = BPE::train(Corpus(), 300);
    if (bpe.vocab_size() <= 256) {
      std::printf("FAIL  training did not grow the vocabulary\n");
      return 1;
    }

    const size_t failed = RunAllSamples(bpe);
    std::printf("%zu sample(s) failed\n", failed);
    return failed == 0 ? 0 : 1;
  } catch (const std::exception &error) {
    std::printf("bpe_roundtrip raised: %s\n", error.what());
    return 1;
  }
}
