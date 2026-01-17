# Homebrew tap formula (HEAD-only for now).
class Beatit < Formula
  desc "Minimal BPM/beat tracker using CoreML"
  homepage "https://github.com/tillt/BeatIt"
  license "MIT"

  head "https://github.com/tillt/BeatIt.git", branch: "main"

  depends_on :macos
  depends_on "cmake" => :build
  depends_on "ninja" => :build

  def install
    system "cmake", "-S", ".", "-B", "build", "-G", "Ninja",
           "-DCMAKE_BUILD_TYPE=Release"
    system "cmake", "--build", "build", "--target", "beatit"
    system "cmake", "--build", "build", "--target", "beatit_lib"

    bin.install "build/beatit"
    lib.install "build/libbeatit_lib.a" if File.exist?("build/libbeatit_lib.a")
    lib.install "build/libbeatit_lib.dylib" if File.exist?("build/libbeatit_lib.dylib")
    include.install Dir["include/beatit/*.h"]
    share.install "models/beatit.mlmodelc"
  end

  test do
    system "#{bin}/beatit", "--help"
  end
end
