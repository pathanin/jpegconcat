class Jpegconcat < Formula
  include Language::Python::Virtualenv

  desc "Concatenate JPEG images while preserving original encoding parameters"
  homepage "https://github.com/pathanin/jpegconcat"
  url "https://github.com/pathanin/jpegconcat/releases/download/v0.2.1/jpegconcat-0.2.1.tar.gz"
  sha256 "de40b66f22b2970fd45807c91abe00c0333f00bdd3d54bb04b3085a2362286f4"
  license "MIT"

  depends_on "python@3.12"
  depends_on "jpeg-turbo"

  # Pre-built wheels — fast install, no compilation
  on_arm do
    resource "pillow" do
      url "https://files.pythonhosted.org/packages/d8/95/0a351b9289c2b5cbde0bacd4a83ebc44023e835490a727b2a3bd60ddc0f4/pillow-12.2.0-cp312-cp312-macosx_11_0_arm64.whl"
      sha256 "f3f40b3c5a968281fd507d519e444c35f0ff171237f4fdde090dd60699458421"
    end

    resource "numpy" do
      url "https://files.pythonhosted.org/packages/ea/12/92c4c131527599e8288d6918e888d88726f84d805d784b771f32408aeaef/numpy-2.4.6-cp312-cp312-macosx_11_0_arm64.whl"
      sha256 "ebfb099f8dcf083deef3ac1ca4c1503f387cf76296fcb3816b66f5ecb5f54fdb"
    end
  end

  on_intel do
    resource "pillow" do
      url "https://files.pythonhosted.org/packages/58/be/7482c8a5ebebbc6470b3eb791812fff7d5e0216c2be3827b30b8bb6603ed/pillow-12.2.0-cp312-cp312-macosx_10_13_x86_64.whl"
      sha256 "2d192a155bbcec180f8564f693e6fd9bccff5a7af9b32e2e4bf8c9c69dbad6b5"
    end

    resource "numpy" do
      url "https://files.pythonhosted.org/packages/95/2a/3d7b5ac8aac24feaf9ad7ed58f45b0bbc06d37e4338ae84c9f2298b570f9/numpy-2.4.6-cp312-cp312-macosx_10_13_x86_64.whl"
      sha256 "001fbb8e08d942dd57599e781f2472269ee7f2755fae407b4f67b2f0b17da3f1"
    end
  end

  # Source fallbacks — used if a pre-built wheel is unavailable
  resource "pillow-sdist" do
    url "https://files.pythonhosted.org/packages/8c/21/c2bcdd5906101a30244eaffc1b6e6ce71a31bd0742a01eb89e660ebfac2d/pillow-12.2.0.tar.gz"
    sha256 "a830b1a40919539d07806aa58e1b114df53ddd43213d9c8b75847eee6c0182b5"
  end

  resource "numpy-sdist" do
    url "https://files.pythonhosted.org/packages/d0/ad/fed0499ce6a338d2a03ebae59cd15093910c8875328855781952abf6c2fe/numpy-2.4.6.tar.gz"
    sha256 "f3a3570c4a2a16746ac2c31a7c7c7b0c186b95ce902e33db6f28094ed7387dda"
  end

  def install
    venv = virtualenv_create(libexec, "python3.12")

    ["pillow", "numpy"].each do |pkg|
      whl_src = resource(pkg).cached_download
      # Copy wheel to buildpath with a clean filename — the Homebrew 6.0 sandbox
      # blocks exec of scripts inside the keg, so we use the system python3.12
      # (sandbox-allowed) with --python targeting the venv instead of libexec/bin/pip.
      wheel_name = whl_src.basename.to_s.sub(/\A[0-9a-f]+-+/, "")
      whl_dst = buildpath/wheel_name
      cp whl_src, whl_dst
      begin
        system "python3.12", "-m", "pip",
               "--python=#{libexec}/bin/python",
               "install", "--no-deps", "--no-compile", whl_dst
      rescue BuildError => e
        opoo "Pre-built #{pkg} wheel failed (#{e.message}), building from source..."
        venv.pip_install resource("#{pkg}-sdist")
      end
    end

    libexec.install "concat_jpeg.py"

    (bin/"jpegconcat").write <<~EOS
      #!/bin/bash
      exec "#{libexec}/bin/python" -B "#{libexec}/concat_jpeg.py" "$@"
    EOS
    (bin/"jpegconcat").chmod 0755
  end

  test do
    system bin/"jpegconcat", "--help"
  end
end
