class Jpegconcat < Formula
  include Language::Python::Virtualenv

  desc "Concatenate JPEG images while preserving original encoding parameters"
  homepage "https://github.com/pathanin/jpegconcat"
  url "https://github.com/pathanin/jpegconcat/releases/download/v0.1.1/jpegconcat-0.1.1.tar.gz"
  sha256 "60a63da913de5304a892b7335f7af39a17204c2be376e3249a229724e582e25c"
  license "MIT"

  depends_on "python@3.12"
  depends_on "jpeg-turbo"

  resource "pillow" do
    url "https://files.pythonhosted.org/packages/8c/21/c2bcdd5906101a30244eaffc1b6e6ce71a31bd0742a01eb89e660ebfac2d/pillow-12.2.0.tar.gz"
    sha256 "a830b1a40919539d07806aa58e1b114df53ddd43213d9c8b75847eee6c0182b5"
  end

  resource "numpy" do
    url "https://files.pythonhosted.org/packages/d0/ad/fed0499ce6a338d2a03ebae59cd15093910c8875328855781952abf6c2fe/numpy-2.4.6.tar.gz"
    sha256 "f3a3570c4a2a16746ac2c31a7c7c7b0c186b95ce902e33db6f28094ed7387dda"
  end

  def install
    venv = virtualenv_create(libexec, "python3.12")
    venv.pip_install resources

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
