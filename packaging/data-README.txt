HalalifyVid keeps everything it installs in this folder:

  bin\      ONNX Runtime and, on NVIDIA machines, the CUDA libraries
  models\   the neural network models
  cache\    small caches; downloads in progress
  work\     temporary files of a running job

Because this folder sits next to the program, the whole release folder is
self-contained: move or copy it anywhere, or delete it to uninstall.
Delete only this "data" folder to make HalalifyVid use the per-user location
instead (%LOCALAPPDATA%\HalalifyVid on Windows, ~/.local/share/HalalifyVid
on Linux).
