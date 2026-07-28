# CH1 note

- [CH1 note](#ch1-note)
  - [SSL 憑證](#ssl-憑證)
  - [CPU CUDA](#cpu-cuda)


## SSL 憑證
在 main.py 前面加上一段 code

## CPU CUDA
一開始載到的CUDA是CPU。

先 pip uninstall torch torchvision torchaudio -y

卸載 CPU 的 CUDA

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 (下載的有點久，2.4GB)

安裝 GPU 的就可以解決。

後來解除安裝 刪掉再重新安裝好幾次 終於弄好了 目前可以正常跑 [gemini對話過程](https://share.gemini.google/tgca18gu891s)

