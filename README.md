# paft-run

## QNN/HTP Setup (QCS6490 / RubikPi 3)

### Prerequisites

```bash
sudo apt install qcom-fastrpc1 qcom-fastrpc-dev
```

### Build ONNX Runtime with QNN

```bash
git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime && git checkout v1.23.2
pip3 install -r requirements-dev.txt
./build.sh --use_qnn --qnn_home ~/qnn-sdk --config Release --build_shared_lib --parallel --skip_tests
```

### Install Libraries

```bash
sudo cp build/Linux/Release/libonnxruntime.so* /usr/local/lib/
sudo cp build/Linux/Release/libonnxruntime_providers_shared.so /usr/local/lib/
sudo cp build/Linux/Release/libonnxruntime_providers_qnn.so /usr/local/lib/
sudo ln -sf /usr/local/lib/libonnxruntime.so.1.23.2 /usr/local/lib/libonnxruntime.so
sudo ldconfig

sudo mkdir -p /usr/lib/rfsa/adsp
sudo cp ~/qnn-sdk/lib/hexagon-v68/unsigned/*.so /usr/lib/rfsa/adsp/
```

### Device Permissions

```bash
sudo chmod 666 /dev/fastrpc-cdsp /dev/fastrpc-cdsp-secure /dev/fastrpc-adsp-secure
```

### Environment

```bash
export LD_LIBRARY_PATH=~/qnn-sdk/lib/aarch64-ubuntu-gcc9.4:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=/usr/lib/rfsa/adsp
```

### Validate

```bash
~/qnn-sdk/bin/aarch64-ubuntu-gcc9.4/qnn-platform-validator --backend dsp --testBackend
```

### Audio Visualization

```bash
cava
```

### Audio Playback 
```bash 
paplay <file>
```
