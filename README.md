# 🎙️ 开源播客音频工作台

一个面向播客、访谈、配音和会议录音的本地音频处理套件。项目目标是提供类似 Adobe Speech Enhance 的开源工作流：上传语音，选择增强方案，在本机完成降噪、减弱混响、语音修复和带宽扩展。

在常规语音增强之外，项目还提供独立的 **声场匹配** 功能：使用参考音频统一不同录音的频率响应、响度、峰值和立体声宽度。

> 本项目与 Adobe 没有隶属、授权或合作关系。“Adobe Speech Enhance”仅用于说明产品定位和使用场景。

## 当前状态

项目目前是可运行原型，中文界面、三条增强方案和声场匹配入口已经完成。代码审查发现的采样率衔接、输出重名和核心依赖漂移问题已经修复；正式发布前仍建议完成真实音频的端到端回归测试，并确定项目整体许可证。

| 类别 | 引擎 | 当前用途 | 输出 |
| --- | --- | --- | --- |
| 音频增强 | ClearVoice + UniverSR | 强降噪、减弱混响；仅对 8/12/16/24 kHz 输入继续执行带宽扩展 | 48 kHz、24-bit WAV |
| 音频增强 | LavaSR | 快速语音降噪与带宽扩展 | 48 kHz、24-bit WAV |
| 音频增强 | Sidon | 多语言语音修复、长音频分块处理 | 单声道 48 kHz、24-bit WAV |
| 声场匹配 | Matchering | 让目标音频接近参考音频的频响、RMS、峰值和立体声宽度 | 24-bit WAV |

## 功能定位

### 类似 Adobe Speech Enhance 的语音增强

- 降低持续底噪、环境噪声和部分瞬态噪声。
- 减弱房间混响，突出人声主体。
- 修复低带宽、低采样率或受损语音，并统一输出为 48 kHz。
- 提供多个开源模型，允许用户根据质量、速度和语言选择处理方案。
- 音频推理在本机执行；首次使用模型时仍需联网下载权重。

增强效果依赖录音内容。严重削波、多人重叠说话、音乐占比很高或极端混响不保证可以完全恢复，也不能把不存在的真实语音细节无损还原。

### 独有功能：声场匹配

声场匹配使用 [Matchering](https://github.com/sergree/matchering)：

1. 上传需要处理的目标音频。
2. 上传一段声音风格合适的参考音频。
3. Matchering 根据参考音频调整目标音频的频率响应、RMS 响度、峰值和立体声宽度。
4. 输出 24-bit WAV。

这里的“声场匹配”是本项目的产品功能名称，本质上是参考音频匹配与自动母带处理；它不是房间脉冲响应复制，也不是环绕声、声源定位或三维空间音频模拟。

## 音频增强方案

### 🧹 ClearVoice + UniverSR（两阶段）

用于底噪和混响都比较严重的录音：

1. [ClearVoice / MossFormer2](https://github.com/modelscope/ClearerVoice-Studio) 负责语音增强和去混响。
2. 当原始输入采样率为 8/12/16/24 kHz 时，[UniverSR](https://github.com/woongzip1/UniverSR) 继续将低带宽结果超分辨率到 48 kHz。
3. 当原始输入为 44.1/48 kHz 等非 UniverSR 支持档位时，自动跳过超分辨率，保留 ClearVoice 的增强结果并规范化为 48 kHz、24-bit WAV。

### 🌋 LavaSR（快速）

[LavaSR](https://github.com/ysharma3501/LavaSR) 在单次推理中完成降噪和带宽扩展。上游项目声明支持 8–48 kHz 输入，输出为 48 kHz，适合需要较快处理速度的语音录音。

### 🐋 Sidon（多语言语音修复）

[Sidon](https://huggingface.co/spaces/sarulab-speech/sidon_demo_beta) 使用官方 CPU 或 CUDA TorchScript 权重。当前实现会：

- 读取输入文件的真实采样率；
- 将立体声下混为单声道；
- 在模型内部重采样到 16 kHz；
- 以最长约 96 秒的分块进行修复；
- 输出单声道 48 kHz、24-bit WAV。

## 采样率和格式

- **44.1 kHz 语音可以处理**，但不同方案的行为并不相同。
- LavaSR 上游声明支持 8–48 kHz 输入。
- Sidon 会把 SoundFile 能解码的输入重采样到模型需要的 16 kHz，再生成 48 kHz 单声道结果。
- UniverSR 只接受 8、12、16、24 kHz 的有效输入带宽；程序会根据原始输入采样率决定是否调用它，不会再把 44.1 或 48 kHz 错误传入。
- 所有语音增强方案当前统一写出 48 kHz、24-bit WAV。
- WAV 和 FLAC 通常可直接读取。MP3、M4A、AAC、Opus 等压缩格式是否可用取决于上游库和系统 FFmpeg/编解码器配置。

## 安装

### 环境要求

- Python 3.10 或更高版本。
- 建议使用 NVIDIA GPU；CPU 可以运行部分方案，但速度和内存占用取决于模型及音频长度。
- 当前 `requirements.txt` 使用 PyTorch CUDA 12.8 额外索引。CPU、macOS 或其他 CUDA 版本应按照 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)调整 Torch/Torchaudio 安装方式。
- 如需处理更多压缩音频格式，请安装 FFmpeg。

### 安装步骤

```bash
python -m venv venv
```

Windows：

```powershell
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS / Linux：

```bash
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

依赖中包含直接从 GitHub 安装的 UniverSR 和 LavaSR，因此安装阶段需要网络和 Git。

## 使用

启动中文 Web 界面：

```bash
python app.py
```

默认访问地址为 `http://127.0.0.1:7860`。

### 音频增强

1. 打开“音频增强”。
2. 上传或录制待增强语音。
3. 选择增强方案。
4. 点击“开始增强”。
5. 试听并保存 48 kHz 结果。

### 声场匹配

1. 打开“声场匹配”。
2. 上传待处理音频。
3. 上传参考音频。
4. 点击“开始匹配”。
5. 试听并保存 24-bit WAV 结果。

处理结果默认写入 `temp_output/`。每个文件名都包含随机任务标识，避免同名文件和并发任务相互覆盖；该目录已加入 `.gitignore`。

## 已知限制与发布前检查

### 1. 尚未完成完整端到端测试

目前已通过 Python 编译、管线路由和 Matchering 调用适配检查，但尚未在同一个正式环境中完成三种模型的真实音频回归、GPU/CPU 对照、长音频压力测试和主流格式兼容测试。

### 2. 当前依赖锁定面向 CUDA 12.8

核心 PyPI 依赖和两个 Git 模型仓库已经锁定到审查时的版本/提交。CPU、macOS 或其他 CUDA 版本仍需要单独建立并验证安装矩阵，更新锁定版本时应重新执行真实音频回归。

### 3. 当前没有自动质量评估

界面不会自动检测削波、静音、语音占比、处理伪影或增强前后的客观质量。正式产品可考虑增加响度、峰值、DNSMOS/SRMR 等检测，但这些指标不能替代人工试听。

## 项目结构

```text
.
├── app.py             # 中文 Gradio 界面
├── processor.py       # 三条语音增强管线及 Matchering 适配
├── requirements.txt   # Python 与模型依赖
└── README.md          # 项目说明、限制和发布检查
```

## 上游项目

- [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)
- [UniverSR](https://github.com/woongzip1/UniverSR)
- [LavaSR](https://github.com/ysharma3501/LavaSR)
- [Sidon 模型](https://huggingface.co/sarulab-speech/sidon-v0.1)与[官方演示](https://huggingface.co/spaces/sarulab-speech/sidon_demo_beta)
- [Matchering](https://github.com/sergree/matchering)
- [Gradio](https://github.com/gradio-app/gradio)

## 许可证说明

当前仓库**没有 `LICENSE` 文件**，因此尚未正式声明本项目代码的许可证，不应显示 MIT 徽章或直接声称整个项目采用 MIT。

各上游代码和模型继续受各自许可证约束。尤其需要注意：Matchering 使用 GPL-3.0；Sidon 模型页面标记为 MIT；ClearerVoice-Studio 和 LavaSR 使用 Apache-2.0；UniverSR 使用 MIT。对外分发、打包或商业使用前，请逐项核对当时的上游许可证，并确定本项目整体采用的兼容许可证。
