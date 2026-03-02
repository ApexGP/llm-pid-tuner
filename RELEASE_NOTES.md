# Release v2.0 PRO - 硬件调参专用版

🎉 我们非常高兴地发布 LLM-PID-Tuner v2.0 PRO 的独立可执行版本！

## 📦 这个版本包含什么？

此 Release 包含打包好的 `llm-pid-tuner.exe`，它是专为**真实硬件调参**设计的上位机程序。
*   **无需 Python 环境**：直接双击运行，不再需要安装 Python 或任何依赖库。
*   **交互式串口选择**：程序会自动扫描并列出当前连接的串口设备，您可以轻松选择目标设备。
*   **增强版内核**：内置了最新的 History-Aware 和 Chain-of-Thought 调参逻辑。

## 🚀 快速开始

1.  **下载**：点击下方 Assets 中的 `llm-pid-tuner.exe` 进行下载。
2.  **准备硬件**：将您的单片机（如 Arduino, ESP32）连接到电脑。
3.  **配置 API Key**：
    *   **方法一（推荐）**：在 PowerShell 中设置环境变量后运行：
        ```powershell
        $env:LLM_API_KEY="sk-您的API密钥"
        .\llm-pid-tuner.exe
        ```
    *   **方法二**：如果不想每次都输，可以在 Windows 系统的“环境变量”设置中添加 `LLM_API_KEY`。
4.  **运行**：双击 `llm-pid-tuner.exe`。
5.  **选择串口**：程序会显示可用串口列表，输入对应序号即可连接。

## 📝 注意事项

*   默认波特率为 `115200`。如果您的硬件使用其他波特率，请设置环境变量 `$env:BAUD_RATE="9600"`。
*   默认 API Base URL 为 OpenAI 官方地址。如需使用其他服务商（如 DeepSeek, MiniMax），请设置 `$env:LLM_API_BASE_URL="您的地址"`。

---
*Happy Tuning!* 🎛️
