@echo off
chcp 65001 > nul
echo 安装Gemini节点依赖...

set "PYTHON_EXE=C:\Users\6\Desktop\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\python_embeded\python.exe"

echo 使用Python: %PYTHON_EXE%
echo.

echo 📦 安装requests...
"%PYTHON_EXE%" -m pip install requests

echo 📦 安装Pillow...
"%PYTHON_EXE%" -m pip install Pillow

echo.
echo ================================
echo ✅ 依赖安装完成！
echo ================================
echo.
echo 下一步：
echo 1. 重启ComfyUI
echo 2. 右键 → Add Node → image → ai_generation → 🤖 Gemini图像生成器
echo.
pause