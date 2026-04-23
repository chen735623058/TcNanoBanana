import json
import requests
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from typing import List, Dict, Optional
import re
import time
from enum import Enum
import random

class ResolutionOptions(Enum):
    # 枚举成员名称（内部标识）建议与显示文本对应，方便理解
    RES_1K = "1K"   # 成员名称：RES_1K，UI显示："1K"
    RES_2K = "2K"   # 成员名称：RES_2K，UI显示："2K"
    RES_4K = "4K"   # 成员名称：RES_4K，UI显示："4K"

class AspectRatioOptions(Enum):
    # 枚举值（value）为 UI 显示的宽高比文本，成员名称（name）为内部标识（建议清晰易懂）
    SQUARE = "1:1"         # 正方形
    PORTRAIT_2_3 = "2:3"   # 竖屏 2:3（如手机）
    PORTRAIT_3_4 = "3:4"   # 竖屏 3:4
    PORTRAIT_4_5 = "4:5"   # 竖屏 4:5（如社交媒体）
    PORTRAIT_5_4 = "5:4"   # 竖屏 5:4
    LANDSCAPE_3_2 = "3:2"  # 横屏 3:2（如相机）
    LANDSCAPE_4_3 = "4:3"  # 横屏 4:3（如传统显示器）
    LANDSCAPE_16_9 = "16:9" # 横屏 16:9（如高清视频）
    LANDSCAPE_21_9 = "21:9" # 横屏 21:9（如超宽屏）
    LANDSCAPE_9_16 = "9:16" # 横屏 9:16（注意：9:16 实际是竖屏，此处仅为示例分类）
class GeminiOpenAIProxyNode:
    """
    ComfyUI节点: Gemini图像生成 - 非流式模式，支持多张图片和种子
    """

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "text")
    FUNCTION = "generate_images"
    CATEGORY = "image/ai_generation"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "place-your-key-here"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Hello nano banana!"
                }),
                "model_type": (
                    [
                        "nanobanana2-art-gufeng",
                        "nanobanana2-art-xiandai",
                        "nanobanana-art-xiandai",
                        "nanobanana-art-gufeng",
                        "nanobanana2-art-gufeng-flash",
                        "nanobanana2-art-xiandai-flash",
                        "chatgptImage"
                    ],
                    {
                        "default": "nanobanana2-art-gufeng",
                        "tooltip": "选择图像生成模型\n【NanoBanana系列】使用 aspect_ratio 和 resolution 参数\n【ChatGPT】使用 chatgpt_size 和 chatgpt_quality 参数"
                    }
                )
            },
            "optional": {
                # ===== 输入图像 =====
                "input_image_1": ("IMAGE",),
                "input_image_2": ("IMAGE",),
                "input_image_3": ("IMAGE",),
                "input_image_4": ("IMAGE",),
                "input_image_5": ("IMAGE",),

                # ===== NanoBanana 模型专用参数 =====
                "nanobanana_aspect_ratio": (
                    [option.value for option in AspectRatioOptions],
                    {
                        "default": AspectRatioOptions.LANDSCAPE_16_9.value,
                        "tooltip": "❗ 仅用于 NanoBanana 模型\n选择宽高比 (例如: 16:9, 1:1)\nChatGPT 模型请忽略此参数"
                    }
                ),
                "nanobanana_resolution": (
                    [option.value for option in ResolutionOptions],
                    {
                        "default": ResolutionOptions.RES_1K.value,
                        "tooltip": "❗ 仅用于 NanoBanana 模型\n选择分辨率 (1K/2K/4K)\nChatGPT 模型请忽略此参数"
                    }
                ),

                # ===== ChatGPT 模型专用参数 =====
                "chatgpt_size": (
                    [
                        "auto",
                        "1024x1024",
                        "1536x1024",
                        "1024x1536",
                        "2048x2048",
                        "2048x1152",
                        "3840x2160",
                        "2160x3840"
                    ],
                    {
                        "default": "auto",
                        "tooltip": "❗ 仅用于 ChatGPT 模型\n选择生成图片的尺寸大小\nNanoBanana 模型请忽略此参数"
                    }
                ),
                "chatgpt_quality": (
                    [
                        "auto",
                        "low",
                        "medium",
                        "high"
                    ],
                    {
                        "default": "auto",
                        "tooltip": "❗ 仅用于 ChatGPT 模型\n选择生成图片的质量等级\nNanoBanana 模型请忽略此参数"
                    }
                ),
                "seed": ("INT", {
                    "default": 1024, "min": -1, "max": 102400
                }),
            }
        }
    
    def tensor_to_base64(self, tensor: torch.Tensor) -> str:
        """将tensor转换为base64"""
        img_array = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def base64_to_tensor(self, base64_strings: List[str]) -> torch.Tensor:
        """将base64转换为tensor，并确保所有图像尺寸一致 (以最后一张为基准)"""
        pil_images = []

        # 第一步：先将所有 base64 解析为 PIL Image 对象，过滤掉损坏的数据
        for b64_str in base64_strings:
            try:
                img_data = base64.b64decode(b64_str)
                img = Image.open(BytesIO(img_data)).convert('RGB')
                pil_images.append(img)
            except Exception as e:
                print(f"解析 Base64 图像时出错并跳过: {e}")
                continue

        # 兜底：如果所有图片都解析失败，返回一张 64x64 的黑图防止节点崩溃
        if not pil_images:
            print("警告: 没有成功解析出任何有效图像，输出默认黑图")
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        # 第二步：获取列表里最后一张图的尺寸作为全局基准
        target_size = pil_images[-1].size
        if len(pil_images) > 1:
            print(f"设定基准尺寸为最后一张图的尺寸: {target_size}")

        # 确定高质量缩放算法 (兼容不同版本的 Pillow 库)
        if hasattr(Image, 'Resampling'):
            resample_filter = Image.Resampling.LANCZOS
        else:
            resample_filter = Image.LANCZOS

        # 第三步：统一尺寸并转换为 numpy 数组
        images = []
        for img in pil_images:
            # 如果某张图片的尺寸与最后一张的基准尺寸不一致，则进行强制缩放
            if img.size != target_size:
                print(f"节点警告: 发现尺寸不一致的图像 ({img.size})，正在强制缩放至基准尺寸 ({target_size})")
                img = img.resize(target_size, resample_filter)

            # 转为 ComfyUI 需要的 Float32 格式 (0.0 到 1.0 之间)
            img_array = np.array(img).astype(np.float32) / 255.0
            images.append(img_array)

        return torch.from_numpy(np.stack(images))
    
    def create_request_data(self, prompt: str, model_type: str, nanobanana_aspect_ratio: str, nanobanana_resolution: str,
                           input_images: List[torch.Tensor] = None, chatgpt_size: str = None,
                           chatgpt_quality: str = None) -> Dict:
        """构建请求数据

        支持两种模型类型的不同参数配置：
        - NanoBanana 模型: 使用 nanobanana_aspect_ratio 和 nanobanana_resolution 参数
        - ChatGPT 模型: 使用 chatgpt_size 和 chatgpt_quality 参数
        """
        final_prompt = prompt

        # ===== ChatGPT 模型专用处理 =====
        if model_type == "chatgptImage":
            # ChatGPT 参数直接使用
            size = chatgpt_size if chatgpt_size else "auto"
            quality = chatgpt_quality if chatgpt_quality else "auto"

            print(f"ChatGPT Size: {size}")
            print(f"ChatGPT Quality: {quality}")
            print(f"使用 ChatGPT 模型配置: size={size}, quality={quality}")

            # ChatGPT 也使用相同的结构，只是 generationConfig 参数不同
            parts = [{"text": final_prompt}]

            # 添加输入图像
            if input_images:
                for image_tensor in input_images:
                    if image_tensor is not None:
                        base64_image = self.tensor_to_base64(image_tensor)
                        parts.append({
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": base64_image
                            }
                        })

            generation_config = {
                "responseModalities": ["TEXT", "IMAGE"],
                "temperature": 0.8,
                "maxOutputTokens": 8192,
                "size": size,
                "quality": quality
            }

            return {
                "model": model_type,
                "data": {
                    "contents": [{"role": "user", "parts": parts}],
                    "generationConfig": generation_config
                }
            }

        # ===== NanoBanana 模型专用处理 =====
        # 这些模型使用 nanobanana_aspect_ratio 和 nanobanana_resolution 参数
        print(f"使用 NanoBanana 模型配置: aspect_ratio={nanobanana_aspect_ratio}, resolution={nanobanana_resolution}")

        parts = [{"text": final_prompt}]

        # 添加输入图像
        if input_images:
            for image_tensor in input_images:
                if image_tensor is not None:
                    base64_image = self.tensor_to_base64(image_tensor)
                    parts.append({
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": base64_image
                        }
                    })

        generation_config = {
            "responseModalities": ["TEXT", "IMAGE"],
            "temperature": 0.8,
            "maxOutputTokens": 8192,
            "aspect_ratio": nanobanana_aspect_ratio,
            "resolution": nanobanana_resolution
        }

        return {
            "model": model_type,
            "data": {
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": generation_config
            }
        }

    def send_request(self, api_key: str, request_data: Dict) -> Dict:
        """发送API请求 - 仅非流式模式"""

        url = "https://tech.seasungame.com/ai_in_one/v2/createImage"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
            'User-Agent': 'ComfyUI-Gemini-Node/1.0'
        }
        
        print(f"发起非流式请求到")

        response = requests.post(url, headers=headers, json=request_data, timeout=600)
        
        if response.status_code != 200:
            raise Exception(f"API请求失败 {response.status_code}: {response.text}")
            
        return response.json()

    def extract_content(self, response_data: Dict) -> tuple[List[str], str]:
        """提取响应中的图像和文本"""
        base64_images = []
        text_content = ""
        
        candidates = response_data.get('candidates', [])
        if not candidates:
            raise ValueError("响应中没有candidates")
        
        content = candidates[0].get('content', {})
        parts = content.get('parts', [])
        
        for part in parts:
            if 'text' in part:
                text_content += part['text']
            elif 'inlineData' in part and 'data' in part['inlineData']:
                base64_images.append(part['inlineData']['data'])
        
        # 备用方案：从文本中提取base64图像
        if not base64_images and text_content:
            patterns = [
                r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)',
                r'!\[.*?\]\(data:image/[^;]+;base64,([A-Za-z0-9+/=]+)\)',
            ]
            for pattern in patterns:
                base64_images.extend(re.findall(pattern, text_content))

        return base64_images, text_content.strip()

    def generate_images(self, api_key, prompt, model_type, nanobanana_aspect_ratio, nanobanana_resolution,
                       input_image_1=None, input_image_2=None, input_image_3=None,
                       input_image_4=None, input_image_5=None, chatgpt_size=None,
                       chatgpt_quality=None, seed=0):
        batch_size = 1
        start_time = time.time()
        input_images = [img for img in [input_image_1, input_image_2, input_image_3, input_image_4, input_image_5] if img is not None]

        np.random.seed(seed)
        random.seed(seed)    

        all_b64_images = []
        all_texts = []

        for i in range(batch_size):
            print(f"\n生成第 {i+1}/{batch_size} 张图片")

            try:
                # 构建请求数据
                request_data = self.create_request_data(prompt, model_type, nanobanana_aspect_ratio,
                                                        nanobanana_resolution, input_images, chatgpt_size,
                                                        chatgpt_quality)

                # 发送请求（仅非流式）
                response_data = self.send_request(api_key, request_data)

                # 提取内容
                base64_images, text_content = self.extract_content(response_data)

                all_b64_images.extend(base64_images)
                if text_content:
                    all_texts.append(text_content)

            except Exception as e:
                error_msg = f"生成第 {i+1} 张图片失败: {str(e)}"
                print(error_msg)
                all_texts.append(error_msg)

        # 计算耗时
        total_time = time.time() - start_time
        time_info = f"生图完成，耗时: {total_time:.2f}秒"
        print(f"\n{time_info}")

        if not all_b64_images:
            if all_texts:
                return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), f"{time_info}\n" + "\n".join(all_texts))
            else:
                raise Exception("未生成任何图像或文本内容")

        # 转换图像
        image_tensor = self.base64_to_tensor(all_b64_images)
        combined_text = f"{time_info}\n" + ("\n".join(all_texts) if all_texts else f"成功生成 {len(all_b64_images)} 张图像")

        print(f"最终完成！共生成 {len(all_b64_images)} 张图片")
        return (image_tensor, combined_text)

# 注册节点
NODE_CLASS_MAPPINGS = {"GeminiOpenAIProxyNode": GeminiOpenAIProxyNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiOpenAIProxyNode": "TC nano banana! 🍌"}