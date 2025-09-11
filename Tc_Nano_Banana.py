import json
import requests
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from typing import List, Dict, Optional
import re
import random
import time

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
                "api_base_url": ("STRING", {
                    "default": "https://example.com"
                }),
                "api_key": ("STRING", {
                    "default": "place-your-key-here"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Hello nano banana!"
                }),
                "model_type": ("STRING", {
                    "default": "gemini-2.5-flash-image-preview"
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 4
                }),
                "seed": ("INT", {
                    "default": 1024, "min": -1, "max": 102400
                }),
            },
            "optional": {
                "input_image_1": ("IMAGE",),
                "input_image_2": ("IMAGE",),
                "input_image_3": ("IMAGE",),
                "input_image_4": ("IMAGE",),
                "input_image_5": ("IMAGE",),
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
        """将base64转换为tensor"""
        images = []
        for b64_str in base64_strings:
            img_data = base64.b64decode(b64_str)
            img = Image.open(BytesIO(img_data)).convert('RGB')
            img_array = np.array(img).astype(np.float32) / 255.0
            images.append(img_array)
        return torch.from_numpy(np.stack(images))
    
    def create_request_data(self, prompt: str, seed: int, input_images: List[torch.Tensor] = None) -> Dict:
        """构建请求数据"""
        # 基于种子添加风格变化
        if seed != -1:
            np.random.seed(seed)
            random.seed(seed)
            style_variations = [
                "detailed, high quality",
                "masterpiece, ultra detailed", 
                "photorealistic, stunning",
                "artistic, beautiful composition",
                "vibrant colors, sharp focus"
            ]
            style = style_variations[seed % len(style_variations)]
            final_prompt = f"{prompt}, {style}"
        else:
            final_prompt = prompt
            
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
        }
        
        if seed != -1:
            generation_config["seed"] = seed
        
        return {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config
        }

    def send_request(self, api_key: str, request_data: Dict, model_type: str, 
                    api_base_url: str) -> Dict:
        """发送API请求 - 仅非流式模式"""
        endpoint = "generateContent"
        url = f"{api_base_url.rstrip('/')}/v1beta/models/{model_type}:{endpoint}"
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
            'User-Agent': 'ComfyUI-Gemini-Node/1.0'
        }
        
        print(f"发起非流式请求到: {url}")
        
        response = requests.post(url, headers=headers, json=request_data, timeout=180)
        
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

    def generate_images(self, api_base_url, api_key, prompt, model_type, batch_size, seed,
                       input_image_1=None, input_image_2=None, input_image_3=None, 
                       input_image_4=None, input_image_5=None):
        
        start_time = time.time()
        input_images = [img for img in [input_image_1, input_image_2, input_image_3, input_image_4, input_image_5] if img is not None]
        
        # 处理种子
        if seed == -1:
            base_seed = random.randint(0, 102400)
        else:
            base_seed = seed
        
        all_b64_images = []
        all_texts = []
        
        for i in range(batch_size):
            current_seed = base_seed + i if seed != -1 else -1
            print(f"\n生成第 {i+1}/{batch_size} 张图片 (种子: {current_seed if current_seed != -1 else '随机'})")
            
            try:
                # 构建请求数据
                request_data = self.create_request_data(prompt, current_seed, input_images)
                
                # 发送请求（仅非流式）
                response_data = self.send_request(api_key, request_data, model_type, api_base_url)
                
                # 提取内容
                base64_images, text_content = self.extract_content(response_data)
                
                all_b64_images.extend(base64_images)
                if text_content:
                    all_texts.append(f"[种子: {current_seed}] {text_content}")
                    
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