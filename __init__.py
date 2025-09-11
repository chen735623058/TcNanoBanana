"""
通用ComfyUI自定义节点加载器
支持任何文件夹名称，自动检测并加载节点
"""

import os
import sys
import importlib.util
from pathlib import Path

# 获取当前文件夹路径
current_dir = Path(__file__).parent

# 初始化节点映射字典
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 自动查找并加载所有Python文件中的节点
for py_file in current_dir.glob("*.py"):
    if py_file.name == "__init__.py":
        continue
    
    try:
        # 动态导入模块
        module_name = py_file.stem
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 合并节点映射
        if hasattr(module, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
        
        if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            
        print(f"✅ 成功加载节点文件: {py_file.name}")
        
    except Exception as e:
        print(f"❌ 加载节点文件失败 {py_file.name}: {str(e)}")

# 打印加载的节点信息
if NODE_CLASS_MAPPINGS:
    print(f"🎉 总共加载了 {len(NODE_CLASS_MAPPINGS)} 个自定义节点:")
    for node_name in NODE_CLASS_MAPPINGS.keys():
        display_name = NODE_DISPLAY_NAME_MAPPINGS.get(node_name, node_name)
        print(f"   - {display_name} ({node_name})")
else:
    print("⚠️  未找到任何有效的节点")

# ComfyUI需要的变量
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
WEB_DIRECTORY = "./web"