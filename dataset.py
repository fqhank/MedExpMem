import json
import os
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional

class EuroradDataset:
    """
    Eurorad 医学影像数据集
    
    数据划分:
      - train: 前5000个样本（仅文本，不加载图像）
      - test:  5000之后的样本（文本 + 从figures/加载图像）
    
    标准 PyTorch Dataset API:
      - __len__() → 当前 split 的样本数
      - __getitem__(idx) → 返回第 idx 个样本（0-based，相对于当前 split）
    """
    
    def __init__(
        self,
        jsonl_path: str = "data/all_parsed_cases_with_meta.jsonl",
        figures_root: str = "figures",
        split: str = "train",      # 'train' 或 'test'
        split_index: int = 7000    # 分割点：前N个为训练集
    ):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        
        self.jsonl_path = Path(jsonl_path)
        self.figures_root = Path(figures_root)
        self.split = split
        self.split_index = split_index
        
        # 验证文件存在性
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
        
        # 一次性加载全部文本数据（JSONL 通常 < 100MB，内存友好）
        self._all_data = self._load_all_cases()
        
        # 根据 split 截取数据视图（不复制，仅保存索引范围）
        if split == "train":
            self._start_idx = 0
            self._end_idx = min(split_index, len(self._all_data))
        else:  # test
            self._start_idx = split_index
            self._end_idx = len(self._all_data)
        
        print(f"✅ {split.capitalize()} set: {len(self)} samples "
              f"(indices {self._start_idx}-{self._end_idx-1})")

    def _load_all_cases(self) -> List[Dict]:
        """加载全部JSONL数据到内存"""
        cases = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        cases.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"[Warning] Skip invalid JSON at line {line_num}: {e}")
        return cases

    def _match_image_path(self, case_id: str, jsonl_filename: str) -> Optional[Path]:
        """
        智能匹配图像路径（解决文件名前缀差异）
        
        JSONL记录: "10011/01_000001.jpg"
        实际下载:  "figures/10011/000001.jpg"  (无"01_"前缀)
        
        匹配优先级:
          1. 原始basename (01_000001.jpg)
          2. 去序号前缀 (000001.jpg) ← 通常匹配成功
          3. 扩展名变体 (.jpg ↔ .png)
        """
        case_dir = self.figures_root / str(case_id)
        if not case_dir.exists():
            return None
        
        basename = Path(jsonl_filename).name  # "01_000001.jpg"
        candidates = [basename]
        
        # 策略1: 去掉序号前缀 "XX_"
        if '_' in basename:
            no_prefix = basename.split('_', 1)[-1]  # "000001.jpg"
            candidates.append(no_prefix)
        
        # 策略2: 扩展名变体
        for cand in candidates[:]:
            if cand.endswith('.jpg'):
                candidates.append(cand[:-4] + '.png')
            elif cand.endswith('.png'):
                candidates.append(cand[:-4] + '.jpg')
        
        # 按优先级查找
        for name in candidates:
            img_path = case_dir / name
            if img_path.exists():
                return img_path
        return None

    def _load_images(self, case_id: str, figures_meta: List[Dict]) -> List[Dict]:
        """加载测试集图像（训练集不调用此方法）"""
        if not figures_meta:
            return []
        
        images = []
        for fig in figures_meta:
            jsonl_filename = fig.get("filename", "")
            if not jsonl_filename:
                continue
            
            img_path = self._match_image_path(case_id, jsonl_filename)
            if not img_path:
                continue  # 静默跳过缺失图像（医学数据常见）
            
            try:
                img = Image.open(img_path).convert("RGB")
                images.append({
                    "id": img_path.name,
                    "caption": fig.get("caption", ""),
                    "data": img  # PIL Image
                })
            except Exception as e:
                # 单图失败不影响整个case
                continue
        
        return images

    def __getitem__(self, idx: int) -> Dict:
        """
        获取样本（标准PyTorch API）
        
        Args:
            idx: 0-based 索引（相对于当前split）
        
        Returns:
            dict 包含:
              - case_id (str)
              - section (str)
              - clinical_history (str)
              - imaging_findings (str)
              - label (str): final_diagnosis
              - [test only] images (List[Dict]): 每个含 id/caption/data(PIL Image)
              - [test only] discussion (str)
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for {self.split} set (size={len(self)})"
            )
        
        # 映射到全局索引
        global_idx = self._start_idx + idx
        case = self._all_data[global_idx]
        case_id = str(case.get("case_id", ""))
        
        # 基础字段（训练/测试共用）
        sample = {
            "case_id": case_id,
            "section": case["meta"].get("section", "Unclassified"),
            "clinical_history": case.get("clinical_history", "").strip(),
            "imaging_findings": case.get("imaging_findings", "").strip(),
            "diagnosis": case.get("final_diagnosis", "").strip(),
            "discussion": case.get("discussion", "").strip(),
            "differential": case.get("differential_diagnosis", "").strip(),
            "url": case["meta"].get("url", " "),
        }
        
        # 测试集额外加载图像和讨论
        if self.split == "test" or self.split == "train":
            sample["images"] = case.get("figures", [])
        
        return sample

    def __len__(self) -> int:
        """返回当前split的样本数（标准PyTorch API）"""
        return self._end_idx - self._start_idx

    @staticmethod
    def get_split_indices(total_size: int, split_index: int = 5000):
        """静态工具：获取训练/测试索引范围（便于外部使用）"""
        train_indices = list(range(min(split_index, total_size)))
        test_indices = list(range(split_index, total_size))
        return train_indices, test_indices



# ==================== 快速验证 ====================
if __name__ == "__main__":
    print("="*70)
    print("🧪 EuroradDataset 验证")
    print("="*70)
    
    # 1. 训练集验证（前5000）
    print("\n[1] 训练集 (split='train')")
    try:
        train_ds = EuroradDataset(split="train", split_index=5000)
        for d in train_ds:
            print(d)
            exit()
        print(f"   ✓ 样本数: {len(train_ds)}")
        sample = train_ds[0]
        print(f"   ✓ 首样本 case_id: {sample['case_id']}")
        print(f"   ✓ 包含字段: {list(sample.keys())}")
        print(f"   ✓ 无图像字段: {'images' not in sample}")
    except Exception as e:
        print(f"   ✗ 失败: {e}")
    
    # 2. 测试集验证（5000之后）
    print("\n[2] 测试集 (split='test')")
    try:
        test_ds = EuroradDataset(split="test", split_index=5000)
        print(f"   ✓ 样本数: {len(test_ds)}")
        if len(test_ds) > 0:
            sample = test_ds[0]
            print(f"   ✓ 首样本 case_id: {sample['case_id']}")
            print(f"   ✓ 图像数量: {len(sample.get('images', []))}")
            if sample.get("images"):
                img = sample["images"][0]
                print(f"   ✓ 首图尺寸: {img['data'].size}, 标题: {img['caption'][:40]}...")
        else:
            print("   ⚠️  测试集为空（可能总样本数 ≤ 5000）")
    except Exception as e:
        print(f"   ✗ 失败: {e}")
    
    # 3. 索引范围验证
    print("\n[3] 索引范围验证")
    total = len(train_ds._all_data) if 'train_ds' in locals() else 0
    if total > 0:
        train_idx, test_idx = EuroradDataset.get_split_indices(total, 5000)
        print(f"   ✓ 总样本数: {total}")
        print(f"   ✓ 训练索引: [{train_idx[0]}, ..., {train_idx[-1]}] (共{len(train_idx)}个)")
        print(f"   ✓ 测试索引: [{test_idx[0] if test_idx else 'N/A'}, ..., "
              f"{test_idx[-1] if test_idx else 'N/A'}] (共{len(test_idx)}个)")
    
    print("\n" + "="*70)
    print("✅ 验证完成！可直接用于训练/推理流程")
    print("="*70)