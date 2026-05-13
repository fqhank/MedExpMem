# import transformers.utils.import_utils
# if not hasattr(transformers.utils.import_utils, 'is_torch_fx_available'):
#     transformers.utils.import_utils.is_torch_fx_available = lambda: False
import json
import os
from datetime import datetime
from agents import Agent
import re
from sentence_transformers import SentenceTransformer
# from FlagEmbedding import FlagReranker
import torch

def json_cleaner(text):
    """鲁棒的 JSON 提取器，防止模型输出包含 Markdown 代码块"""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else text

STANDARD_KEYWORDS = [
    # --- 成分 ---
    "solid", "cystic", "mixed", "fatty",
    
    # --- 密度/信号 ---
    "hyperintense", "hypointense", "isointense",  # 不区分T1/T2，简化
    
    # --- 强化 ---
    "non_enhancing", "homogeneous_enh", "heterogeneous_enh", "rim_enh",
    
    # --- 边界 ---
    "well_defined", "ill_defined",
    
    # --- 特殊成分 ---
    "calcification", "hemorrhage", "fat_containing", "gas",
    
    # --- 功能 ---
    "diffusion_restriction",
    
    # --- 形态 ---
    "size_abnormal", "position_abnormal",
    
    # --- 临床特征（新增，重要！）---
    "elderly", "pediatric",           # 年龄对诊断影响大
    "painless", "rapid_growth",       # 临床表现
    "markers_negative", "markers_elevated",  # 肿瘤标志物
    
    # --- 兜底 ---
    "others"
]

DEPARTMENTS = [
    "neuroradiology",           # 神经：脑、脊髓、脊柱
    "head_and_neck",            # 头颈：眼眶、鼻窦、颞骨、咽喉、甲状腺
    "thoracic",                 # 胸部：肺、纵隔、胸膜
    "cardiac",                  # 心脏：心脏、大血管
    "breast",                   # 乳腺
    "abdominal",                # 腹部：肝胆胰脾、胃肠道
    "urogenital",               # 泌尿生殖：肾、膀胱、前列腺、子宫、卵巢、睾丸
    "musculoskeletal",          # 骨肌：骨、关节、软组织
    "vascular",                 # 血管：外周血管、介入
    "pediatric",                # 儿科（跨系统，但有特殊性）
    "others",                   # 兜底
]

DEPARTMENT_ORGANS = {
    # =========================================================================
    # 神经放射：脑、脊髓、脊柱
    # =========================================================================
    "neuroradiology": [
        "brain",                    # 大脑 (含大脑半球、白质、灰质、基底节、丘脑等)
        "cerebellum",               # 小脑 (含后颅窝)
        "brainstem",                # 脑干
        "pituitary",                # 垂体 (含鞍区、鞍上)
        "meninges",                 # 脑膜 (含硬脑膜、软脑膜、蛛网膜)
        "skull_base",               # 颅底 (含岩尖、颈静脉孔、斜坡等)
        "skull",                    # 颅骨
        "cistern",                  # 脑池 (含CPA、桥前池等)
        "cerebral_vessels",         # 脑血管 (含Willis环、静脉窦等)
        "cavernous_sinus",          # 海绵窦
        "optic_nerve",              # 视神经/视路
        "pineal",                   # 松果体区
        "spine",                    # 脊柱/脊髓 (神经放射角度，含脊髓病变)
        "others",                   # 其他
    ],
    
    # =========================================================================
    # 头颈部：眼眶、鼻窦、颞骨、咽喉、甲状腺
    # =========================================================================
    "head_and_neck": [
        "orbit",                    # 眼眶 (含眼球、眼肌、泪腺)
        "sinonasal",                # 鼻窦鼻腔 (含上颌窦、筛窦、额窦、蝶窦)
        "pharynx",                  # 咽部 (含鼻咽、口咽、扁桃体)
        "oral_cavity",              # 口腔 (含舌、口底)
        "larynx",                   # 喉
        "thyroid",                  # 甲状腺
        "salivary_gland",           # 唾液腺 (含腮腺、颌下腺)
        "temporal_bone",            # 颞骨 (含中耳、内耳)
        "mandible",                 # 下颌骨 (含牙齿)
        "maxilla",                  # 上颌骨
        "neck",                     # 颈部软组织 (含颈部淋巴结、颈动脉间隙等)
        "face",                     # 面部/头皮
        "others",                   # 其他
    ],
    
    # =========================================================================
    # 胸部：肺、纵隔、胸膜
    # =========================================================================
    "thoracic": [
        "lung",                     # 肺
        "airways",                  # 气道 (含气管、支气管)
        "pleura",                   # 胸膜
        "mediastinum",              # 纵隔 (含淋巴结)
        "esophagus",                # 食管
        "diaphragm",                # 膈肌
        "chest_wall",               # 胸壁 (含肋骨)
        "pericardium",              # 心包 (胸部角度)
        "others",                   # 其他
    ],
    
    # =========================================================================
    # 心脏：心脏、大血管
    # =========================================================================
    "cardiac": [
        "heart",                    # 心脏 (含心腔、心肌、瓣膜)
        "pericardium",              # 心包
        "coronary",                 # 冠状动脉/冠状静脉
        "pulmonary_vessels",        # 肺动脉/肺静脉
        "aortic_root",              # 主动脉根部
        "others",                   # 其他
    ],
    
    # =========================================================================
    # 乳腺
    # =========================================================================
    "breast": [
        "breast",                   # 乳腺
        "axilla",                   # 腋窝 (含淋巴结)
        "others",                   # 其他
    ],
    
    # =========================================================================
    # 腹部：肝胆胰脾、胃肠道
    # =========================================================================
    "abdominal": [
        "liver",                    # 肝脏
        "biliary",                  # 胆道系统 (含胆囊、胆管)
        "pancreas",                 # 胰腺
        "spleen",                   # 脾脏
        "stomach",                  # 胃
        "small_bowel",              # 小肠 (含十二指肠、空肠、回肠)
        "colon",                    # 结直肠 (含结肠、直肠)
        "appendix",                 # 阑尾
        "peritoneum",               # 腹膜腔 (含系膜、网膜)
        "retroperitoneum",          # 腹膜后
        "adrenal",                  # 肾上腺
        "abdominal_wall",           # 腹壁
        "groin",                    # 腹股沟
        "portal_vein",              # 门静脉系统
        "others",                   # 其他
    ],
    
    # =========================================================================
    # 泌尿生殖：肾、膀胱、前列腺、子宫、卵巢、睾丸
    # =========================================================================
    "urogenital": [
        "kidney",                   # 肾脏
        "ureter",                   # 输尿管
        "bladder",                  # 膀胱
        "urethra",                  # 尿道
        "prostate",                 # 前列腺 (含精囊)
        "testis",                   # 睾丸/阴囊 (含附睾、精索)
        "penis",                    # 阴茎
        "uterus",                   # 子宫 (含宫颈)
        "ovary",                    # 卵巢/附件
        "vagina",                   # 阴道
        "pelvis",                   # 盆腔 (非特定器官的盆腔病变)
        "perineum",                 # 会阴
        "others",                   # 其他
    ],
    
    # =========================================================================
    # 骨肌：骨、关节、软组织
    # =========================================================================
    "musculoskeletal": [
        "shoulder",                 # 肩部 (含肩关节、肩胛骨、锁骨)
        "upper_arm",                # 上臂 (含肱骨)
        "elbow",                    # 肘关节
        "forearm",                  # 前臂 (含尺桡骨)
        "hand_wrist",               # 手/腕
        "hip",                      # 髋关节
        "thigh",                    # 大腿 (含股骨)
        "knee",                     # 膝关节
        "lower_leg",                # 小腿 (含胫腓骨)
        "ankle_foot",               # 踝/足
        "spine",                    # 脊柱 (骨科角度，含椎骨、椎间盘)
        "pelvis",                   # 骨盆
        "sacroiliac",               # 骶髂关节/骶骨
        "skeleton",                 # 全身骨骼 (多发骨病变)
        "soft_tissue",              # 软组织
        "others",                   # 其他
    ],
    
    # =========================================================================
    # 血管：外周血管、介入
    # =========================================================================
    "vascular": [
        "aorta",                    # 主动脉 (含胸主、腹主动脉)
        "carotid_artery",           # 颈动脉
        "peripheral_artery",        # 外周动脉 (含髂动脉、股动脉等)
        "celiac_trunk",             # 腹腔干/内脏动脉
        "inferior_vena_cava",       # 下腔静脉
        "peripheral_vein",          # 外周静脉 (含下肢深静脉等)
        "portal_vein",              # 门静脉 (介入角度，如TIPS)
        "others",                   # 其他
    ],
    
    # =========================================================================
    # 儿科（跨系统，但有特殊性）
    # 说明：儿科病例organ沿用其他department的标准，但疾病谱有特殊性
    # =========================================================================
    "pediatric": [
        # 神经系统
        "brain",                    # 大脑 (含胎儿/新生儿脑)
        "spine",                    # 脊柱/脊髓
        # 头颈部
        "orbit",                    # 眼眶
        "neck",                     # 颈部
        # 胸部
        "lung",                     # 肺
        "mediastinum",              # 纵隔
        # 腹部
        "liver",                    # 肝脏
        "biliary",                  # 胆道
        "small_bowel",              # 小肠
        "retroperitoneum",          # 腹膜后 (含神经母细胞瘤等)
        # 泌尿生殖
        "kidney",                   # 肾脏 (含Wilms瘤等)
        "bladder",                  # 膀胱
        "pelvis",                   # 盆腔
        "ovary",                    # 卵巢
        # 骨肌
        "skeleton",                 # 骨骼 (含骨发育异常)
        "hip",                      # 髋关节 (含DDH等)
        "others",                   # 其他
    ],
    
    # =========================================================================
    # 兜底
    # =========================================================================
    "others": [
        "lymph_node",               # 淋巴结 (全身性)
        "skin",                     # 皮肤/皮下
        "multisystem",              # 多系统病变
        "others",                   # 其他
    ],
}


class ExperienceBank:

    def __init__(self, storage_path="medical_exp_bank.json"):
        self.storage_path = storage_path
        # 1. Note_Storage: 存放具体笔记实体 {note_id: {data}}
        self.notes = {}
        # 2. Index_Tree: 动态层级索引 {科室: {器官: {关键词: [note_ids]}}}
        self.index_tree = {}
        self.emb_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO', trust_remote_code=True)
        # self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        # self.emb_model = SentenceTransformer("NeuML/pubmedbert-base-embeddings")
        
        self.load()

    def update_note(self, note_id, note_content):
        self.notes[note_id]["note"] = note_content

    def add_note(self, department, organ, diag, note_content):
        """
        根据 Agent 的输入动态构建目录并存储笔记
        note_id 采用递增数字：note_1, note_2 ...
        """

#         if department in self.index_tree.keys() and organ in self.index_tree[department].keys() and diag in self.index_tree[department][organ].keys():
#             print('Note merging')
#             agent = Agent()
#             diag_a = note_content['differential diagnosis pair'].split(' v.s. ')[0]
#             diag_b = note_content['differential diagnosis pair'].split(' v.s. ')[1]
#             prompt_librarian = f"""
# ### ROLE
# You are a medical librarian merging two experience notes into a more sound one.

# ### INPUT
# Existed Note: {self.index_tree[department][organ][diag]}
# New Note: {note_content}

# ### TASK
# 1. Merge the new note with the existing one, keep each keys unchanged. keeping the most important information from both.

# ### OUTPUT FORMAT (JSON)
# {{  "department": "the most relevant department that the note belong to. MUST choose one from {DEPARTMENTS}.",
#     "organ/region": "the most relvant organ/region that the note belong to. Must be standarized and choose from pre-defined organs under the department from {DEPARTMENT_ORGANS}.",
#     "differential diagnosis pair":{note_content['differential diagnosis pair']},
#     "explanation of {diag_a}": "1 sentence: extract the medical description and definition (what it is) of the diagnosis if ground-truth provided (if applicable)",
#     "confusion points": [
#         "1. e.g. Both diagnosis can present with...",
#         "2. e.g. Imaging overlap includes...",
#         "3. e.g. Clinical presentations may be similar when..."
#     ],
#     "key discriminators": "What discriminates one diagnosis from another, and other differentials if applicable.",
#     "decision rule": "If [X1] → favor {diag_a}, if [X2] → exclude/unlikey .. {diag_a}; If [Y1] → favor {diag_b}, if [Y2] → exclude/unlikey ... {diag_b}; If [Z] → consider alternatives [other diagnosis] (if applicable)",
#     "error analysis": "decompose junior's chain of thought and point out how it made mistakes: e.g. scenario → junior thinking node 1 → ... → thinking node n [mistake: incorrectly interpretate ..., miss ...] → ..."
# }}
# """
#             response = agent.chat(prompt_librarian)
#             note_content = json.loads(json_cleaner(response))
#             print('merged')

        # --- 自动生成递增 ID ---
        next_index = len(self.notes) + 1
        note_id = f"note_{next_index}"

        # --- 1. 更新索引树 (Index Tree) ---
        if department not in self.index_tree:
            department = department if department in DEPARTMENTS else "others"
            self.index_tree[department] = {}
        
        if organ not in self.index_tree[department]:
            organ = organ if organ in DEPARTMENT_ORGANS[department] else "others"
            self.index_tree[department][organ] = {}
            
        if 'differential diagnosis pair' in note_content.keys():
            self.index_tree[department][organ][note_content['differential diagnosis pair']] = note_content
        else:
            self.index_tree[department][organ][note_content['differential_diagnosis_pair']] = note_content


    def _get_existing_organs(self, department):
        """获取某科室下已有的所有organ名称"""
        if department in self.index_tree:
            return list(self.index_tree[department].keys())
        return []

    def get_skeleton(self):
        """
        返回经验库的骨架结构（去掉末梢的 note_id 列表）
        输出格式：{ "Department": { "Organ": ["keyword1", "keyword2"] } }
        """
        skeleton = {}
        for dept, organs in self.index_tree.items():
            skeleton[dept] = {}
            for organ, kws_dict in organs.items():
                # kws_dict.keys() 包含了所有的关键词，我们将其转化为列表
                # 这样 Agent 就能看到哪些关键词已经有笔记了
                skeleton[dept][organ] = list(kws_dict.keys())
        return skeleton

    def get_skeleton_2layers(self):
        skeleton = {}
        for dept, organs in self.index_tree.items():
            skeleton[dept] = [k for k in organs.keys()]
        return skeleton

    def summary(self):
        """
        以树形大纲格式输出。
        基于“所有分支皆非空”的原则，直接进行三层嵌套遍历。
        """
        if not self.index_tree:
            return "Experience Bank is currently empty."
            
        output = ["=== Medical Experience Bank Global Index ==="]
        
        for cat, organs in self.index_tree.items():
            output.append(f"\n{cat}") # 第一层：科室
            
            for organ, kws in organs.items():
                output.append(f"  └── {organ}") # 第二层：器官
                
                for kw, note_ids in kws.items(): # 第三层：关键词
                    # 直接获取诊断预览
                    output.append(f"      ├── [# {kw}] {note_ids}")
                    
        return "\n".join(output)

    def query_intersection(self, department, organ, keywords):
        """
        交叉检索：寻找同时具备多个标签的笔记
        """
        skeleton = self.get_skeleton()
    
        if skeleton=={}:
            return ['No experience note can be found.'], []

        agent = Agent(model_info='Qwen/Qwen3-VL-8B-Instruct')
        prompt_librarian = f"""
### ROLE
You are a medical librarian helping to retrieve relevant experience notes.

### USER QUERY
Department: {department}
Organ/Region: {organ}
Keywords: {keywords}

### CURRENT EXPERIENCE BANK INDEX
{json.dumps(skeleton, indent=2)}

### TASK
Find the most relevant EXISTING paths in the index that match the user's query.

**Rules:**
1. You can ONLY select from paths that EXIST in the index above
2. Each path must specify: department, organ, and 1-3 keywords
3. All selected departments, organs, and keywords MUST exist in the index
4. If nothing relevant exists, return empty list

**Matching Strategy:**
- Match department first (exact match required)
- Match organ (consider synonyms: lung/lungs, kidney/renal, liver/hepatic)
- Match keywords by relevance to the query

### OUTPUT FORMAT (JSON)
{{
    "selected_paths": 
        {{
            "department": "existing department or empty",
            "organ": "existing organ (exact spelling from index) or empty",
            "keywords": ["existing_kw1", "existing_kw2"... or empty]
        }}
    ,
    "reasoning": "brief explanation of selection"
}}
"""
        response = agent.chat(prompt_librarian)
        result = json.loads(json_cleaner(response))

        department = result["selected_paths"]['department']
        organ = result["selected_paths"]['organ'].lower().strip()
        keywords = result["selected_paths"]['keywords']

        if department not in self.index_tree or organ not in self.index_tree[department]:
            return [], []

        query_keywords = [kw.lower().strip() for kw in keywords]
                         
        id_sets = []
        for kw in query_keywords:
            kw = kw.lower().strip()
            ids = self.index_tree[department][organ].get(kw, [])
            if ids:
                id_sets.append(set(ids))

        if not id_sets:
            return [], []

        # 取交集
        intersect_ids = set.intersection(*id_sets)
        return [self.notes[nid] for nid in intersect_ids], [intersect_ids]

    def query_flexible(self, query, scenario, diffs):
        """
        柔性检索逻辑：
        1. 尝试所有 query_keywords 的严格交集。
        2. 若交集为空，则按顺序移除最后一个（最不重要的）关键词，重新尝试。
        3. 直到找到匹配结果，或关键词列表被扣减至 0。
        """

        skeleton = self.get_skeleton_2layers()
    
        if skeleton=={}:
            return ['No experience note can be found.'], []

#         agent = Agent(model_info='Qwen/Qwen3-VL-8B-Instruct')
#         prompt_librarian = f"""
# ### ROLE
# You are a medical librarian helping to retrieve relevant experience notes.

# ### USER QUERY
# Department: {department}
# Organ/Region: {organ}

# ### CURRENT EXPERIENCE BANK INDEX
# {skeleton}

# ### TASK
# Find the most relevant EXISTING paths in the index that match the user's query.

# **Rules:**
# 1. You can ONLY select from paths that EXIST in the index above
# 2. All selected departments, organs, and keywords MUST exist in the index
# 3. Never return the empty path, determine the most related one, even if abandoning some query keywrods

# **Matching Strategy:**
# - Match department first (exact match required)
# - Match organ (consider synonyms: lung/lungs, kidney/renal, liver/hepatic)

# ### OUTPUT FORMAT (JSON)
# {{
#     "selected_paths": 
#         {{
#             "department": "existing department or empty",
#             "organ": "existing organ (exact spelling from index) or empty",
#         }},
#     "reasoning": "brief explanation of selection",
# }}
# """
#         response = agent.chat(prompt_librarian)
#         result = json.loads(json_cleaner(response))
#         del agent

#         department = result["selected_paths"]['department'].lower().strip()
#         organ = result["selected_paths"]['organ'].lower().strip()

        search_pair = []
        for pair in query:
            department = pair[0].lower().strip()
            organ = pair[1].lower().strip()
            if department in self.index_tree and organ in self.index_tree[department]:
                search_pair.append([department, organ])
                print(department, organ)

        if len(search_pair)==0:
            print('not match')
            return ['No experience note can be found.'], [] # 返回笔记列表和最终匹配使用的关键词

        notes = []
        # stage 1: embedding sort
        embeddings = self.emb_model.encode(diffs, task="retrieval")
        keys = []
        dep_org = []
        for pair in search_pair:
            department = pair[0].lower().strip()
            organ = pair[1].lower().strip()
            keys_temp = [k for k in self.index_tree[department][organ].keys()]
            dep_org_temp = [[department, organ] for i in range(len(keys_temp))]
            keys = keys + keys_temp
            dep_org = dep_org + dep_org_temp
        embeddings_keys = self.emb_model.encode(keys, task="retrieval")
        similarities = self.emb_model.similarity(embeddings, embeddings_keys)
        idx_x, idx = torch.where(similarities >= 0.92)
        idx = idx.unique().tolist()

        high_scores = []
        if len(idx)>10:
            for r in range(len(idx)):
                high_scores.append(similarities[idx_x[r]][idx[r]])
            v, top5_indices = torch.tensor(high_scores).topk(10)
            idx = [idx[x] for x in top5_indices]

        print(idx)
        
        if len(idx)>0:
            notes = []
            for i in idx:
                note = self.index_tree[dep_org[i][0]][dep_org[i][1]][keys[i]]
                notes.append(self.index_tree[dep_org[i][0]][dep_org[i][1]][keys[i]])
        
        if len(idx) == 0:
            return [], []
#         else:
#             ratings = []
#             for i in range(len(idx)):
#                 agent = Agent()
#                 prompt_librarian = f"""
# ### ROLE
# You are filtering experience notes to find those useful for the current differential diagnosis task.

# ### CURRENT CASE
# **Clinical Scenario:**
# {scenario}

# **Options to Differentiate:**
# {diffs}

# ### CANDIDATE NOTES
# {notes[i]}

# ### TASK
# For each note, determine: **Can this note help distinguish between ANY of the current options?**

# ### SCORING GUIDE (0-10)

# **Score 8-10: KEEP - Directly Useful**
# The note can help differentiate at least one option from others:
# - Note's diagnosis matches or is very similar to one option
# - Note contains imaging features that can rule in/out an option
# - Note discusses pitfalls relevant to distinguishing these options

# **Score 5-7: KEEP - Potentially Useful**
# The note provides relevant background:
# - Same disease category as one option (e.g., both are liver cysts)
# - Discusses a closely related condition that shares features with an option
# - May help understand why one option is more/less likely

# **Score 0-4: EXCLUDE - Not Useful**
# The note is unlikely to help:
# - Different organ system
# - Unrelated disease category
# - No overlap with any option

# ### KEY QUESTION FOR EACH NOTE
# Ask yourself: "If I'm trying to decide between {diffs}, would reading this note help me?"
# - YES, directly → 8-10
# - YES, somewhat → 5-7
# - NO → 0-4

# ### OUTPUT JSON
# {{
#     "rating": score,
#     "rational": "briefly reasoning why you give such rates",
# }}
# """
#                 response = agent.chat(prompt_librarian)
#                 # print(response)
#                 rating = json.loads(json_cleaner(response))['rating']
#                 ratings.append(float(rating))

#                 del agent

#         temp_notes = []
#         for i in range(len(ratings)):
#             if float(ratings[i])>4.0:
#                 temp_notes.append(self.index_tree[dep_org[i][0]][dep_org[i][1]][keys[i]])
#         if len(temp_notes)>0:
#             print(len(temp_notes),'filtered notes')
#             return temp_notes, []
        
        if len(notes)>0:
            print(len(notes),'filtered notes')
            return notes, []
        
        print('found 0 related')
        return ['No experience note can be found.'], [] # 彻底搜不到
        
    def save(self):
        data = {
            "index_tree": self.index_tree,
        }
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.index_tree = data.get("index_tree", {})
                    self.notes = data.get("notes", {})
            except:
                print("加载失败，初始化新库")

    def self_check(self):
        count = 0
        for dep in self.index_tree.keys():
            for org in self.index_tree[dep].keys():
                for diag in self.index_tree[dep][org].keys():
                    diags = diag.split(' v.s. ')
                    print(diags)
                    if diags[0].lower().strip() == diags[1].lower().strip():
                        print(diags)
                        count+=1
        print(count)