import argparse
import json
import os
import re
from tqdm import tqdm

from dataset import EuroradDataset
from expbank import ExperienceBank, DEPARTMENT_ORGANS
from agents import Agent
import torch
import os
from search_tools import SearchTools

test_case_ids =  [18814, 18815, 18816, 18818, 18819, 18823, 18826, 18829, 18830, 18835, 18838, 18840, 18844, 18845, 18847, 18849, 18851, 18857, 18861, 18862, 18866, 18867, 18868, 18870, 18872, 18876, 18877, 18881, 18882, 18889, 18890, 18892, 18893, 18895, 18896, 18901, 18904, 18905, 18906, 18908, 18911, 18912, 18913, 18915, 18916, 18923, 18925, 18927, 18928, 18929, 18930, 18931, 18933, 18934, 18935, 18936, 18937, 18938, 18943, 18948, 18949, 18953, 18954, 18955, 18958, 18960, 18961, 18964, 18965, 18967, 18968, 18970, 18971, 18972, 18973, 18974, 18979, 18980, 18981, 18982, 18985, 18986, 18987, 18988, 18989, 18990, 18992, 18994, 18997, 18998, 19001, 19002, 19003, 19004, 19005, 19007, 19009, 19010, 19011, 19012, 19019, 19021, 19028, 19030, 19031, 19033, 19036, 19037, 19038, 19039, 19042, 19043, 19044, 19045, 19046, 19049, 19053, 19057, 19058, 19059, 19060, 19061, 19062, 19065, 19066, 19069, 19071, 19075, 19077, 19079, 19089, 19092, 19093, 19094, 19096, 19097, 19099, 19104, 19105, 19110, 19111, 19113, 19114, 19119, 19120, 19122, 19123, 19128, 19131, 19135, 19136, 19139, 19144, 19147, 19148, 19149, 19152, 19153, 19157, 19163, 19166, 19170, 19172, 19174, 19176, 19183, 19186, 19187, 19189, 19191, 19194, 19196, 19198, 19199, 19201, 19202, 19203, 19208, 19209, 19211, 19212, 19216, 19217, 19226, 19229, 19231, 19233, 19237, 19238, 19240, 19241, 19242, 19247, 19248, 19250, 19252, 19255, 19260, 19262, 19263, 19264, 19268, 19269, 19271, 19275, 19276, 19278, 19280, 19285, 19287, 19290, 19295, 19315, 19316, 19320, 19327, 19328, 19331, 19333, 19344, 19300, 19274, 19195, 19310, 19340, 19326, 19044, 19146]

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

def parse_args():
    parser = argparse.ArgumentParser(description="Gemini-Radiologist Main Entry")
    parser.add_argument("--mode", type=str, choices=['build', 'test'], required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    # parser.add_argument("--model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--exp_path", type=str, default="bank_qwen35vl_4b_1round.json")
    parser.add_argument("--start_idx", type=int, default=0, help="Resume index for build mode")
    return parser.parse_args()

# def json_cleaner(text):
#     """鲁棒的 JSON 提取器，防止模型输出包含 Markdown 代码块"""
#     match = re.search(r'\{.*\}', text, re.DOTALL)
#     return match.group(0) if match else text

def json_cleaner(text):
    """鲁棒的 JSON 提取器，处理 Markdown 代码块、多行文本和特殊字符"""
    # 步骤1：移除 Markdown 代码块的边界（```json/```）
    text = re.sub(r'```(?:json)?\s*', '', text)  # 移除开头的 ``` 或 ```json
    text = re.sub(r'\s*```', '', text)           # 移除结尾的 ```
    
    # 步骤2：使用非贪婪匹配提取最外层的 {} 内容（re.DOTALL 匹配换行符）
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return text  # 无匹配时返回原始清理后的文本
    
    # 步骤3：提取匹配的 JSON 字符串并清理首尾空白
    json_str = match.group(0).strip()
    
    # 步骤4：验证 JSON 格式（可选，增强鲁棒性）
    try:
        json.loads(json_str)  # 尝试解析，验证格式是否正确
        return json_str
    except json.JSONDecodeError:
        # 解析失败时返回原始匹配内容（或根据需求进一步处理）
        return json_strss

def main():
    args = parse_args()
    
    # 实例化库：构建模式下若 start_idx=0 则重置
    bank = ExperienceBank(storage_path=args.exp_path)
    # bank.load()
    # print(bank.get_skeleton_2layers())
    # exit()
    # ds = EuroradDataset("data/bench/2025_cases.jsonl",split='test',split_index=0)
    # ds = EuroradDataset("data/all_parsed_cases_with_meta.jsonl",split='train')
    # with open('medical_exp_bank_qwen8_gpt52_2rounds','r') as j:
    #     original = json.load(j)
    

    with open('test_options.json','r') as o:
        test_options = json.load(o)
    with open('train_options_check.json','r') as o1:
        train_options = json.load(o1)

    # if args.mode == 'build' and args.start_idx == 0:
    #     if os.path.exists(args.exp_path):
    #         os.rename(args.exp_path, f"{args.exp_path}.bak") # 备份旧库
    #         bank = ExperienceBank(storage_path=args.exp_path)
    
    # ==========================================
    # 模式一：构建模式 (Build Mode)
    # ==========================================
    if args.mode == 'build':
        ds = EuroradDataset("data/all_parsed_cases_with_meta.jsonl",split='train')

        print('STAGE 1')
        for t in tqdm(range(len(ds)), desc="[Building Bank]"):
            try:
                case = ds[t]
                if len(case['diagnosis'].strip())==0 or case['case_id'] not in train_options.keys():
                    continue
                print(f'=============================================\nCASE [{t}]')

                diffs = train_options[case['case_id']]
                options = {}
                for i in range(len(diffs)):
                    options[f'{chr(ord('A')+i)}'] = diffs[i]
                
                # Step 1: make diagnosis
                print('+++++ STAGE 1 STEP 1: DIAGNOSIS +++++')
                agent = Agent(model_info=args.model)
                prompt_step1 = f"""
### ROLE
You are a senior radiologist making a diagnostic decision.

### CLINICAL HISTORY
{case['clinical_history']}

### OPTIONS
{options}

### TASK
Analyze the images and clinical history systematically, then select the most likely diagnosis.

Analysis Thinking:
1. Identify: What is the primary abnormality? (location, size, composition)
2. Characterize: Key imaging features (borders, enhancement, signal/density)
3. Correlate: How do findings match the clinical history?
4. Differentiate: What features distinguish between the options?
5. Decide: What's the final diagnosis?

### OUTPUT (JSON)
{{
    "diagnosis": "Single letter of your final diagnosis (A/B/C/D/E)",
    "reasoning": "brief reasoning within 1 line 3 sentences",
}}
"""
                response_1 = agent.chat(prompt_step1, case['images'])
                response_1 = json.loads(json_cleaner(response_1))
                print(response_1)
                
                del agent
                torch.cuda.empty_cache()

                # 判断正确与否
                if case['diagnosis'].lower().strip(' .,!()[]') in options[response_1['diagnosis'].upper()].lower().strip(' .,!()[]'):
                    print('@ CORRECT! PASS')
                    if (t>0 and t % 10 == 0) or t==len(ds)-1:
                        bank.save()
                        print('saved')
                    continue

                print('@ FALSE! NOTE REQUIRED')
                
                # step 2
                print('+++++ STAGE 1 STEP 2: NOTE +++++')
                agent = Agent(model_info=args.model)
                prompt_2 = f"""
### ROLE
You are a senior radiologist and medical education expert. You need to review the diagnosis a junior made and help him extracting an experience note, thus to prevent future false fatal diagnosis.
This is a medical reflection task to extract generalizable differential diagnosis knowledge.

### CASE INFORMATION
**Clinical History:** {case['clinical_history']}

### OPTIONS
{options}

### DIAGNOSTIC
**Junior's Diagnosis (Note that the option align with OPTIONS):** {response_1}

### Ground-truth Reference
**Correct Diagnosis:** {case['diagnosis']}
**Imaging Findings (from expert):** {case['imaging_findings']}
**Expert Discussion:** {case['discussion']}

### YOUR TASK

**Step 1: Confusion Points (Generalizable)**
Summarize 2-4 key points explaining why these two diagnoses are commonly confused in clinical practice.
- These must be generalizable medical knowledge, NOT case-specific
- Focus on overlapping features that lead to diagnostic confusion
- Each point should be a complete, actionable statement

**Step 2: Key Discriminators**
For EACH diagnosis, list 2-3 features that help distinguish it from the other.
- Focus on imaging features, clinical context, and lab findings
- Be specific and actionable

**Step 3: Decision Rule**
Provide a concise rule for differentiating these two diagnoses.

**Step 4: Error Analysis (Case-specific)**
In 1-2 sentences, explain what specific finding or reasoning was missed in THIS case.

### OUTPUT FORMAT (JSON)
{{  "department": "the most relevant department that the note belong to. MUST choose one from {DEPARTMENTS}.",
    "organ/region": "the most relvant organ/region that the note belong to. Must be standarized and choose from pre-defined organs under the department from {DEPARTMENT_ORGANS}.",
    "differential diagnosis pair": "{case['diagnosis']} v.s. {options[response_1['diagnosis'].upper().strip()]}",
    "explanation of {case['diagnosis']}": "1 sentence: extract the medical description and definition (what it is) of the diagnosis if ground-truth provided (if applicable)",
    "confusion points": [
        "1. e.g. Both diagnosis can present with...",
        "2. e.g. Imaging overlap includes...",
        "3. e.g. Clinical presentations may be similar when..."
    ],
    "key discriminators": "What discriminates one diagnosis from another, and other differentials if applicable. If the discriminators of both diagnosis are provided by ground-truth information, extract both, else only extract the one with solid information ...",
    "decision rule": "If [X1] → favor {case['diagnosis']}, if [X2] → exclude/unlikey {case['diagnosis']}; If [Y1] → favor {options[response_1['diagnosis'].upper().strip()]}, if [Y2] → exclude/unlikey {options[response_1['diagnosis'].upper().strip()]}; If [Z] → consider alternatives [other diagnosis] (if applicable)",
    "error analysis": "decompose junior's chain of thought and point out how it made mistakes: e.g. [case scenario] → junior thinking node 1 → ... → thinking node n [mistake: incorrectly interpretate ..., miss ...] → ..."
}}

### GUIDELINES
- The department should from {DEPARTMENTS}
- Focus on IMAGING-based differentiation
- Confusion points should be transferable to future cases
- Key discriminators should be observable on imaging or known from clinical history
- Decision rule should be practical and memorable, use words that are not persuasive like 'favor', 'unlikely', 'consider'...
- Use specific diagnosis name instead of ambiguous 'Diagnosis A'
- Make use of Ground-truth Reference and prevent medical mistakes
"""
                final_response = agent.chat(prompt_2)
                final_response = json.loads(json_cleaner(final_response))
            
                print('note needed. Extracting note and indexing into bank...')
                try:
                    bank.add_note(final_response['department'], final_response['organ/region'], case['diagnosis'], final_response)
                    print(final_response)
                    print('[Note Added]')
                except Exception as e:
                    print(f"Error indexing case {i}: {e}")
            
                del agent
                torch.cuda.empty_cache()

                if (t>0 and t % 10 == 0) or t==len(ds)-1:
                    bank.save()
                    print('saved')
            except:
                continue

        os.system(f'cp {args.exp_path} ./copy_{args.exp_path}')

        print('Stage 2 Start')
        bank = ExperienceBank(storage_path=args.exp_path)
        for t in tqdm(range(len(ds)), desc="[Building Bank]"):
            try:
                case = ds[t]
                if len(case['diagnosis'].strip())==0 or case['case_id'] not in train_options.keys():
                    continue
                print(f'=============================================\nCASE [{t}]')

                diffs = train_options[case['case_id']]
                options = {}
                for i in range(len(diffs)):
                    options[f'{chr(ord('A')+i)}'] = diffs[i]

                # Step 1：提供病例 + 目录 -> 获取检索参数
                print('+++++ STAGE 2 STEP 1: SEARCH +++++')
                agent = Agent(model_info=args.model)
                skeleton_summary = bank.get_skeleton_2layers()
                prompt_step1 = f"""
### ROLE
You are a radiologist analyzing a new case and giving the diagnosis. You have access to an Experience Bank containing diagnostic notes from previous cases to help you.

### CLINICAL HISTORY
{case['clinical_history']}

### OPTIONS
{options}

### EXPERIENCE BANK DIRECTORY
The Experience Bank is organized as: Department → Organ/Region → Notes
Select the most related department and organ/region for this case and the related notes will be retrieved automatically.
[Current coverage]:
{skeleton_summary}

### TASK
1. Form a preliminary impression based on clinical history and images
2. Decide what deparments and organs in ### EXPERIENCE BANK DIRECTORY are most related to this case
3. Give 1 or 2 [DEPT, ORGAN] searching pair to retrieve notes

### STRICT OUTPUT FORMAT (JSON)
{{
    "query": [["DEPT1: one from current [Current coverage] department list","ORGAN1:specific organ or region from [Current coverage] under department you choose"],["DEPT2","ORGAN2"]],  // You can search 1 or 2 [DEPT, ORGAN] pair to enable good notes retrieval
}}

### GUIDELINES
- Request retrieval when you're uncertain and need help in a case for better accuracy
- Must contain all predefined keys 
- DEPT2 can be the same as DEPT1, but [DEPT1, ORGAN1] should never be as same as [DEPT2, ORGAN2]
- If the case are closely related to different departments/organs, search 2 [DEPT, ORGAN] pairs. Else search 1 is enough to cover the case
- DEPT and ORGAN you search MUST exist in [Current coverage] and ORGAN MUST under DEPT
"""
                response_1 = agent.chat(prompt_step1, case['images'])
                response_1 = json.loads(json_cleaner(response_1))

                use_notes = 0
                if 0:
                    retrieved_notes = ['No relevant experience note can be found. Please make diagnosis with your own experience.']
                    print('[NOTES] FALSE')
                # 执行检索
                else:
                    try:
                        query = response_1["query"]
                        print(query)
                        diffs = [options[k] for k in options.keys()]
                        retrieved_notes, matched_kws = bank.query_flexible(query, case['clinical_history'], diffs)
                        if len(retrieved_notes)>0 and retrieved_notes[0]!='No experience note can be found.':
                            use_notes = 1
                            retrieved_notes = [f'{retrieved_notes[i]}' for i in range(len(retrieved_notes))]
                            print('[NOTES]','\n'.join(retrieved_notes))
                        else:
                            use_notes = 0
                            print('[NOTES] EMPTY')
                    except:
                        retrieved_notes, _ = ['No relevant experience note can be found. Please make diagnosis with your own experience.'], []
                        print('[NOTES] FAIL')
                    
                del agent
                torch.cuda.empty_cache()
                
                # Step 2: make diagnosis
                print('+++++ STAGE 2 STEP 2: DIAGNOSIS +++++')
                agent = Agent(model_info=args.model)
                if use_notes == 0: 
                    prompt_step2 = f"""
### ROLE
You are a senior radiologist making a diagnostic decision.

### CLINICAL HISTORY
{case['clinical_history']}

### OPTIONS
{options}

### TASK
Analyze the images and clinical history systematically, then select the most likely diagnosis.

Analysis Thinking:
1. Identify: What is the primary abnormality? (location, size, composition)
2. Characterize: Key imaging features (borders, enhancement, signal/density)
3. Correlate: How do findings match the clinical history?
4. Differentiate: What features distinguish between the options?
5. Decide: What's the final diagnosis?

### OUTPUT (JSON)
{{
    "diagnosis": "Single letter of your final diagnosis (A/B/C/D/E)",
    "reasoning": "brief reasoning within 1 line 3 sentences",
}}
"""
                else:
                    prompt_step2 = f"""
### ROLE
You are a senior radiologist making a diagnostic decision.

### CLINICAL HISTORY
{case['clinical_history']}

### OPTIONS
{options}

### OUTER EXPERIENCE NOTES
The following notes describe commonly confused diagnosis pairs. Use them to guide your differential reasoning.
{retrieved_notes}

### TASK
Analyze systematically, leveraging the experience notes where relevant.

Analysis Thinking:
1. Identify: What is the primary abnormality? (location, size, composition)
2. Characterize: Key imaging features (borders, enhancement, signal/density)
3. Match Notes: Which experience notes cover diagnoses in the current options? 
4. Apply Discriminators: For matched notes, check if the case findings align with the `key_discriminators` for either diagnosis.
5. Apply Decision Rules: Follow the `decision_rule` — does the case satisfy conditions favoring one diagnosis over another?
6. Correlate: How do findings match the clinical history?
7. Decide: Based on imaging features, clinical context, and experience notes, select the most likely diagnosis.

### OUTPUT (JSON)
{{
    "diagnosis": "Single letter of your final diagnosis (A/B/C/D/E)",
    "reasoning": "brief reasoning within 1 line 3 sentences",
}}

## Guidelines
- output only json with 2 keys:"diagnosis" and "reasoning"
"""
                response_2 = agent.chat(prompt_step2, case['images'])
                response_2 = json.loads(json_cleaner(response_2))
                print(response_2)
                
                del agent
                torch.cuda.empty_cache()

                # 判断正确与否
                if case['diagnosis'].lower().strip(' .,!()[]') in options[response_2['diagnosis'].upper()].lower().strip(' .,!()[]'):
                    print('@ CORRECT! PASS')
                    if (t>0 and t % 10 == 0) or t==len(ds)-1:
                        bank.save()
                        print('saved')
                    continue

                print('@ FALSE! NOTE REQUIRED')
                
                # 第三轮：提供 Ground Truth -> 提炼并保存经验
                print('+++++ STAGE 2 STEP 3: NOTE +++++')
                agent = Agent(model_info=args.model)
                prompt_3 = f"""
### ROLE
You are a senior radiologist and medical education expert. You need to review the diagnosis a junior made and help him extracting an experience note, thus to prevent future false fatal diagnosis.
This is a medical reflection task to extract generalizable differential diagnosis knowledge.

### CASE INFORMATION
**Clinical History:** {case['clinical_history']}

### OPTIONS
{options}

### DIAGNOSTIC
**Junior's Diagnosis (Note that the option align with OPTIONS):** {response_2}

### Ground-truth Reference
**Correct Diagnosis:** {case['diagnosis']}
**Imaging Findings (from expert):** {case['imaging_findings']}
**Expert Discussion:** {case['discussion']}

### YOUR TASK

**Step 1: Confusion Points (Generalizable)**
Summarize 2-4 key points explaining why these two diagnoses are commonly confused in clinical practice.
- These must be generalizable medical knowledge, NOT case-specific
- Focus on overlapping features that lead to diagnostic confusion
- Each point should be a complete, actionable statement

**Step 2: Key Discriminators**
For EACH diagnosis, list 2-3 features that help distinguish it from the other.
- Focus on imaging features, clinical context, and lab findings
- Be specific and actionable

**Step 3: Decision Rule**
Provide a concise rule for differentiating these two diagnoses.

**Step 4: Error Analysis (Case-specific)**
In 1-2 sentences, explain what specific finding or reasoning was missed in THIS case.

### OUTPUT FORMAT (JSON)
{{  "department": "the most relevant department that the note belong to. MUST choose one from {DEPARTMENTS}.",
    "organ/region": "the most relvant organ/region that the note belong to. Must be standarized and choose from pre-defined organs under the department from {DEPARTMENT_ORGANS}.",
    "differential diagnosis pair": "{case['diagnosis']} v.s. {options[response_2['diagnosis'].upper().strip()]}", // use key name "differential diagnosis pair" instead of "differential_diagnosis_pair"
    "explanation of {case['diagnosis']}": "1 sentence: extract the medical description and definition (what it is) of the diagnosis if ground-truth provided (if applicable)",
    "confusion points": [
        "1. e.g. Both diagnosis can present with...",
        "2. e.g. Imaging overlap includes...",
        "3. e.g. Clinical presentations may be similar when..."
    ],
    "key discriminators": "What discriminates one diagnosis from another, and other differentials if applicable. If the discriminators of both diagnosis are provided by ground-truth information, extract both, else only extract the one with solid information ...",
    "decision rule": "If [X1] → favor {case['diagnosis']}, if [X2] → exclude/unlikey {case['diagnosis']}; If [Y1] → favor {options[response_2['diagnosis'].upper().strip()]}, if [Y2] → exclude/unlikey {options[response_2['diagnosis'].upper().strip()]}; If [Z] → consider alternatives [other diagnosis] (if applicable)",
    "error analysis": "decompose junior's chain of thought and point out how it made mistakes: e.g. scenario → junior thinking node 1 → ... → thinking node n [mistake: incorrectly interpretate ..., miss ...] → ..."
}}

### GUIDELINES
- The department should from {DEPARTMENTS}
- Focus on IMAGING-based differentiation
- Confusion points should be transferable to future cases
- Key discriminators should be observable on imaging or known from clinical history
- Decision rule should be practical and memorable, use words that are not persuasive like 'favor', 'unlikely', 'consider'...
- Use specific diagnosis name instead of ambiguous 'Diagnosis A'
- Make use of Ground-truth Reference and prevent medical mistakes
- For "unrelated_factors", only include factors explicitly present in this case that are misleading or non-discriminative.
"""

                final_response = agent.chat(prompt_3)
                final_response = json.loads(json_cleaner(final_response))

                # if case['diagnosis'].lower().strip(' .,!()[]') not in options[response_2['diagnosis'].upper()].lower().strip(' .,!()[]'):
                #     final_response['need_record'] = 'yes'

                # if final_response['need_record'].lower().strip() in ['no', 'no note needed']:
                #     print('No note needed. PASS.')
                #     if (t>0 and t % 10 == 0) or t==len(ds)-1:
                #         bank.save()
                #         print('saved')
                #     continue
            
                print('note needed. Extracting note and indexing into bank...')
                try:
                    print(final_response)
                    bank.add_note(final_response['department'], final_response['organ/region'], case['diagnosis'], final_response)
                    print('[Note Added]')
                except Exception as e:
                    print(f"Error indexing case {i}: {e}")
            
                del agent
                torch.cuda.empty_cache()

                if (t>0 and t % 10 == 0) or t==len(ds)-1:
                    bank.save()
                    print('saved')

            except:
                continue

    # ==========================================
    # 模式二：测试模式 (Test Mode)
    # ==========================================
    elif args.mode == 'test':
        search_tools = SearchTools()
        ds = EuroradDataset("data/bench/2025_cases.jsonl",split='test',split_index=0)
        stats = {"notes": 0, "notes_correct": 0, "correct": 0, "partial_correct": 0, "total": 0, "note_case_plain": 0, "w/o note correct": 0}
        skeleton_summary = bank.get_skeleton_2layers()
        dirty_id = []
        a_list = []
        b_list = []
        c_list = []
        d_list = []
        for i in tqdm(range(len(ds))):
            # 注意：测试集需要加载真实图像
            try:
                case = ds[i]
                # if case['case_id'] not in test_options.keys() or test_options[case['case_id']] == [] or case['case_id'] in ['18843', '18886', '18915', '18921', '18927', '18965', '18987', '19060', '19152', '19202', '19242', '19252', '18847', '18935', '18951', '18952', '19009', '19159', '19250']:
                #     continue
                if case['case_id'] not in test_options.keys() or test_options[case['case_id']] == [] or int(case['case_id']) not in test_case_ids:
                    continue
                
                option_list = test_options[case['case_id']]
                options = {}
                for k in range(len(option_list)):
                    options[f'{chr(ord('A')+k)}'] = option_list[k]
            
                # 第一轮：提供病例描述 + 图像专家解读 + 目录 -> 获取检索参数
                print('===============================================================')
                print(case['case_id'])
                print(case['clinical_history'])
                print(options)
                diffs = option_list
                agent = Agent(model_info=args.model)
                prompt_step1 = f"""
### ROLE
You are a radiologist analyzing a new case and giving the diagnosis. You have access to an Experience Bank containing diagnostic notes from previous cases to help you.

### CLINICAL HISTORY
{case['clinical_history']}

### OPTIONS
{options}

### EXPERIENCE BANK DIRECTORY
The Experience Bank is organized as: Department → Organ/Region → Notes
Select the most related department and organ/region for this case and the related notes will be retrieved automatically.
[Current coverage]:
{skeleton_summary}

### TASK
1. Form a preliminary impression based on clinical history and images
2. Decide what deparments and organs in ### EXPERIENCE BANK DIRECTORY are most related to this case
3. Give 1 or 2 [DEPT, ORGAN] searching pair to retrieve notes

### STRICT OUTPUT FORMAT (JSON)
{{
    "query": [["DEPT1: one from current [Current coverage] department list","ORGAN1:specific organ or region from [Current coverage] under department you choose"],["DEPT2","ORGAN2"]],  // You can search 1 or 2 [DEPT, ORGAN] pair to enable good notes retrieval
}}

### GUIDELINES
- Request retrieval when you're uncertain and need help in a case for better accuracy
- Must contain all predefined keys 
- DEPT2 can be the same as DEPT1, but [DEPT1, ORGAN1] should never be as same as [DEPT2, ORGAN2]
- If the case are closely related to different departments/organs, search 2 [DEPT, ORGAN] pairs. Else search 1 is enough to cover the case
- DEPT and ORGAN you search mMUST exist in [Current coverage] and ORGAN MUST under DEPT
"""

                response_1 = agent.chat(prompt_step1, case['images'])
                response_1 = json.loads(json_cleaner(response_1))
                flag = 0
                use_notes = 0
                # if 'false' in str(response_1["retrieval_decision"]["need_retrieval"]):
                #     retrieved_notes = ['No relevant experience note can be found. Please make diagnosis with your own experience.']
                #     print('[SEARCH] NO')
                #     flag = 0
                if 0:
                    flag = 0
                # 执行检索
                else:
                    try:
                        query = response_1["query"]
                        print(query)
                        retrieved_notes, matched_kws = bank.query_flexible(query, case['clinical_history'], diffs)
                        if len(retrieved_notes)==0 or retrieved_notes[0]=='No experience note can be found.':
                            flag = 0
                            use_notes = 0
                        else:
                            flag = 1
                            use_notes = 1
                    except:
                        retrieved_notes, matched_kws = ['No relevant experience note can be found. Please make diagnosis with your own experience.'], []
                        flag = 0
                    retrieved_notes = [f'{retrieved_notes[i]}' for i in range(len(retrieved_notes))]
                    print('[NOTES]','\n'.join(retrieved_notes[:min(3,len(retrieved_notes))]))

                del agent
                torch.cuda.empty_cache()

                # Step 
                # use_notes = 0
                agent = Agent(model_info=args.model)
                if use_notes == 0: 
                    prompt_step2 = f"""
### ROLE
You are a senior radiologist making a diagnostic decision.

### CLINICAL HISTORY
{case['clinical_history']}

### OPTIONS
{options}

### TASK
Analyze the images and clinical history systematically, then select the most likely diagnosis.

Analysis Thinking:
1. Identify: What is the primary abnormality? (location, size, composition)
2. Characterize: Key imaging features (borders, enhancement, signal/density)
3. Correlate: How do findings match the clinical history?
4. Differentiate: What features distinguish between the options?
5. Decide: What's the final diagnosis?

### OUTPUT (JSON)
{{
    "diagnosis": "Single letter of your final diagnosis (A/B/C/D/E)",
    "reasoning": "brief reasoning within 1 line 3 sentences",
}}
"""
#                     prompt_step2 = f"""
# ### ROLE
# You are a senior radiologist making a diagnostic decision.

# ### CLINICAL HISTORY
# {case['clinical_history']}

# ### OPTIONS
# {options}

# ### TASK
# Analyze the images and clinical history systematically, then select the most likely diagnosis.

# Analysis Thinking:
# 1. Identify: What is the primary abnormality? (location, size, composition)
# 2. Characterize: Key imaging features (borders, enhancement, signal/density)
# 3. Correlate: How do findings match the clinical history?
# 4. Differentiate: What features distinguish between the options?
# 5. Decide: What's the final diagnosis?

# ### OUTPUT (JSON)
# {{
#     "diagnosis": "Single letter of your final diagnosis (A/B/C/D/E)",
#     "reasoning": "brief reasoning within 1 line 3 sentences",
# }}
# """
                else:
                    prompt_step2 = f"""
### ROLE
You are a senior radiologist making a diagnostic decision.

### CLINICAL HISTORY
{case['clinical_history']}

### OPTIONS
{options}

### OUTER EXPERIENCE NOTES
The following notes describe commonly confused diagnosis pairs. Use them to guide your differential reasoning.
{retrieved_notes}

### TASK
Analyze systematically, leveraging the experience notes where relevant.

Analysis Thinking:
1. Identify: What is the primary abnormality? (location, size, composition)
2. Characterize: Key imaging features (borders, enhancement, signal/density)
3. Match Notes: Which experience notes cover diagnoses in the current options? 
4. Apply Discriminators: For matched notes, check if the case findings align with the `key_discriminators` for either diagnosis.
5. Apply Decision Rules: Follow the `decision_rule` — does the case satisfy conditions favoring one diagnosis over another?
6. Correlate: How do findings match the clinical history?
7. Decide: Based on imaging features, clinical context, and experience notes, select the most likely diagnosis.

### OUTPUT (JSON)
{{
    "diagnosis": "Single letter of your final diagnosis (A/B/C/D/E)",
    "reasoning": "brief reasoning within 1 line 3 sentences",
}}

## Guidelines
- output only json with 2 keys:"diagnosis" and "reasoning"
"""
                response_2 = agent.chat(prompt_step2, case['images'])
                response_2 = json.loads(json_cleaner(response_2))
                print('Agent Diagnosis:', response_2)
                
                del agent
                torch.cuda.empty_cache()
                

                flag=0
                print('GT: ', case['diagnosis'], '\nGT Discussion: ', case['discussion'])
                if case['diagnosis'].lower().strip() in options[response_2['diagnosis'].upper()].lower().strip():
                    stats["correct"] += 1
                    stats["notes_correct"] += use_notes
                    stats["w/o note correct"] += 1 - use_notes
                    if use_notes:
                        flag = 1
                        a_list.append(case['case_id'])
                    else:
                        c_list.append(case['case_id'])
                else:
                    if use_notes:
                        b_list.append(case['case_id'])
                    else:
                        d_list.append(case['case_id'])

#                 if use_notes == 1:
#                     agent = Agent(model_info=args.model)
#                     prompt_step2 = f"""
# ### ROLE
# You are a senior radiologist making a diagnostic decision.

# ### CLINICAL HISTORY
# {case['clinical_history']}

# ### OPTIONS
# {options}

# ### TASK
# Analyze the images and clinical history systematically, then select the most likely diagnosis.

# Analysis Thinking:
# 1. Identify: What is the primary abnormality? (location, size, composition)
# 2. Characterize: Key imaging features (borders, enhancement, signal/density)
# 3. Correlate: How do findings match the clinical history?
# 4. Differentiate: What features distinguish between the options?
# 5. Decide: What's the final diagnosis?

# ### OUTPUT (JSON)
# {{
#     "diagnosis": "Single letter of your final diagnosis (A/B/C/D/E)",
#     "reasoning": "brief reasoning within 1 line 3 sentences",
# }}
# """                 
#                     response_2 = agent.chat(prompt_step2, case['images'])
#                     response_2 = json.loads(json_cleaner(response_2))
#                     print('w/o note response: ', response_2)
#                     if case['diagnosis'].lower().strip() in options[response_2['diagnosis'].upper()].lower().strip():
#                         stats["note_case_plain"] += 1
#                         stats["w/o note correct"] += 1
                    #     if flag == 0:
                    #         w_list.append(case['case_id'])
                    #     else:
                    #         both_c.append(case['case_id'])
                    # elif flag==1:
                    #     c_list.append(case['case_id'])
                    # elif flag==0:
                    #     both_w.append(case['case_id'])
                    del agent

                stats["total"] += 1
                stats['notes'] += use_notes
                print(f"Note: {stats["note_case_plain"]} / {stats['notes_correct']} / {stats['notes']} / {stats['total']}")
                print(f"Note: {stats["w/o note correct"]} / {stats['total']}")
                print(f"Accuracy: {stats['correct']}/{stats['total']} {stats['correct'] / stats['total'] * 100:.2f}%")
                
            except:
                continue
            
        print(f"Final Accuracy: {stats['correct'] / stats['total'] * 100:.2f}%")
        print(a_list)
        print(b_list)
        print(c_list)
        print(d_list)

if __name__ == "__main__":
    main()