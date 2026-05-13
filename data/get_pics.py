import json
import os
import re
import time
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urljoin, urlparse, unquote
from pathlib import Path
import ssl
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

# ==================== 配置参数 ====================
JSONL_FILE = "all_parsed_cases_with_meta.jsonl"
BASE_DIR = "figures"
BASE_URL = "https://www.eurorad.org"
CASES_URL = f"{BASE_URL}/cases"  # 作为 Referer 来源
MAX_RETRIES = 3
REQUEST_DELAY = (1.5, 2.5)  # 增加延迟避免触发风控
TIMEOUT = 30

# ==================== 关键修复：完整浏览器级请求头 ====================
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",  # 必须包含！否则返回403
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Referer": CASES_URL,  # 关键：模拟从案例列表页跳转
    "DNT": "1",
    "Cache-Control": "max-age=0",
}

# ==================== SSL 修复（绕过某些环境的证书问题） ====================
class TLSAdapter(HTTPAdapter):
    """强制使用 TLS 1.2+ 避免 SSL 错误"""
    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context()
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)

def create_session():
    """创建带 SSL 修复和重试机制的会话"""
    session = requests.Session()
    session.mount('https://', TLSAdapter())
    
    # 添加重试机制
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    
    return session

# ==================== 辅助函数 ====================
def extract_filename_from_url(url):
    """从URL提取安全文件名"""
    clean_url = url.split('?')[0]
    filename = unquote(urlparse(clean_url).path.split('/')[-1])
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    if not filename or '.' not in filename:
        filename = f"image_{int(time.time())}.jpg"
    return filename

def download_image(session, url, save_path, retry_count=0):
    """带重试的图片下载"""
    if os.path.exists(save_path):
        return True, "exists"
    
    try:
        response = session.get(
            url, 
            headers={**HEADERS, "Referer": BASE_URL},  # 图片请求使用基础 Referer
            timeout=TIMEOUT,
            stream=True
        )
        
        # 处理 429 限流
        if response.status_code == 429:
            wait_time = int(response.headers.get("Retry-After", 10))
            print(f"\n[Rate Limit] Waiting {wait_time}s (case: {save_path.parent.name})")
            time.sleep(wait_time)
            return download_image(session, url, save_path, retry_count)
        
        if response.status_code == 200:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=16384):
                    if chunk:
                        f.write(chunk)
            return True, "success"
        else:
            return False, f"HTTP {response.status_code}"
            
    except requests.exceptions.Timeout:
        error_msg = "timeout"
    except requests.exceptions.ConnectionError:
        error_msg = "connection"
    except Exception as e:
        error_msg = f"exception: {str(e)[:60]}"
    
    if retry_count < MAX_RETRIES:
        wait = 2 ** retry_count
        time.sleep(wait)
        return download_image(session, url, save_path, retry_count + 1)
    
    return False, error_msg

def parse_case_images(html_content, base_url, case_id):
    """
    精准提取高清原图URL（关键：从modal容器提取figure-modal-lazy）
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    image_urls = []
    
    # === 策略1: 优先提取 modal 中的高清图 (figure-modal-lazy) ===
    # Eurorad 的高清图存储在 class 包含 "figure-modal-lazy" 的 img 标签中
    modal_imgs = soup.find_all('img', class_=lambda x: x and 'figure-modal-lazy' in x)
    
    if modal_imgs:
        for img in modal_imgs:
            data_src = img.get('data-src') or img.get('src')
            if data_src and '/styles/figure_image/public/figure_image/' in data_src:
                clean_url = data_src.split('?')[0]  # 移除 ?itok 参数
                full_url = urljoin(base_url, clean_url)
                image_urls.append(full_url)
        return image_urls
    
    # === 策略2: 回退到 gallery 中的图片 (figure-image-lazy) ===
    gallery_imgs = soup.find_all('img', class_=lambda x: x and 'figure-image-lazy' in x)
    for img in gallery_imgs:
        data_src = img.get('data-src') or img.get('src')
        if data_src and '/figure_image/' in data_src:
            clean_url = data_src.split('?')[0]
            full_url = urljoin(base_url, clean_url)
            image_urls.append(full_url)
    
    # 去重
    return list(dict.fromkeys(u for u in image_urls if u.startswith('http')))

def process_case(session, case_data, pbar):
    """处理单个案例（关键修复：URL 去空格 + 完整请求头）"""
    case_id = str(case_data.get("case_id", "")).strip()
    case_url = case_data.get("url", "").strip()  # ✅ 关键修复：去除URL尾部空格！
    
    if not case_id or not case_url:
        pbar.write(f"[Skip] Invalid case data: {case_id}")
        return 0, 0
    
    # 验证URL格式
    if not case_url.startswith("http"):
        case_url = urljoin(BASE_URL, case_url)
    
    case_dir = Path(BASE_DIR) / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取页面内容（关键：使用完整请求头 + Referer）
    try:
        response = session.get(
            case_url, 
            headers={**HEADERS, "Referer": CASES_URL},  # 从案例列表页跳转
            timeout=TIMEOUT
        )
        response.raise_for_status()
        
        # 调试：验证是否成功获取页面
        if "Eurorad" not in response.text[:200]:
            pbar.write(f"[Warning] Unexpected content for {case_id} (status: {response.status_code})")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            pbar.write(f"[403 Forbidden] {case_id} - URL: '{case_url}' (check for spaces!)")
            # 尝试移除所有空格重试
            clean_url = re.sub(r'\s+', '', case_url)
            if clean_url != case_url:
                pbar.write(f"  → Retrying with cleaned URL: {clean_url}")
                try:
                    response = session.get(clean_url, headers=HEADERS, timeout=TIMEOUT)
                    response.raise_for_status()
                except Exception as e2:
                    pbar.write(f"  → Still failed: {str(e2)[:60]}")
                    return 0, 0
            else:
                return 0, 0
        else:
            pbar.write(f"[HTTP Error {e.response.status_code}] {case_id}: {str(e)[:60]}")
            return 0, 0
    except Exception as e:
        pbar.write(f"[Fetch Error] {case_id}: {str(e)[:70]}")
        return 0, 0
    
    # 解析高清图URL
    image_urls = parse_case_images(response.text, BASE_URL, case_id)
    
    if not image_urls:
        # 调试：输出部分HTML帮助诊断
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('h1')
        pbar.write(f"[No Images] {case_id} - Page title: {title.text.strip() if title else 'N/A'}")
        return 0, 0
    
    # 下载图片
    success_count = 0
    with tqdm(
        total=len(image_urls),
        desc=f"Case {case_id}",
        leave=False,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    ) as img_pbar:
        for idx, img_url in enumerate(image_urls, 1):
            filename = extract_filename_from_url(img_url)
            save_path = case_dir / filename
            
            success, status = download_image(session, img_url, str(save_path))
            
            if success:
                success_count += 1
                img_pbar.set_postfix(status="✓")
            else:
                img_pbar.set_postfix(status=f"✗ {status}")
                pbar.write(f"  [Fail] {case_id}/{filename}: {status}")
            
            img_pbar.update(1)
            time.sleep(REQUEST_DELAY[0] + (REQUEST_DELAY[1] - REQUEST_DELAY[0]) * idx / max(1, len(image_urls)))
    
    return len(image_urls), success_count

def main():
    # 验证输入文件
    if not os.path.exists(JSONL_FILE):
        print(f"❌ Error: '{JSONL_FILE}' not found!")
        print(f"   Current working directory: {os.getcwd()}")
        return
    
    # 统计案例数
    with open(JSONL_FILE, 'r', encoding='utf-8') as f:
        total_cases = sum(1 for line in f if line.strip())
    
    print(f"📊 Starting download for {total_cases} cases")
    print(f"📁 Output directory: {os.path.abspath(BASE_DIR)}")
    print(f"🌐 Base URL: {BASE_URL}")
    print(f"⚠️  Critical fixes applied:")
    print(f"   • URL whitespace stripping")
    print(f"   • Full browser-grade headers (including Accept-Encoding)")
    print(f"   • Referer spoofing from /cases page")
    print(f"   • TLS 1.2+ enforcement")
    print("-" * 70)
    
    # 创建会话
    session = create_session()
    total_images = 0
    total_success = 0
    
    # 处理所有案例
    with open(JSONL_FILE, 'r', encoding='utf-8') as f:
        with tqdm(
            total=total_cases,
            desc="Overall Progress",
            unit="case",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        ) as pbar:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    case_data = json.loads(line)
                    found, success = process_case(session, case_data, pbar)
                    total_images += found
                    total_success += success
                    
                    pbar.set_postfix(
                        imgs=f"{total_success}/{total_images}",
                        success=f"{total_success/total_images*100:.1f}%" if total_images else "0%"
                    )
                except json.JSONDecodeError as e:
                    pbar.write(f"[JSON Error L{line_num}] {str(e)}")
                except Exception as e:
                    pbar.write(f"[Unexpected L{line_num}] {str(e)[:70]}")
                    import traceback
                    traceback.print_exc()
                
                pbar.update(1)
                time.sleep(REQUEST_DELAY[0] * 2.0)  # 案例间延迟
    
    # 最终统计
    print("\n" + "="*70)
    print(f"✅ Download completed!")
    print(f"   Total cases processed: {total_cases}")
    print(f"   Total images found:    {total_images}")
    print(f"   Successfully downloaded: {total_success} ({total_success/total_images*100:.1f}%)")
    print(f"   Output directory: {os.path.abspath(BASE_DIR)}")
    print("="*70)

if __name__ == "__main__":
    Path(BASE_DIR).mkdir(exist_ok=True)
    
    # 测试连接（关键诊断步骤）
    print("🔍 Testing connection to Eurorad...")
    session = create_session()
    try:
        test_resp = session.get(
            CASES_URL,
            headers=HEADERS,
            timeout=15
        )
        print(f"   ✅ Connection successful (status: {test_resp.status_code})")
        if test_resp.status_code == 200 and "Eurorad" in test_resp.text:
            print("   ✅ Server response validated")
        else:
            print(f"   ⚠️  Unexpected response (length: {len(test_resp.text)})")
    except Exception as e:
        print(f"   ❌ Connection failed: {str(e)}")
        print("   → Common causes:")
        print("      • Network firewall blocking requests")
        print("      • Missing proxy configuration")
        print("      • SSL certificate issues (try: pip install --upgrade certifi)")
        exit(1)
    
    # 运行主程序
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user. Progress saved.")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()