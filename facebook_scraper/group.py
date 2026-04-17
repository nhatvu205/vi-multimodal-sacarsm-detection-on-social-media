from playwright.sync_api import sync_playwright
import time
import os
import json
import random
import requests
import re

GROUP_URL = "https://www.facebook.com/groups/987761062274391/?sorting_setting=CHRONOLOGICAL&locale=vi_VN"
TARGET = 300
SAVE_DIR = "dataset_post_5"
IMG_DIR = os.path.join(SAVE_DIR, "images")
JSON_PATH = os.path.join(SAVE_DIR, "data.json")

os.makedirs(IMG_DIR, exist_ok=True)

if os.path.exists(JSON_PATH):
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
else:
    data = []


def extract_fb_img_id(url):
    """Lấy phần ID ổn định từ URL Facebook CDN, bỏ token hay thay đổi"""
    match = re.search(r'/([0-9]{9,})[_.]', url)
    return match.group(1) if match else url

seen_urls = set(
    extract_fb_img_id(item.get("image_url", ""))
    for item in data if "image_url" in item
)
post_id = max((item.get("id", 0) for item in data), default=0) + 1
collected = len(data)


def download_image(url, path):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://www.facebook.com/"
        }
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200 and len(r.content) > 10000:
            with open(path, "wb") as f:
                f.write(r.content)
            return True
    except Exception as e:
        print(f"⚠️ Download lỗi: {e}")
    return False


def expand_see_more(page, post_el):
    try:
        page.evaluate("""(postEl) => {
            const buttons = postEl.querySelectorAll('div[role="button"], span[role="button"]');
            buttons.forEach(btn => {
                const text = btn.innerText.trim();
                if (text === 'Xem thêm' || text === 'See more' || text.includes('Xem thêm')) {
                    btn.click();
                }
            });
        }""", post_el)
        time.sleep(0.8)
    except:
        pass


def get_text_from_post(post_el):
    try:
        text = post_el.evaluate("""(el) => {
            let messageBody = el.querySelector('[data-ad-comet-preview="message"]') || el;

            function extractCleanContent(root) {
                let result = "";
                const treeWalker = document.createTreeWalker(
                    root,
                    NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT,
                    null,
                    false
                );

                let node;
                while (node = treeWalker.nextNode()) {
                    if (node.nodeType === 3) {
                        const parent = node.parentElement;
                        if (!parent) continue;

                        const style = window.getComputedStyle(parent);
                        const rect = parent.getBoundingClientRect();

                        if (style.display !== 'none' &&
                            style.visibility !== 'hidden' &&
                            style.opacity !== '0' &&
                            rect.width > 0 &&
                            style.position !== 'absolute') {
                            result += node.textContent;
                        }
                    } else if (node.nodeType === 1 && node.tagName === 'IMG') {
                        const alt = node.getAttribute('alt');
                        if (alt) {
                            result += " " + alt + " ";
                        }
                    }
                }
                return result;
            }

            const contentSource = messageBody.querySelector('h3, strong, [dir="auto"]') || messageBody;
            return extractCleanContent(contentSource);
        }""")

        if not text:
            return ""

        text = text.strip()
        half = len(text) // 2
        if len(text) > 20 and text[:half].strip() == text[half:].strip():
            text = text[:half].strip()

        text = re.sub(r'(\b[a-zA-Z0-9]\s){2,}[a-zA-Z0-9]\b', '', text)
        text = re.sub(r'\S+\.[a-z]{2,4}\b', '', text)
        text = re.sub(r'\b[A-Za-z0-9]{15,}\b', '', text)
        text = ' '.join(text.split())

        return text.strip()

    except Exception as e:
        print(f"Lỗi lấy text chi tiết: {e}")
        return ""


def get_posts(page):
    results = []
    seen_post_containers = set()

    img_elements = page.query_selector_all('img[data-imgperflogname="feedImage"]')
    print(f"  → Tìm thấy {len(img_elements)} ảnh feedImage")

    for i, img_el in enumerate(img_elements):
        try:
            # Bỏ qua video thumbnail
            is_video_thumb = page.evaluate("""(img) => {
                let el = img;
                for (let i = 0; i < 8; i++) {
                    el = el.parentElement;
                    if (!el) break;
                    if (el.querySelector('video') ||
                        el.querySelector('[aria-label="Video"]') ||
                        el.querySelector('[data-sigil="inlineVideo"]') ||
                        (el.getAttribute('aria-label') || '').toLowerCase().includes('video')) {
                        return true;
                    }
                }
                return false;
            }""", img_el)

            if is_video_thumb:
                print(f"  [{i}] ↷ Bỏ qua video thumbnail")
                continue

            # Leo lên tìm post container
            post_el = page.evaluate_handle("""(img) => {
                let el = img;
                for (let i = 0; i < 25; i++) {
                    el = el.parentElement;
                    if (!el) return null;
                    const action = el.querySelector('[aria-label*="Chia sẻ"], [aria-label*="Hành động với bài viết"], [aria-label*="Share"]');
                    const name = el.querySelector('h2, h3, strong, a[role="link"]');
                    if (action || (name && el.querySelector('div[dir="auto"]'))) {
                        return el;
                    }
                }
                return null;
            }""", img_el)

            if not post_el:
                print(f"  [{i}] SKIP: không tìm được post container")
                continue

            bbox = post_el.bounding_box()
            if not bbox:
                print(f"  [{i}] SKIP: bbox = None")
                continue
            if bbox["height"] < 250:
                print(f"  [{i}] SKIP: bbox height={bbox['height']:.0f} < 250")
                continue

            container_id = page.evaluate("""(el) => {
                if (!el._crawlId) el._crawlId = Math.random().toString(36).slice(2);
                return el._crawlId;
            }""", post_el)

            if container_id in seen_post_containers:
                print(f"  [{i}] SKIP: container đã xử lý (ảnh đôi)")
                continue
            seen_post_containers.add(container_id)

            img_src = img_el.get_attribute("src")
            if not img_src or img_src.startswith("data:"):
                print(f"  [{i}] SKIP: img_src rỗng hoặc data URI")
                continue
            if extract_fb_img_id(img_src) in seen_urls:
                print(f"  [{i}] SKIP: đã thu thập rồi")
                continue

            expand_see_more(page, post_el)
            caption = get_text_from_post(post_el)

            if len(caption) < 20:
                print(f"  [{i}] SKIP: text quá ngắn ({len(caption)} chars): '{caption[:50]}'")
                continue

            print(f"  [{i}] OK: {caption[:60]}...")
            results.append({"caption": caption, "img_src": img_src})

        except Exception as e:
            print(f"  [{i}] Exception: {e}")
            continue

    print(f"Tìm thấy {len(results)} post có ĐỦ ảnh + text")
    return results


def human_scroll(page, times=1):
    for _ in range(times):
        # Scroll vừa phải để Facebook kịp lazy-load, tránh virtualize post
        page.evaluate("window.scrollBy(0, window.innerHeight * 1.8 + 600)")
        time.sleep(random.uniform(2.0, 3.5))

with sync_playwright() as p:
    browser = p.chromium.connect_over_cdp("http://localhost:9222")
    context = browser.contexts[0]

    page = next((pg for pg in context.pages if "facebook.com" in pg.url), None)
    if not page or GROUP_URL not in page.url:
        page = context.new_page()
        page.goto(GROUP_URL)

    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except:
        pass

    print("Chờ feed load xong...")
    time.sleep(4)

    print(f"Đang crawl Group - TARGET: {TARGET} post (BẮT BUỘC cả ảnh + text)")

    no_new_rounds = 0

    while collected < TARGET:
        posts = get_posts(page)

        new_this_round = 0
        for item in posts:
            if extract_fb_img_id(item["img_src"]) in seen_urls:
                continue

            img_path = os.path.join(IMG_DIR, f"post_{post_id}.jpg")

            if download_image(item["img_src"], img_path):
                data.append({
                    "id": post_id,
                    "text": item["caption"],
                    "image_path": img_path.replace("\\", "/"),
                    "image_url": item["img_src"]
                })
                seen_urls.add(extract_fb_img_id(item["img_src"]))
                collected += 1
                new_this_round += 1
                print(f"[{collected}/{TARGET}] {item['caption'][:90]}...")

            post_id += 1
            if collected >= TARGET:
                break

            time.sleep(random.uniform(1.2, 2.5))

        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Tiến độ: {collected} / {TARGET} post")

        if collected >= TARGET:
            break

        if new_this_round == 0:
            no_new_rounds += 1
            print(f"Không có post mới ({no_new_rounds}/15)")
            if no_new_rounds >= 15:
                print("Dừng vì không còn post mới.")
                break
            human_scroll(page, times=6)
        else:
            no_new_rounds = 0
            human_scroll(page, times=1)

    print(f"\n HOÀN THÀNH — Tổng {len(data)} post (có cả ảnh + text)")
    print(f"   Ảnh lưu tại: {IMG_DIR}")