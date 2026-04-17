from playwright.sync_api import sync_playwright
import time, os, json, random, requests

PAGE_URL = "https://www.facebook.com/hoinguoithucdungvietnam"
TARGET = 100
SAVE_DIR = "dataset_post_3"
IMG_DIR = os.path.join(SAVE_DIR, "images")
JSON_PATH = os.path.join(SAVE_DIR, "data.json")
os.makedirs(IMG_DIR, exist_ok=True)

if os.path.exists(JSON_PATH):
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
else:
    data = []

seen = set(item["text"] for item in data if "text" in item)
post_id = max((item["id"] for item in data), default=-1) + 1
collected = 0


def download_image(url, path):
    try:
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.facebook.com/"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
            return True
    except Exception as e:
        print(f"⚠️ Download ảnh lỗi: {e}")
    return False


def expand_see_more(page, post_el):
    try:
        page.evaluate("""(postEl) => {
            const btns = postEl.querySelectorAll('[role="button"][tabindex="0"]');
            btns.forEach(btn => {
                const t = btn.innerText.trim();
                if (t === 'Xem thêm' || t === 'See more' || t === 'See More') {
                    btn.click();
                }
            });
        }""", post_el)
        time.sleep(0.5)
    except:
        pass


def get_posts_with_image_and_text(page):
    results = []
    seen_positions = set()

    msg_handles = page.query_selector_all('[data-ad-comet-preview="message"]')

    for msg_el in msg_handles:
        try:
            post_el = page.evaluate_handle("""(msgEl) => {
                let el = msgEl;
                for (let i = 0; i < 15; i++) {
                    el = el.parentElement;
                    if (!el) return null;
                    const h2 = el.querySelector('h2');
                    const share = el.querySelector(
                        '[aria-label*="Chia sẻ"], [aria-label*="Share"], [aria-label*="Gửi nội dung"]'
                    );
                    if (h2 && share) return el;
                }
                return null;
            }""", msg_el)

            if not post_el:
                continue

            bbox = post_el.bounding_box()
            if not bbox:
                continue
            key = round(bbox["y"])
            if key in seen_positions:
                continue
            seen_positions.add(key)

            img_el = post_el.query_selector('img[data-imgperflogname="feedImage"]')
            if not img_el:
                continue

            img_src = img_el.get_attribute("src")
            if not img_src or img_src.startswith("data:"):
                continue

            expand_see_more(page, post_el)

            caption = page.evaluate("""(postEl) => {
                const container = postEl.querySelector('[data-ad-comet-preview="message"]');
                if (!container) return '';
                const nodes = container.querySelectorAll('[dir="auto"]');
                const parts = [];
                nodes.forEach(n => {
                    const t = n.innerText.trim();
                    if (t && t.length >= 3) parts.push(t);
                });
                return [...new Set(parts)].join(' ').trim();
            }""", post_el)

            if not caption or len(caption) < 5:
                continue

            results.append({
                "post_el": post_el,
                "caption": caption,
                "img_src": img_src,
            })

        except:
            continue

    print(f"✅ Tìm thấy {len(results)} post có đủ ảnh + text")
    return results


def human_scroll(page):
    page.evaluate("""() => {
        return new Promise((resolve) => {
            let total = 0;
            const target = 1600 + Math.random() * 600;
            const step = 70 + Math.random() * 50;
            const delay = 25 + Math.random() * 20;
            const timer = setInterval(() => {
                window.scrollBy(0, step);
                total += step;
                if (total >= target) { clearInterval(timer); resolve(); }
            }, delay);
        });
    }""")
    time.sleep(random.uniform(2.5, 4.0))


with sync_playwright() as p:
    browser = p.chromium.connect_over_cdp("http://localhost:9222")
    context = browser.contexts[0]

    page = next(
        (pg for pg in context.pages if "facebook.com" in pg.url),
        None
    )
    if page:
        if PAGE_URL not in page.url:
            page.goto(PAGE_URL)
    else:
        page = context.new_page()
        page.goto(PAGE_URL)

    try:
        page.wait_for_load_state("networkidle", timeout=10000)
    except:
        time.sleep(3)

    print(f"🚀 Đang quét: {page.url}")

    no_new_rounds = 0

    while collected < TARGET:
        posts = get_posts_with_image_and_text(page)

        new_this_round = 0
        for p_data in posts:
            caption = p_data["caption"]
            img_src = p_data["img_src"]

            if caption in seen:
                continue

            img_filename = f"post_{post_id}.jpg"
            img_path = os.path.join(IMG_DIR, img_filename)
            ok = download_image(img_src, img_path)
            if not ok:
                print(f"⚠️ Bỏ qua post {post_id} do không download được ảnh")
                continue

            data.append({
                "id": post_id,
                "text": caption,
                "image_path": img_path,
                "image_url": img_src,
            })
            seen.add(caption)
            collected += 1
            print(f"[{collected}/{TARGET}] {caption[:70]}...")
            post_id += 1
            new_this_round += 1

            if collected >= TARGET:
                break

        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        if collected >= TARGET:
            break

        if new_this_round == 0:
            no_new_rounds += 1
            print(f"⚠️ Không có post mới ({no_new_rounds}/5)")
            if no_new_rounds >= 5:
                print("🏁 Hết post mới, dừng.")
                break
        else:
            no_new_rounds = 0

        human_scroll(page)

    print(f"🏁 DONE — Tổng {len(data)} post trong {JSON_PATH}")