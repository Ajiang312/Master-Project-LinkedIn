from playwright.sync_api import sync_playwright
import pandas as pd
import time
import csv
import os
import re
import math

def safe_text(page, selector):
    el = page.query_selector(selector)
    return el.inner_text().strip() if el else None

def get_total_pages(page):
    try:
        result_spans = page.query_selector_all('span[dir="ltr"]')
        for span in result_spans:
            text = span.inner_text().lower().strip()
            print("📊 Texte détecté :", text)
            match = re.search(r'([\d\s\u202f]+)\s*résultat', text)
            if match:
                total = int(match.group(1).replace(' ', '').replace('\u202f', ''))
                return math.ceil(total / 25)
    except Exception as e:
        print("⚠️ Erreur lors du comptage des pages :", e)
    return 1


def prepare_visible_job_cards(page, max_scrolls=50):
    container = page.query_selector('.scaffold-layout__list')
    if not container:
        print("❌ Colonne gauche introuvable")
        return

    last_count = 0
    stable_scrolls = 0

    for i in range(max_scrolls):
        page.evaluate("""
            () => {
                const container = document.querySelector('.scaffold-layout__list');
                if (container) {
                    container.scrollBy(0, container.clientHeight / 2);
                }
            }
        """)
        time.sleep(0.8)

        items = page.query_selector_all('li.scaffold-layout__list-item')
        for item in items:
            try:
                item.hover()
            except:
                continue

        current_count = len(page.query_selector_all('.job-card-container'))

        if current_count == last_count:
            stable_scrolls += 1
            if stable_scrolls >= 3:
                break
        else:
            stable_scrolls = 0
            last_count = current_count

    print(f"✅ {last_count} annonces prêtes à scraper.")

def extract_job_id_from_url(url):
    match = re.search(r'/jobs/view/(\d+)', url)
    return match.group(1) if match else None

def extract_location_and_posted(page):
    container = page.query_selector('.job-details-jobs-unified-top-card__primary-description-container')
    if not container:
        return None, None

    spans = container.query_selector_all('span.tvm__text')
    visible_texts = [span.inner_text().strip() for span in spans if span.inner_text().strip()]
    location = visible_texts[0] if len(visible_texts) > 0 else None
    posted = next((t for t in visible_texts if "il y a" in t.lower()), None)
    return location, posted

def extract_tokens_split(page):
    contract_format = None
    work_mode = None

    lis = page.query_selector_all('div.mt2.mb2 li.job-details-jobs-unified-top-card__job-insight')
    for li in lis:
        try:
            span = li.query_selector('span.text-body-small')
            if not span or 'visually-hidden' in span.get_attribute('class'):
                continue
            for hidden in span.query_selector_all('.visually-hidden'):
                hidden.evaluate("e => e.remove()")
            text = span.inner_text().strip()
            if text:
                contract_format = text
                break
        except:
            continue

    spans = page.query_selector_all('span.job-details-jobs-unified-top-card__job-insight-view-model-secondary')
    for span in spans:
        try:
            for hidden in span.query_selector_all('.visually-hidden'):
                hidden.evaluate("e => e.remove()")
            text = span.inner_text().strip()
            if text:
                work_mode = text
                break
        except:
            continue

    return contract_format, work_mode

def extract_description(page):
    container = page.query_selector('div.jobs-box__html-content')
    if not container:
        return None

    paragraphs = container.query_selector_all('p[dir="ltr"]')
    text_parts = []

    for p in paragraphs:
        for br in p.query_selector_all('br'):
            br.evaluate("el => el.replaceWith(document.createTextNode(' '))")

        text = p.inner_text().replace('\n', ' ').replace('\r', ' ').strip()
        if text:
            text_parts.append(text)

    return "\n\n".join(text_parts).strip() if text_parts else None

def extract_qualifications(page):
    qualifications = []
    try:
        # Cliquer sur le bouton pour ouvrir le modal
        qual_button = page.query_selector("button span.artdeco-button__text:text('Détails sur les qualifications')")
        if qual_button:
            qual_button.click()
            time.sleep(1.5)  # laisser le temps au modal de s'ouvrir

            # Attendre que la liste soit visible
            page.wait_for_selector("ul.job-details-skill-match-status-list", timeout=15000, state="visible")

            # Récupérer les compétences
            skill_items = page.query_selector_all("ul.job-details-skill-match-status-list li div")
            qualifications = [el.inner_text().strip() for el in skill_items if el.inner_text().strip()]

            time.sleep(0.5)  # donner le temps de charger entièrement les items

            # Fermer le modal
            close_btn = page.query_selector("button span.artdeco-button__text:text('Terminé')")
            if close_btn:
                close_btn.click()
            else:
                cross_btn = page.query_selector(".artdeco-modal__dismiss")
                if cross_btn:
                    cross_btn.click()
    except Exception as e:
        print("⚠️ Erreur extraction qualifications:", e)

    return ", ".join(qualifications)


def scrape_linkedin_jobs(base_url, output_file="linkedin-ingenieur-database.csv"):
    existing_job_ids = set()
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        if 'job_id' in existing_df.columns:
            existing_job_ids = set(existing_df['job_id'].dropna().astype(str).tolist())

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(storage_state="linkedin_auth.json")
        page = context.new_page()

        page.goto(base_url, timeout=60000)
        total_pages = get_total_pages(page)
        print(f"🔢 {total_pages} pages détectées pour cette recherche.")

        for page_index in range(total_pages):
            start_param = page_index * 25
            full_url = f"{base_url}&start={start_param}"
            print(f"\n📄 Chargement de la page {page_index + 1} : {full_url}")
            page.goto(full_url, timeout=60000)

            prepare_visible_job_cards(page)
            job_cards = page.query_selector_all('.job-card-container')
            print(f"🔍 {len(job_cards)} annonces détectées.")

            if len(job_cards) == 0:
                print("⛔ Fin : aucune annonce trouvée.")
                break

            for i in range(len(job_cards)):
                try:
                    visible_cards = page.query_selector_all('.job-card-container')
                    if i >= len(visible_cards):
                        break

                    job_card = visible_cards[i]
                    link_el = job_card.query_selector('a')
                    link = link_el.get_attribute('href') if link_el else None
                    if link and not link.startswith("http"):
                        link = "https://www.linkedin.com" + link

                    job_id = extract_job_id_from_url(link)
                    if not link or not job_id or job_id in existing_job_ids:
                        continue  # déjà vu

                    job_card.scroll_into_view_if_needed()
                    job_card.click()
                    time.sleep(3)
                    page.wait_for_selector('div.jobs-box__html-content', timeout=5000)

                    location, posted = extract_location_and_posted(page)
                    contract_format, work_mode = extract_tokens_split(page)
                    description = extract_description(page)
                    qualifications = extract_qualifications(page)

                    job = {
                        'job_id': job_id,
                        'title': safe_text(page, '.job-details-jobs-unified-top-card__job-title a'),
                        'company': safe_text(page, '.job-details-jobs-unified-top-card__company-name'),
                        'location': location,
                        'posted': posted,
                        'contract_format': contract_format,
                        'work_mode': work_mode,
                        'qualifications': qualifications,
                        'description': description,
                        'url': link
                    }


                    existing_job_ids.add(job_id)
                    new_df = pd.DataFrame([job])
                    write_header = not os.path.exists(output_file)
                    new_df.to_csv(output_file, mode='a', index=False, header=write_header,
                                  encoding='utf-8-sig', quoting=csv.QUOTE_ALL)

                    print(f"✅ Ajouté : {job['title']} chez {job['company']}")

                except Exception as e:
                    print(f"❌ Erreur carte {i + 1} :", e)

        browser.close()
        print("🎉 Scraping terminé.")

# === Lancement ===
base_url = "https://www.linkedin.com/jobs/search?keywords=Ing%C3%A9nieur&location=Ile-de-France"
scrape_linkedin_jobs(base_url)

