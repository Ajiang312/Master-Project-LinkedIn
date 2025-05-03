from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://www.linkedin.com/login")
    input("✅ Connecte-toi à LinkedIn, puis appuie sur Entrée ici...")
    context.storage_state(path="linkedin_auth.json")
    browser.close()

print("✅ Session enregistrée dans linkedin_auth.json")
