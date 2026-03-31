from playwright.sync_api import sync_playwright
import sys

URL = "https://final-project-2vl3fcyz5qpc5mbhpfy26a.streamlit.app"

def main():
    with sync_playwright() as p:
        print(f"Launching browser to visit {URL}...")
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        
        try:
            # Navigate to the app, waiting for network idle is usually best for SPAs
            page.goto(URL, wait_until="domcontentloaded", timeout=60000)
            print("Page loaded. Waiting a few seconds for Streamlit JS to initialize...")
            page.wait_for_timeout(5000)
            
            # Look for the wake up button in case the app went to sleep
            try:
                button = page.get_by_role("button", name="Yes, get this app back up!")
                if button.is_visible():
                    print("App is sleeping. Clicking wake up button...")
                    button.click()
                    # Wait for the app to wake up and reload
                    page.wait_for_timeout(20000)
                    print("Wake up process triggered and waited.")
                else:
                    print("No wake up button found. App is awake!")
            except Exception as e:
                print("No wake up button found. App is awake!")
                
            print("Ping completed successfully.")
        except Exception as e:
            print(f"Error navigating: {e}")
            sys.exit(1)
        finally:
            browser.close()

if __name__ == "__main__":
    main()
