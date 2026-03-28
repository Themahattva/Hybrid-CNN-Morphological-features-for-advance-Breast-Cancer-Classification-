import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
import time
from webdriver_manager.chrome import ChromeDriverManager

class WebsiteRegistrationBot:
    def __init__(self, excel_file_path, website_url, headless=False):
        self.excel_file_path = excel_file_path
        self.website_url = website_url
        self.headless = headless
        self.driver = None
        self.data = None

    def setup_driver(self):
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        self.driver.maximize_window()

    def load_excel_data(self):
        try:
            self.data = pd.read_excel(self.excel_file_path)
            print(f"Loaded {len(self.data)} records from Excel file")
            return True
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            return False

    def wait_for_element(self, by, value, timeout=10):
        try:
            return WebDriverWait(self.driver, timeout).until(
                EC.element_to_be_clickable((by, value))
            )
        except Exception as e:
            print(f"Element not found: {by}={value}. Error: {e}")
            return None

    def click_initial_register_button(self):
        print("Looking for Register button on landing page...")
        selectors = [
            "//button[contains(text(),'Register')]",
            "//a[contains(text(),'Register')]",
            "//div[contains(text(),'Register')]",
            "//span[contains(text(),'Register')]",
            "//button[contains(@class,'register')]",
            "//a[contains(@class,'register')]"
        ]
        for selector in selectors:
            btn = None
            try:
                btn = self.wait_for_element(By.XPATH, selector, timeout=5)
                if btn:
                    print(f"Found Register button: {selector}")
                    btn.click()
                    time.sleep(2)
                    return True
            except:
                continue
        print("Register button not found on landing page")
        return False

    def fill_registration_form(self, name, email):
        print("Filling registration form on new page...")
        time.sleep(2)
        name_selectors = [
            "input[name='name']", "input[placeholder*='Name' i]", "//input[@placeholder='Name']"
        ]
        for selector in name_selectors:
            field = None
            try:
                if selector.startswith("//"):
                    field = self.wait_for_element(By.XPATH, selector, timeout=4)
                else:
                    field = self.wait_for_element(By.CSS_SELECTOR, selector, timeout=4)
                if field:
                    field.clear()
                    field.send_keys(name)
                    print(f"Filled name: {name}")
                    break
            except:
                continue

        email_selectors = [
            "input[name='email']", "input[type='email']", "input[placeholder*='Email' i]", "//input[@type='email']"
        ]
        for selector in email_selectors:
            field = None
            try:
                if selector.startswith("//"):
                    field = self.wait_for_element(By.XPATH, selector, timeout=4)
                else:
                    field = self.wait_for_element(By.CSS_SELECTOR, selector, timeout=4)
                if field:
                    field.clear()
                    field.send_keys(email)
                    print(f"Filled email: {email}")
                    break
            except:
                continue

    def submit_registration_form(self):
        print("Submitting registration form...")
        selectors = [
            "//button[@type='submit']",
            "//input[@type='submit']",
            "//button[contains(text(),'Register')]"
        ]
        for selector in selectors:
            btn = None
            try:
                btn = self.wait_for_element(By.XPATH, selector, timeout=5)
                if btn:
                    btn.click()
                    print("Submitted registration form")
                    time.sleep(3)
                    return True
            except:
                continue
        print("Submit button not found")
        return False

    def process_registration(self, name, email):
        print(f"\nStarting registration for: {name} ({email})")
        self.driver.get(self.website_url)
        time.sleep(3)
        if not self.click_initial_register_button():
            print(f"Failed to click Register for {name}")
            return False
        self.fill_registration_form(name, email)
        if not self.submit_registration_form():
            print(f"Failed to submit form for {name}")
            return False
        print(f"Successfully registered: {name}")
        return True

    def refresh_page(self):
        self.driver.refresh()
        time.sleep(3)

    def run_automation(self, num_iterations=None, interval_seconds=5):
        if not self.load_excel_data():
            return False
        self.setup_driver()
        try:
            iterations = num_iterations if num_iterations else len(self.data)
            for i in range(iterations):
                row = self.data.iloc[i % len(self.data)]
                name = str(row.get("Name") or row.get("name")).strip()
                email = str(row.get("Email") or row.get("email")).strip()
                if not name or not email:
                    print(f"Skipping - missing name or email")
                    continue
                self.process_registration(name, email)
                if i < iterations - 1:
                    print(f"Waiting {interval_seconds} seconds before next registration...")
                    time.sleep(interval_seconds)
                    self.refresh_page()
        except Exception as e:
            print(f"Automation error: {e}")
        finally:
            self.driver.quit()
            print("Browser closed.")

if __name__ == "__main__":
    excel_file_path = "registration_data.xlsx"  # Path to your Excel file
    website_url = "https://luma.com/lagazg6k"   # Replace with your website URL
    num_iterations = 50                         # Set how many times to run
    interval_seconds = 1                        # Interval between registrations
    headless_mode = False                       # True for headless, False for GUI

    bot = WebsiteRegistrationBot(
        excel_file_path=excel_file_path,
        website_url=website_url,
        headless=headless_mode
    )
    bot.run_automation(num_iterations=num_iterations, interval_seconds=interval_seconds)
