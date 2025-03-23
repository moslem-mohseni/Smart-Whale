import pickle
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# مسیر پروفایل کروم (مقداردهی کن با مسیر خودت)
CHROME_PROFILE_PATH = r"C:\Users\ASUS\AppData\Local\Google\Chrome\User Data\Profile 7"

# تنظیم WebDriver
chrome_options = Options()
chrome_options.add_argument(f"--user-data-dir={CHROME_PROFILE_PATH}")  # استفاده از پروفایل اصلی
chrome_options.add_argument("--profile-directory=Default")  # پروفایل مورد نظر

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# باز کردن صفحه‌ی لاگین توییتر (X)
driver.get("https://x.com/home")

# صبر می‌کنیم که کاربر لاگین کند
input("🔹 لطفاً وارد حساب کاربری شوید و سپس Enter را بزنید...")

# ذخیره کوکی‌ها بعد از ورود موفق
pickle.dump(driver.get_cookies(), open("cookies.pkl", "wb"))
print("✅ کوکی‌ها ذخیره شدند!")

driver.quit()
