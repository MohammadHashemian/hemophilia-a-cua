from dataclasses import dataclass
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import csv
import asyncio


IRC_URL = "https://irc.fda.gov.ir/nfi/"


@dataclass
class IRCData:
    persian_name: str
    english_name: str
    brand: str
    owner: str
    price: str
    dosage_form: str
    code: str
    generic_code: str


async def fetch_irc_factors(**kwargs):
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(**kwargs)
        page = await browser.new_page()
        await page.goto(IRC_URL)
        await page.locator("#Term").fill("Factor VIII")
        await page.get_by_role("button", name="جستجو").click()
        pag_array = [str(i) for i in range(1, 12)]
        results: list[IRCData] = []

        for pag in pag_array:
            print(f"Scraping page: [{pag}]")
            await page.get_by_role(
                "link", name=pag, exact=True
            ).scroll_into_view_if_needed()
            await page.get_by_role("link", name=pag, exact=True).click()

            cells = await page.locator(".searchRow").all()
            for cell in cells:
                soup = BeautifulSoup(await cell.inner_html(), "html.parser")

                try:
                    persian_name = soup.select_one("span.titleSearch-Link-RtlAlter a").text.strip()  # type: ignore
                    english_name = soup.select_one("span.titleSearch-Link-ltrAlter a").text.strip()  # type: ignore
                except:
                    persian_name = "N/A"
                    english_name = "N/A"

                brand = owner = price = dosage_form = code = generic_code = "N/A"

                rows = soup.select(".row")
                for row in rows:
                    for div in row.select("div"):
                        label = div.select_one("label")
                        if not label:
                            continue
                        label_text = label.text.strip()
                        value_span = div.select_one("span, bdo")
                        if not value_span:
                            continue
                        value = value_span.text.strip()

                        if "صاحب برند" in label_text:
                            brand = value
                        elif "صاحب پروانه" in label_text:
                            owner = value
                        elif "قیمت هر بسته" in label_text:
                            price = value
                        elif "بسته بندی" in label_text:
                            dosage_form = value
                        elif "کد فرآورده" in label_text:
                            code = value
                        elif "کد ژنریک" in label_text:
                            generic_code = value

                data = IRCData(
                    persian_name=persian_name,
                    english_name=english_name,
                    brand=brand,
                    owner=owner,
                    price=price,
                    dosage_form=dosage_form,
                    code=code,
                    generic_code=generic_code,
                )
                results.append(data)

    # Write to CSV
    with open(
        "hemophilia/data/raw/irc_fda.csv", "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(IRCData.__annotations__.keys())  # Header
        for item in results:
            writer.writerow(
                [
                    item.persian_name,
                    item.english_name,
                    item.brand,
                    item.owner,
                    item.price,
                    item.dosage_form,
                    item.code,
                    item.generic_code,
                ]
            )


if __name__ == "__main__":
    asyncio.run(fetch_irc_factors(headless=False, devtools=True))
