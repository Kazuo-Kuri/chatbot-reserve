import json
import os
import time

import gspread
from dotenv import load_dotenv
from oauth2client.service_account import ServiceAccountCredentials


TRANSIENT_HTTP_STATUSES = {429, 500, 502, 503, 504}
MAX_GOOGLE_API_ATTEMPTS = 3
INITIAL_RETRY_DELAY_SECONDS = 10


def get_http_status(error):
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(error, "code", None)

    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def fetch_knowledge_records(
    client,
    spreadsheet_id,
    knowledge_sheet,
    max_attempts=MAX_GOOGLE_API_ATTEMPTS,
    initial_delay=INITIAL_RETRY_DELAY_SECONDS,
    sleep=time.sleep,
):
    for attempt in range(1, max_attempts + 1):
        try:
            spreadsheet = client.open_by_key(spreadsheet_id)
            sheet = spreadsheet.worksheet(knowledge_sheet)
            return sheet.get_all_records()
        except gspread.exceptions.APIError as error:
            status = get_http_status(error)
            if status not in TRANSIENT_HTTP_STATUSES:
                print("Google Sheets API non-retryable error.")
                print(f"Attempt {attempt}/{max_attempts}")
                print(f"HTTP status: {status if status is not None else 'unknown'}")
                raise

            print("Google Sheets API temporary error.")
            print(f"Attempt {attempt}/{max_attempts}")
            print(f"HTTP status: {status}")

            if attempt == max_attempts:
                print(f"Google Sheets API failed after {max_attempts} attempts.")
                raise

            delay = initial_delay * (2 ** (attempt - 1))
            print(f"Retrying in {delay} seconds...")
            sleep(delay)

    raise RuntimeError("Google Sheets API retry loop ended unexpectedly.")


def main():
    if os.getenv("GITHUB_ACTIONS") != "true":
        load_dotenv()

    spreadsheet_id = os.getenv("SPREADSHEET_ID")
    if not spreadsheet_id or not spreadsheet_id.strip():
        raise ValueError("SPREADSHEET_ID is not set.")

    knowledge_sheet = os.getenv("KNOWLEDGE_SHEET") or "reserve_knowledge"
    if not knowledge_sheet.strip():
        knowledge_sheet = "reserve_knowledge"

    output_path = os.getenv("OUTPUT_PATH") or "data/reserve_knowledge.json"

    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        "credentials.json",
        scope,
    )
    client = gspread.authorize(credentials)

    records = fetch_knowledge_records(
        client,
        spreadsheet_id.strip(),
        knowledge_sheet,
    )
    knowledge = {row["title"]: [row["content"]] for row in records}

    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, ensure_ascii=False, indent=2)

    print(f"✅ {output_path} を保存しました。")


if __name__ == "__main__":
    main()
