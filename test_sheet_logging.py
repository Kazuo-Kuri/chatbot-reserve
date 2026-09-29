"""Offline tests for newest-first Google Sheets logging."""
import ast
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import unittest


FAQ_SHEET = "faq_suggestions_reserve"
FEEDBACK_SHEET = "feedback_log_reserve"
CHAT_SHEET = "chat_logs_reserve"


class FakeRequest:
    def __init__(self, action):
        self.action = action

    def execute(self):
        return self.action()


class FakeSheetService:
    def __init__(self, rows_by_title):
        self.rows_by_title = copy.deepcopy(rows_by_title)
        self.sheet_ids = {title: index + 1 for index, title in enumerate(rows_by_title)}
        self.titles_by_id = {sheet_id: title for title, sheet_id in self.sheet_ids.items()}
        self.fail_batch = False
        self.lock = threading.Lock()

    def get(self, spreadsheetId, fields):
        del spreadsheetId, fields
        return FakeRequest(lambda: {
            "sheets": [
                {"properties": {"sheetId": sheet_id, "title": title}}
                for title, sheet_id in self.sheet_ids.items()
            ]
        })

    def batchUpdate(self, spreadsheetId, body):
        del spreadsheetId

        def apply_batch():
            if self.fail_batch:
                raise RuntimeError("mock Sheets API error")

            insert_request, update_request = body["requests"]
            insert = insert_request["insertDimension"]
            update = update_request["updateCells"]
            dimension_range = insert["range"]
            sheet_id = dimension_range["sheetId"]

            if (
                dimension_range["dimension"] != "ROWS"
                or dimension_range["startIndex"] != 1
                or dimension_range["endIndex"] != 2
                or update["start"] != {"sheetId": sheet_id, "rowIndex": 1, "columnIndex": 0}
                or update["fields"] != "userEnteredValue"
            ):
                raise AssertionError("Unexpected batchUpdate request")

            values = []
            for cell in update["rows"][0]["values"]:
                entered = cell["userEnteredValue"]
                values.append(next(iter(entered.values())))

            with self.lock:
                title = self.titles_by_id[sheet_id]
                self.rows_by_title[title].insert(1, values)
            return {}

        return FakeRequest(apply_batch)


def load_insert_function(sheet_service):
    source = Path("app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "insert_log_row"
    )
    env = {
        "sheet_service": sheet_service,
        "sheet_id_cache": {},
        "sheet_write_lock": threading.Lock(),
        "SPREADSHEET_ID": "test-spreadsheet",
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "app.py", "exec"), env)
    return env["insert_log_row"]


class SheetLoggingTests(unittest.TestCase):
    def test_header_only_sheet_gets_new_second_row(self):
        service = FakeSheetService({FAQ_SHEET: [["日時", "質問", "状態", "件数"]]})
        insert_log_row = load_insert_function(service)

        insert_log_row(FAQ_SHEET, ["new", "質問", "未回答", 1])

        self.assertEqual(service.rows_by_title[FAQ_SHEET], [
            ["日時", "質問", "状態", "件数"],
            ["new", "質問", "未回答", 1],
        ])

    def test_existing_rows_are_shifted_without_overwrite(self):
        service = FakeSheetService({
            FEEDBACK_SHEET: [
                ["日時", "質問", "回答", "評価", "理由"],
                ["old", "旧質問", "旧回答", "useful", ""],
            ]
        })
        insert_log_row = load_insert_function(service)

        insert_log_row(FEEDBACK_SHEET, ["new", "新質問", "新回答", "not_useful", "理由"])

        self.assertEqual(service.rows_by_title[FEEDBACK_SHEET][1:], [
            ["new", "新質問", "新回答", "not_useful", "理由"],
            ["old", "旧質問", "旧回答", "useful", ""],
        ])

    def test_two_consecutive_writes_keep_newest_first(self):
        service = FakeSheetService({CHAT_SHEET: [["日時", "質問", "回答", "種別", "未回答"]]})
        insert_log_row = load_insert_function(service)

        insert_log_row(CHAT_SHEET, ["first", "質問1", "回答1", "faq", "false"])
        insert_log_row(CHAT_SHEET, ["second", "質問2", "回答2", "knowledge", "false"])

        self.assertEqual([row[0] for row in service.rows_by_title[CHAT_SHEET]], [
            "日時", "second", "first"
        ])

    def test_batch_error_leaves_existing_rows_unchanged(self):
        initial_rows = {
            FAQ_SHEET: [
                ["日時", "質問", "状態", "件数"],
                ["old", "旧質問", "未回答", 1],
            ]
        }
        service = FakeSheetService(initial_rows)
        service.fail_batch = True
        insert_log_row = load_insert_function(service)

        with self.assertRaises(RuntimeError):
            insert_log_row(FAQ_SHEET, ["new", "新質問", "未回答", 1])

        self.assertEqual(service.rows_by_title, initial_rows)

    def test_concurrent_writes_preserve_both_rows(self):
        service = FakeSheetService({CHAT_SHEET: [["日時", "質問", "回答", "種別", "未回答"]]})
        insert_log_row = load_insert_function(service)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    insert_log_row,
                    CHAT_SHEET,
                    [label, f"質問{label}", f"回答{label}", "faq", "false"],
                )
                for label in ("one", "two")
            ]
            for future in futures:
                future.result()

        rows = service.rows_by_title[CHAT_SHEET]
        self.assertEqual(len(rows), 3)
        self.assertEqual({rows[1][0], rows[2][0]}, {"one", "two"})


if __name__ == "__main__":
    unittest.main()
