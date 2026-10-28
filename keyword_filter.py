"""製品・外装素材・アフタープリント印刷色のキーワード抽出。"""
import re
import unicodedata


def extract_keywords(text):
    text = unicodedata.normalize("NFKC", str(text or ""))
    # 長い製品名を先に判定し、VFR型とVFR増量タイプを混同しない。
    product_aliases = {
        "VFR増量タイプ": ["VFR増量タイプ", "VFR増量", "VFR型増量"],
        "X増量タイプ": ["X増量タイプ", "X増量", "X型増量"],
        "VFR型": ["VFR型", "VFR"],
        "X型": ["X型"],
        "ディップスタイル": ["ディップスタイル", "DipStyle", "Dip Style"],
        "個包装コーヒーバッグ": ["個包装コーヒーバッグ"],
        "水出しコーヒー": ["水出しコーヒー(個包装なし", "水出しコーヒー（個包装なし"],
    }
    film_aliases = {
        "白光沢フィルム": ["白光沢フィルム"],
        "白マットフィルム": ["白マットフィルム"],
        "黒光沢フィルム": ["黒光沢フィルム"],
        "黒マットフィルム": ["黒マットフィルム"],
        "青光沢フィルム": ["青光沢フィルム"],
        "青マットフィルム": ["青マットフィルム"],
        "赤フィルム": ["赤フィルム"],
        "緑フィルム": ["緑フィルム"],
        "サンドベージュフィルム": ["サンドベージュフィルム"],
        "クラフト包材": ["クラフト包材", "クラフト紙", "クラフト"],
        "紙リサイクルマーク付き包材": ["紙リサイクルマーク付き包材", "紙リサイクルマーク"],
        "ハイバリア特殊紙": ["ハイバリア特殊紙", "ハイバリア紙"],
    }
    colors = {
        "シルバー": ["シルバー", "シルバ", "銀色", "銀印刷"],
        "ゴールド": ["ゴールド", "金色", "金印刷"],
        "黒": ["黒色", "黒印刷", "ブラック"],
        "白": ["白色", "白印刷", "ホワイト"],
        "赤": ["赤色", "赤印刷"],
        "青": ["青色", "青印刷"],
        "茶": ["茶色", "茶印刷"],
    }
    result = {"product": [], "film": [], "color": []}
    for canonical, aliases in product_aliases.items():
        if any(a.lower() in text.lower() for a in aliases):
            result["product"].append(canonical)
    # 型の短い名称が増量タイプの内部に含まれている場合は除外。
    if "VFR増量タイプ" in result["product"] and "VFR型" in result["product"]:
        result["product"].remove("VFR型")
    if "X増量タイプ" in result["product"] and "X型" in result["product"]:
        result["product"].remove("X型")
    remaining = text
    for canonical, aliases in film_aliases.items():
        if any(a.lower() in remaining.lower() for a in aliases):
            result["film"].append(canonical)
            for alias in aliases:
                remaining = re.sub(re.escape(alias), " ", remaining, flags=re.IGNORECASE)
    # 印刷色の単独一文字は「黒印刷」「印刷色は黒」など印刷文脈でのみ検出。
    for canonical, aliases in colors.items():
        if any(a.lower() in remaining.lower() for a in aliases):
            result["color"].append(canonical)
            continue
        if canonical in "黒白赤青茶" and re.search(
            rf"(?:印刷(?:色)?(?:は|が|を|に|の|で)?\s*|色(?:は|が|を|に)?\s*){canonical}(?![光沢マットフィルム])|"
            rf"{canonical}\s*(?:で印刷|の印刷|インク)", remaining
        ):
            result["color"].append(canonical)
    print("抽出結果:", result)
    return result
