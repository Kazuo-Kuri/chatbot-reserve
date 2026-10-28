"""外装素材×製品種×単色印刷色の登録マトリクスに基づく確定照合。"""
import json
from keyword_filter import extract_keywords


class ProductFilmMatcher:
    def __init__(self, json_path="data/product_film_color_matrix.json"):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def _products(self, name):
        if not name:
            return list(self.data)
        return [p for p in self.data if p == name or name in p]

    def _films(self, product, name):
        if not name:
            return list(self.data[product])
        return [f for f in self.data[product] if f == name or name in f]

    def get_films_for_product(self, product_name):
        products = self._products(product_name)
        if not products:
            return {"matched": False, "type": "product_to_films", "message": "該当する製品種が見つかりませんでした。"}
        p = products[0]
        return {"matched": True, "type": "product_to_films", "product": p, "films": self._films(p, None)}

    def get_colors_for_film_in_product(self, product_name, film_name):
        for p in self._products(product_name):
            for f in self._films(p, film_name):
                return {"matched": True, "type": "product_film_to_colors", "product": p,
                        "film": f, "colors": self.data[p][f]}
        return {"matched": False, "type": "product_film_to_colors", "message": "該当する製品と外装素材の組み合わせが見つかりませんでした。"}

    def get_products_for_film(self, film_name):
        products = [p for p in self.data if self._films(p, film_name)]
        return {"matched": bool(products), "type": "film_to_products", "film": film_name, "products": products}

    def get_films_for_color(self, color_names):
        films = list(dict.fromkeys(f for p, ff in self.data.items() for f, cc in ff.items()
                                   if any(c in cc for c in color_names)))
        return {"matched": bool(films), "type": "color_to_films", "color": ", ".join(color_names), "films": films}

    def get_products_for_color(self, color_names):
        products = [p for p, ff in self.data.items() if any(any(c in cc for c in color_names) for cc in ff.values())]
        return {"matched": bool(products), "type": "color_to_products", "color": ", ".join(color_names), "products": products}

    def get_film_colors_for_color(self, color_names):
        res = self.get_films_for_color(color_names)
        return {"matched": res["matched"], "type": "color_to_film_colors", "color": res["color"], "film_colors": res["films"]}

    def match(self, user_input, history=None):
        """登録済み組み合わせをすべて列挙。未指定の軸は全候補を検索。"""
        try:
            words = extract_keywords(user_input)
            products, films, colors = (words[k] for k in ("product", "film", "color"))
            text = str(user_input or "")
            # 製品名だけ含む納期・ロット等の質問をマトリクス回答で妨げない。
            material_intent = any(w in text for w in ("印刷", "色", "フィルム", "包材", "外装", "素材", "ゴールド", "シルバー", "クラフト", "ハイバリア", "使える", "対応可能"))
            if not (films or colors or (products and material_intent)):
                return {"matched": False, "type": "no_match", "message": "製品仕様照合の対象外です。"}
            if not (products or films or colors):
                return {"matched": False, "type": "no_match", "message": "製品・外装素材・印刷色を特定できませんでした。"}
            rows = []
            for p in self.data:
                if products and p not in self._matching_product_names(products):
                    continue
                for f, available in self.data[p].items():
                    if films and not any(name in f for name in films):
                        continue
                    matching_colors = [c for c in available if not colors or c in colors]
                    if matching_colors:
                        rows.append({"product": p, "film": f, "colors": matching_colors})
            return {"matched": bool(rows), "type": "matrix_combinations", "rows": rows,
                    "requested": words, "message": "登録された対応表には、指定条件に一致する組み合わせがありません。"}
        except Exception as exc:
            print("ProductFilmMatcher error:", exc)
            return {"matched": False, "type": "error", "message": "製品仕様照合中にエラーが発生しました。"}

    def _matching_product_names(self, names):
        return {p for name in names for p in self._products(name)}

    def format_match_info(self, info, fallback=False):
        if not isinstance(info, dict):
            return "" if not fallback else "【製品フィルム・カラー情報】確認できませんでした。"
        if info.get("type") == "matrix_combinations":
            if not info.get("matched"):
                return "【製品フィルム・カラー情報】" + info.get("message", "一致する組み合わせがありません。") if fallback else ""
            lines = ["【製品フィルム・カラー情報：登録対応表に基づく確定情報】"]
            grouped = {}
            for row in info["rows"]:
                grouped.setdefault(row["product"], []).append(row)
            for product, rows in grouped.items():
                lines.append(f"■ {product}")
                for row in rows:
                    lines.append(f"- {row['film']}：{', '.join(row['colors'])}")
            lines.append("※上記は登録済みのアフタープリント単色印刷の対応表に基づきます。")
            return "\n".join(lines)
        if not info.get("matched"):
            return ("【製品フィルム・カラー情報】" + info.get("message", "該当情報がありません。")) if fallback else ""
        kind = info.get("type")
        if kind == "product_to_films":
            return f"【製品フィルム・カラー情報】{info['product']}：{', '.join(info['films'])}"
        if kind == "product_film_to_colors":
            return f"【製品フィルム・カラー情報】{info['product']} × {info['film']}：{', '.join(info['colors'])}"
        if kind == "film_to_products":
            return f"【製品フィルム・カラー情報】{info['film']}：{', '.join(info['products'])}"
        if kind == "color_to_products":
            return f"【製品フィルム・カラー情報】{info['color']}：{', '.join(info['products'])}"
        if kind in ("color_to_films", "color_to_film_colors"):
            return f"【製品フィルム・カラー情報】{info['color']}：{', '.join(info.get('films', info.get('film_colors', [])))}"
        return ""
