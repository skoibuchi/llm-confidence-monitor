"""
Japanese dataset generation script.

Generates knowledge probe training data for Japanese models
(e.g. rinna/japanese-gpt2-medium).
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
from src.data.dataset_generator import DatasetGenerator
from src.data.dataset import split_dataset


# --- Japanese-specific templates and database ---

JAPANESE_HIGH_CONFIDENCE_TEMPLATES = [
    # 地理
    ("{ja_country}の首都はどこですか？", "{ja_capital}", "geography", "easy"),
    ("富士山の高さは何メートルですか？", "3776メートル", "geography", "easy"),
    ("日本最長の川は何ですか？", "信濃川", "geography", "medium"),
    ("本州・四国・九州・北海道は日本の何ですか？", "四大島", "geography", "easy"),
    ("{ja_mountain}はどの国にありますか？", "{ja_mountain_country}", "geography", "medium"),

    # 科学
    ("水の化学式は何ですか？", "H₂O", "science", "easy"),
    ("光の速度は秒速約何キロメートルですか？", "約30万キロメートル", "science", "medium"),
    ("{ja_element}の元素記号は何ですか？", "{ja_symbol}", "science", "medium"),
    ("地球は太陽の周りを回っていますか？", "はい", "science", "easy"),
    ("人間の体を構成する細胞の数はおよそ何個ですか？", "約37兆個", "science", "medium"),

    # 数学
    ("1 + 1 はいくつですか？", "2", "math", "easy"),
    ("円周率πは約いくつですか？", "約3.14159", "math", "easy"),
    ("{ja_num1} × {ja_num2} はいくつですか？", "{ja_result}", "math", "easy"),
    ("直角三角形の斜辺の二乗は他の二辺の二乗の和に等しい。これを何の定理といいますか？", "ピタゴラスの定理", "math", "medium"),

    # 歴史
    ("{ja_event}は西暦何年に起きましたか？", "{ja_year}年", "history", "medium"),
    ("{ja_person}は何をした人ですか？", "{ja_achievement}", "history", "medium"),
    ("日本の初代内閣総理大臣は誰ですか？", "伊藤博文", "history", "medium"),

    # 文化
    ("日本の国花は何ですか？", "桜（ソメイヨシノ）", "culture", "easy"),
    ("日本語のひらがなは何文字ありますか？", "46文字", "culture", "easy"),
]

JAPANESE_MEDIUM_CONFIDENCE_TEMPLATES = [
    # 技術
    ("Pythonで{ja_operation}する方法は？", "{ja_method}", "technology", "medium"),
    ("SQLで{ja_sql_op}するコマンドは何ですか？", "{ja_sql_cmd}", "technology", "medium"),

    # 文学
    ("{ja_author}の代表作は何ですか？", "{ja_work}", "literature", "medium"),
    ("{ja_work}の著者は誰ですか？", "{ja_author}", "literature", "medium"),

    # 音楽
    ("{ja_composer}の代表曲は何ですか？", "{ja_piece}", "music", "medium"),

    # スポーツ
    ("{ja_sport}は何人でプレイしますか？", "{ja_sport_num}人", "sports", "medium"),
    ("野球の1チームの選手数は何人ですか？", "9人", "sports", "easy"),

    # 食
    ("味噌汁の主な材料は何ですか？", "味噌・だし・具材", "food", "medium"),
]

JAPANESE_LOW_CONFIDENCE_TEMPLATES = [
    # 架空の情報
    ("惑星{ja_fake_planet}の重力加速度は何m/s²ですか？", "存在しない惑星", "fictional", "hard"),
    ("架空の元素{ja_fake_element}の原子番号は？", "存在しない元素", "fictional", "hard"),
    ("{ja_fake_place}の人口は何人ですか？", "存在しない場所", "fictional", "hard"),
    ("架空の国{ja_fake_country}の首都はどこですか？", "存在しない国", "fictional", "hard"),

    # 未来・不確定情報
    ("{ja_future_year}年のノーベル{ja_field}賞受賞者は誰ですか？", "未来の情報", "current_events", "hard"),
    ("{ja_future_year}年の日本の総理大臣は誰ですか？", "未来の情報", "current_events", "hard"),
    ("存在しないAI技術{ja_fake_tech}とは何ですか？", "架空の技術", "fictional", "hard"),
]

JAPANESE_DATA = {
    "ja_country": ["日本", "アメリカ", "フランス", "ドイツ", "中国", "イギリス", "イタリア", "オーストラリア"],
    "ja_capital": ["東京", "ワシントンD.C.", "パリ", "ベルリン", "北京", "ロンドン", "ローマ", "キャンベラ"],
    "ja_mountain": ["富士山", "エベレスト", "キリマンジャロ", "マッターホルン", "アコンカグア"],
    "ja_mountain_country": ["日本", "ネパール/中国", "タンザニア", "スイス/イタリア", "アルゼンチン/チリ"],
    "ja_element": ["水素", "酸素", "炭素", "窒素", "鉄", "金", "銀", "ナトリウム"],
    "ja_symbol": ["H", "O", "C", "N", "Fe", "Au", "Ag", "Na"],
    "ja_event": ["第二次世界大戦終結", "フランス革命", "アメリカ独立", "明治維新", "東日本大震災"],
    "ja_year": ["1945", "1789", "1776", "1868", "2011"],
    "ja_person": ["アインシュタイン", "ニュートン", "ダーウィン", "野口英世", "湯川秀樹"],
    "ja_achievement": [
        "相対性理論を発表した",
        "万有引力の法則を発見した",
        "進化論を提唱した",
        "黄熱病の研究をした細菌学者",
        "日本人初のノーベル賞受賞者（物理学）",
    ],
    "ja_operation": ["リストを逆順にソート", "辞書のキーを一覧取得", "文字列を区切り文字で分割"],
    "ja_method": ["sorted(lst, reverse=True)", "dict.keys()", "str.split(delimiter)"],
    "ja_sql_op": ["テーブルを結合", "重複行を除外して取得", "レコードを削除"],
    "ja_sql_cmd": ["JOIN", "SELECT DISTINCT", "DELETE"],
    "ja_author": ["夏目漱石", "芥川龍之介", "太宰治", "川端康成", "三島由紀夫"],
    "ja_work": ["吾輩は猫である", "羅生門", "人間失格", "雪国", "金閣寺"],
    "ja_composer": ["ベートーヴェン", "モーツァルト", "バッハ", "ショパン", "チャイコフスキー"],
    "ja_piece": ["第九交響曲", "トルコ行進曲", "マタイ受難曲", "夜想曲集", "白鳥の湖"],
    "ja_sport": ["サッカー", "バスケットボール", "バレーボール", "水球"],
    "ja_sport_num": ["11", "5", "6", "7"],
    "ja_fake_planet": ["X", "ゼータ", "オメガ", "ネクサス"],
    "ja_fake_element": ["Z", "ウルトラニウム", "ミスリル", "オリハルコン"],
    "ja_fake_place": ["アトランティス", "エルドラド", "シャングリラ", "ムー大陸"],
    "ja_fake_country": ["ジルコニア", "アスタリア", "ネバーランド"],
    "ja_future_year": ["2035", "2040", "2050", "2060"],
    "ja_field": ["物理学", "化学", "文学", "平和", "経済学"],
    "ja_fake_tech": ["QuantumBrain", "NeuralForge X", "HyperMind"],
}


def create_japanese_generator() -> DatasetGenerator:
    """Return a DatasetGenerator with Japanese templates injected."""
    gen = DatasetGenerator(seed=42)
    gen.high_confidence_templates = JAPANESE_HIGH_CONFIDENCE_TEMPLATES
    gen.medium_confidence_templates = JAPANESE_MEDIUM_CONFIDENCE_TEMPLATES
    gen.low_confidence_templates = JAPANESE_LOW_CONFIDENCE_TEMPLATES
    gen.data = JAPANESE_DATA

    # Add keys for numeric calculations
    gen.data["ja_num1"] = [str(i) for i in range(2, 10)]
    gen.data["ja_num2"] = [str(i) for i in range(2, 10)]

    # Wrap generate_sample to handle {ja_num1}/{ja_num2}/{ja_result} substitution
    original_generate = gen.generate_sample

    def generate_sample_ja(template, confidence):
        sample = original_generate(template, confidence)
        # Compute value if {ja_num1}/{ja_num2} remain
        import random as _r
        if "{ja_num1}" in sample["question"]:
            n1 = _r.randint(2, 9)
            n2 = _r.randint(2, 9)
            sample["question"] = sample["question"].replace("{ja_num1}", str(n1)).replace("{ja_num2}", str(n2))
            sample["answer"] = sample["answer"].replace("{ja_result}", str(n1 * n2))
        return sample

    gen.generate_sample = generate_sample_ja
    return gen


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Japanese knowledge probe dataset")
    parser.add_argument("--num_samples", type=int, default=2000,
                        help="Number of samples to generate (default: 2000)")
    parser.add_argument("--output_path", type=str,
                        default="data/japanese/raw/japanese_dataset.jsonl",
                        help="Output file path")
    parser.add_argument("--processed_dir", type=str,
                        default="data/japanese/processed",
                        help="Directory to save split data")
    parser.add_argument("--high_ratio", type=float, default=0.35)
    parser.add_argument("--medium_ratio", type=float, default=0.35)
    parser.add_argument("--low_ratio", type=float, default=0.30)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Japanese Knowledge Dataset Generation")
    print("=" * 60)
    print(f"  Num samples     : {args.num_samples}")
    print(f"  High conf ratio : {args.high_ratio}")
    print(f"  Mid conf ratio  : {args.medium_ratio}")
    print(f"  Low conf ratio  : {args.low_ratio}")
    print(f"  Output path     : {args.output_path}")

    generator = create_japanese_generator()
    import random
    random.seed(args.seed)

    samples = generator.generate_dataset(
        num_samples=args.num_samples,
        high_ratio=args.high_ratio,
        medium_ratio=args.medium_ratio,
        low_ratio=args.low_ratio,
    )

    # Statistics
    categories = {}
    conf_ranges = {"0.0-0.25": 0, "0.25-0.5": 0, "0.5-0.75": 0, "0.75-1.0": 0}
    for s in samples:
        categories[s["category"]] = categories.get(s["category"], 0) + 1
        c = s["confidence_label"]
        if c < 0.25:
            conf_ranges["0.0-0.25"] += 1
        elif c < 0.5:
            conf_ranges["0.25-0.5"] += 1
        elif c < 0.75:
            conf_ranges["0.5-0.75"] += 1
        else:
            conf_ranges["0.75-1.0"] += 1

    print(f"\nGeneration complete: {len(samples)} samples")
    print("\nCategory breakdown:")
    for cat, cnt in sorted(categories.items()):
        print(f"  {cat}: {cnt} ({cnt/len(samples)*100:.1f}%)")
    print("\nConfidence distribution:")
    for rng, cnt in conf_ranges.items():
        print(f"  {rng}: {cnt} ({cnt/len(samples)*100:.1f}%)")

    print("\nSample examples (first 5):")
    for i, s in enumerate(samples[:5]):
        print(f"  {i+1}. Q: {s['question']}")
        print(f"     A: {s['answer']}  (confidence={s['confidence_label']:.3f})")

    # Save
    output_path = Path(args.output_path)
    generator.save_dataset(samples, output_path)
    print(f"\nRAW data saved: {output_path}")

    # Split
    processed_dir = Path(args.processed_dir)
    splits = split_dataset(
        output_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        output_dir=processed_dir,
        seed=args.seed,
    )
    print(f"\nDataset split:")
    print(f"  train : {len(splits['train'])} samples -> {processed_dir}/train.jsonl")
    print(f"  val   : {len(splits['val'])} samples -> {processed_dir}/val.jsonl")
    print(f"  test  : {len(splits['test'])} samples -> {processed_dir}/test.jsonl")
    print(f"\n{'=' * 60}")
    print("Japanese dataset generation complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
