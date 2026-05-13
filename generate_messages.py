import os
import json
from datetime import datetime, timezone
from google import genai  # type: ignore[import-not-found]

GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-3.1-flash-lite-preview')

client = genai.Client(api_key=GEMINI_API_KEY)

SEASON = {
    (3, 4, 5): ('spring', '春・新生活・桜', 'spring, new beginnings'),
    (6, 7, 8): ('summer', '夏・暑さ・エネルギー', 'summer energy and heat'),
    (9, 10, 11): ('autumn', '秋・実り・涼しさ', 'autumn harvest and cool breeze'),
    (12, 1, 2): ('winter', '冬・温かさ・年末年始', 'winter warmth and year end'),
}

MONTHLY_THEME = {
    1:  ('新年・お正月気分',            'New Year energy'),
    2:  ('節分・冬の終わり',            'end of winter, fresh start'),
    3:  ('春の始まり・卒業・旅立ち',    'spring beginnings, farewells'),
    4:  ('桜・新生活スタート',          'cherry blossoms, new chapter'),
    5:  ('ゴールデンウィーク・初夏',    'golden week, early summer'),
    6:  ('梅雨・夏への準備',            'rainy season, preparing for summer'),
    7:  ('七夕・夏本番',                'midsummer, Tanabata wishes'),
    8:  ('お盆・夏休み・夏の終わり',    'Obon, summer holidays winding down'),
    9:  ('秋の始まり・実りの季節',      'early autumn, harvest season'),
    10: ('秋深まる・ハロウィン',        'deep autumn, Halloween'),
    11: ('紅葉・晩秋',                  'autumn leaves, late autumn'),
    12: ('クリスマス・年末・大掃除',    'Christmas, year-end reflection'),
}

def _get_season(month: int) -> tuple[str, str, str]:
    for months, info in SEASON.items():
        if month in months:
            return info
    return SEASON[(12, 1, 2)]

def _generate(prompt: str) -> list[str]:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={'response_mime_type': 'application/json'},
    )
    result = json.loads(response.text)
    return result if isinstance(result, list) else list(result.values())[0]

def generate_ja(theme: str) -> list[str]:
    return _generate(f"""
休みまでのカウントダウンアプリのウィジェットに表示する励ましメッセージを10件生成してください。
テーマ：{theme}
条件：日本語・10〜20文字・一言レベルの短さ・温かく汎用的（特定個人・職業に限定しない）・JSON文字列配列のみ出力
例：「あと少し。今日も丁寧に。」「ゆっくりでいい。えらい！」
""")

def generate_en(theme: str) -> list[str]:
    return _generate(f"""
Generate 10 encouraging messages for a home-screen widget counting down to the next day off.
Theme: {theme}
Rules: English, 8-35 chars each, one-liner style, warm and universal tone, output JSON string array only.
Examples: "Almost there. Keep going.", "One step at a time.", "You've got this!"
""")

def main():
    now = datetime.now(timezone.utc)
    year, month = now.year, now.month
    _, season_ja, season_en = _get_season(month)
    monthly_ja, monthly_en = MONTHLY_THEME[month]

    data = {
        'year': year,
        'month': month,
        'generated_at': now.isoformat(),
        'ja': {
            'season':          generate_ja(season_ja),
            'monthly':         generate_ja(monthly_ja),
            'holiday_working': generate_ja('休日出勤・休みでも働く人へのねぎらい'),
        },
        'en': {
            'season':          generate_en(season_en),
            'monthly':         generate_en(monthly_en),
            'holiday_working': generate_en('working on holidays, dedication deserves recognition'),
        },
    }

    path = f'messages/{year}_{month:02d}.json'
    os.makedirs('messages', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f'Generated: {path}')

if __name__ == '__main__':
    main()
