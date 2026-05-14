import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from google import genai  # type: ignore[import-not-found]
from google.genai import errors as genai_errors  # type: ignore[import-not-found]

GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-3.1-flash-lite-preview')

client = genai.Client(api_key=GEMINI_API_KEY)

def _load_style_examples() -> tuple[list[str], list[str]]:
    path = Path(__file__).parent / 'reference' / 'encouragement.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    ja_examples: list[str] = []
    en_examples: list[str] = []
    for msgs in data.get('ja', {}).values():
        ja_examples.extend(msgs)
    for msgs in data.get('en', {}).values():
        en_examples.extend(msgs)
    return ja_examples, en_examples

_STYLE_JA, _STYLE_EN = _load_style_examples()

SEASON = {
    (3, 4, 5): ('spring', '春・新生活・桜', 'spring, new beginnings'),
    (6, 7, 8): ('summer', '夏・暑さ・エネルギー', 'summer energy and heat'),
    (9, 10, 11): ('autumn', '秋・実り・涼しさ', 'autumn harvest and cool breeze'),
    (12, 1, 2): ('winter', '冬・温かさ・年末年始', 'winter warmth and year end'),
}

MONTHLY_THEME = {
    1:  ('冬・新年の決意',              'winter, new year mindset'),
    2:  ('冬の終わり・春への期待',      'end of winter, fresh start'),
    3:  ('春の始まり・年度末',          'spring beginnings'),
    4:  ('春・新生活スタート',           'spring, new chapter'),
    5:  ('初夏・新緑・さわやかな季節',  'early summer, fresh greenery'),
    6:  ('梅雨・夏への準備',            'rainy season, preparing for summer'),
    7:  ('夏本番・暑さの盛り',          'peak summer heat and energy'),
    8:  ('真夏・夏の終わり',           'late summer, winding down'),
    9:  ('秋の始まり・実りの季節',      'early autumn, harvest season'),
    10: ('秋深まる・ハロウィン',        'deep autumn, Halloween'),
    11: ('紅葉・晩秋',                  'autumn leaves, late autumn'),
    12: ('年末・大掃除・冬休み',        'year-end, winter break'),
}

def _get_season(month: int) -> tuple[str, str, str]:
    for months, info in SEASON.items():
        if month in months:
            return info
    return SEASON[(12, 1, 2)]

MAX_LEN_JA = 20
MAX_LEN_EN = 30
GENERATE_COUNT = 20
RESULT_COUNT = 10
MAX_RETRIES = 3
RETRY_INTERVAL = 30

def _call_api(prompt: str) -> dict[str, list[str]]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={'response_mime_type': 'application/json'},
            )
            return json.loads(response.text)  # type: ignore[no-any-return]
        except genai_errors.ServerError as e:
            if attempt == MAX_RETRIES:
                raise
            print(f'Attempt {attempt}/{MAX_RETRIES} failed: {e}. Retrying in {RETRY_INTERVAL}s...')
            time.sleep(RETRY_INTERVAL)
    raise RuntimeError("unreachable")

def _pick_shortest(messages: list[str], max_len: int, n: int = RESULT_COUNT) -> list[str]:
    filtered = [m for m in messages if len(m) <= max_len]
    if len(filtered) < n:
        filtered = sorted(messages, key=len)
    return filtered[:n]

def generate(theme_ja: str, theme_en: str) -> tuple[list[str], list[str]]:
    ja_examples = ''.join(f'「{m}」' for m in _STYLE_JA[:8])
    en_examples = ' '.join(f'"{m}"' for m in _STYLE_EN[:8])
    result = _call_api(f"""
スマートフォンのホーム画面ウィジェット（次の休みまでのカウントダウン表示）用の短い励ましメッセージを日本語・英語それぞれ{GENERATE_COUNT}件生成してください。

日本語テーマ：{theme_ja}
英語テーマ：{theme_en}

条件（日英共通）：
- 1文のみ・温かく汎用的（特定個人・職業に限定しない）
- 特定の連休・祝日・行事が「これから来る・待っている」という表現は絶対に使わないこと（例：「連休が待っています」「Golden Week is near」は不可）
- 月のどの日にも違和感なく使えるよう、時期に依存しない表現にすること

日本語スタイル参考（このような簡潔・直接的な表現を目指すこと）：{ja_examples}
英語スタイル参考（aim for this concise, direct tone）：{en_examples}

出力形式（JSONのみ）：{{"ja": ["...", ...], "en": ["...", ...]}}
""")
    return _pick_shortest(result.get('ja', []), MAX_LEN_JA), _pick_shortest(result.get('en', []), MAX_LEN_EN)

def main():
    now = datetime.now(timezone.utc)
    year, month = now.year, now.month
    _, season_ja, season_en = _get_season(month)
    monthly_ja, monthly_en = MONTHLY_THEME[month]

    season_ja_msgs,   season_en_msgs   = generate(season_ja,  season_en)
    monthly_ja_msgs,  monthly_en_msgs  = generate(monthly_ja, monthly_en)
    holiday_ja_msgs,  holiday_en_msgs  = generate('休日出勤・休みでも働く人へのねぎらい', 'working on holidays, dedication deserves recognition')

    data = {
        'year': year,
        'month': month,
        'generated_at': now.isoformat(),
        'ja': {
            'season':          season_ja_msgs,
            'monthly':         monthly_ja_msgs,
            'holiday_working': holiday_ja_msgs,
        },
        'en': {
            'season':          season_en_msgs,
            'monthly':         monthly_en_msgs,
            'holiday_working': holiday_en_msgs,
        },
    }

    path = f'messages/{year}_{month:02d}.json'
    os.makedirs('messages', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f'Generated: {path}')

if __name__ == '__main__':
    main()
