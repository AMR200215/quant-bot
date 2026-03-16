"""Alert helpers for the quant bot."""

import requests


def send_telegram_message(bot_token: str, chat_id: str, text: str) -> bool:
    """Send a simple Telegram message if credentials are configured."""
    if not bot_token or not chat_id:
        print("Telegram credentials missing. Skipping alert.")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    response = requests.post(url, json=payload, timeout=20)
    return response.status_code == 200


if __name__ == "__main__":
    print("Alerts module ready")
