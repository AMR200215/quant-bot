"""
research/tg_auth.py — one-time operator reauth tool for the Telethon session.

Usage:
    python -m research.tg_auth              # QR login (default)
    python -m research.tg_auth --method qr
    python -m research.tg_auth --method phone

Uses the same session path as the production monitor:
    memecoin/data/tg_session

On success: prints "Session authorised as <first_name>. Restart quantbot."

NEVER called by the daemon. Operator-only tool.
"""

import argparse
import asyncio
import os
import sys


# Session file shared with memecoin/telegram_monitor.py
_SESSION_FILE = os.path.join(
    os.path.dirname(__file__), "..", "memecoin", "data", "tg_session"
)


def _get_credentials() -> tuple[int, str]:
    """Read TELEGRAM_API_ID and TELEGRAM_API_HASH from env."""
    api_id_raw = os.environ.get("TELEGRAM_API_ID", "")
    api_hash   = os.environ.get("TELEGRAM_API_HASH", "")
    if not api_id_raw or not api_hash:
        print(
            "ERROR: TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in environment.\n"
            "Source your .env file first:  set -a && source .env && set +a",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        api_id = int(api_id_raw)
    except ValueError:
        print(f"ERROR: TELEGRAM_API_ID must be an integer, got: {api_id_raw!r}", file=sys.stderr)
        sys.exit(1)
    return api_id, api_hash


async def _qr_login(api_id: int, api_hash: str):
    """QR-code login flow. Prints QR to terminal, waits for scan."""
    try:
        from telethon import TelegramClient
        from telethon.errors import SessionPasswordNeededError
    except ImportError:
        print("ERROR: telethon not installed. Run: pip install telethon qrcode", file=sys.stderr)
        sys.exit(1)

    client = TelegramClient(_SESSION_FILE, api_id, api_hash)
    await client.connect()

    if await client.is_user_authorized():
        me = await client.get_me()
        print(f"Session already authorised as {me.first_name}. No action needed.")
        await client.disconnect()
        return

    print("Starting QR login. Open Telegram → Settings → Devices → Link Desktop Device.")
    print("Scan the QR code below with your phone:\n")

    # QR login loop: Telegram issues tokens that expire; refresh until scanned
    try:
        while True:
            try:
                qr_login = await client.qr_login()
            except Exception as e:
                print(f"ERROR starting QR login: {e}", file=sys.stderr)
                await client.disconnect()
                sys.exit(1)

            # Print QR code to terminal (text art)
            _print_qr(qr_login.url)

            try:
                # wait() blocks until scanned or token expires
                await qr_login.wait(30)
                break   # scanned successfully
            except asyncio.TimeoutError:
                # Token expired — loop prints a fresh one
                print("\n[QR expired — refreshing...]\n")
                continue
            except Exception as e:
                err_str = str(e).lower()
                if "session_password_needed" in err_str or "2fa" in err_str:
                    # QR scan succeeded but account has 2FA
                    password = _prompt_2fa()
                    await client.sign_in(password=password)
                    break
                raise

    except KeyboardInterrupt:
        print("\nAborted by operator.")
        await client.disconnect()
        sys.exit(1)

    me = await client.get_me()
    print(f"\nSession authorised as {me.first_name}. Restart quantbot.")
    await client.disconnect()


async def _phone_login(api_id: int, api_hash: str):
    """Phone-number + SMS/app code login flow."""
    try:
        from telethon import TelegramClient
        from telethon.errors import SessionPasswordNeededError
    except ImportError:
        print("ERROR: telethon not installed. Run: pip install telethon", file=sys.stderr)
        sys.exit(1)

    client = TelegramClient(_SESSION_FILE, api_id, api_hash)
    await client.connect()

    if await client.is_user_authorized():
        me = await client.get_me()
        print(f"Session already authorised as {me.first_name}. No action needed.")
        await client.disconnect()
        return

    phone = input("Enter phone number (international format, e.g. +1234567890): ").strip()
    if not phone:
        print("ERROR: phone number required.", file=sys.stderr)
        await client.disconnect()
        sys.exit(1)

    await client.send_code_request(phone)
    code = input("Enter the code sent to your Telegram app (do not share): ").strip()

    try:
        await client.sign_in(phone, code)
    except Exception as e:
        err_str = str(e).lower()
        if "session_password_needed" in err_str or "2fa" in err_str or "password" in err_str:
            password = _prompt_2fa()
            await client.sign_in(password=password)
        else:
            print(f"ERROR during sign-in: {e}", file=sys.stderr)
            await client.disconnect()
            sys.exit(1)

    me = await client.get_me()
    print(f"\nSession authorised as {me.first_name}. Restart quantbot.")
    await client.disconnect()


def _prompt_2fa() -> str:
    """Prompt for 2FA password without echoing to terminal."""
    import getpass
    return getpass.getpass("2FA password (hidden): ")


def _print_qr(url: str):
    """Print QR code to terminal as text art."""
    try:
        import qrcode
        qr = qrcode.QRCode()
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except ImportError:
        # Fallback: just print the URL — operator can use a QR generator
        print(f"QR URL (paste into https://qr.io to scan): {url}")


def main():
    parser = argparse.ArgumentParser(
        description="Reauthorise the Telethon session for the Telegram monitor.",
        epilog="Run this on the server when the monitor emits TELEGRAM_AUTH_REQUIRED.",
    )
    parser.add_argument(
        "--method", choices=["qr", "phone"], default="qr",
        help="Auth method: qr (default) or phone+code",
    )
    args = parser.parse_args()

    api_id, api_hash = _get_credentials()

    # Ensure session directory exists
    session_dir = os.path.dirname(os.path.abspath(_SESSION_FILE))
    os.makedirs(session_dir, exist_ok=True)

    # Check .gitignore protects the session file
    _warn_if_session_unprotected()

    if args.method == "qr":
        asyncio.run(_qr_login(api_id, api_hash))
    else:
        asyncio.run(_phone_login(api_id, api_hash))


def _warn_if_session_unprotected():
    """Warn operator if tg_session is not in .gitignore."""
    gitignore_paths = [
        os.path.join(os.path.dirname(__file__), "..", ".gitignore"),
        os.path.join(os.path.dirname(__file__), "..", "memecoin", ".gitignore"),
    ]
    protected = False
    for p in gitignore_paths:
        p = os.path.abspath(p)
        if os.path.exists(p):
            content = open(p).read()
            if "tg_session" in content:
                protected = True
                break
    if not protected:
        print(
            "WARNING: tg_session does not appear in .gitignore. "
            "Never commit the session file — it grants full Telegram account access.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
