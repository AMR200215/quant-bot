from app.alerts import send_telegram_message
from app.bayes import bayes_update
from app.data_client import fetch_mock_markets
from app.risk import fractional_kelly_size, impact_estimate
from app.signals import choose_side
from app.state import settings


def main():
    print("Quant bot started")
    print("Bankroll:", settings.bankroll)

    markets = fetch_mock_markets()

    for market in markets:
        likelihood_true = 0.65
        likelihood_false = 0.35
        posterior = bayes_update(
            prior=market.yes_price,
            likelihood_true=likelihood_true,
            likelihood_false=likelihood_false,
        )

        signal = choose_side(
            model_p=posterior,
            yes_price=market.yes_price,
            no_price=market.no_price,
            fee_buffer=0.01,
        )

        if signal["side"] == "hold":
            print(f"Skipping market: {market.question}")
            continue

        if signal["side"] == "buy_yes":
            size = fractional_kelly_size(
                bankroll=settings.bankroll,
                p=posterior,
                price=market.yes_price,
                fraction=settings.fractional_kelly,
            )
        else:
            size = fractional_kelly_size(
                bankroll=settings.bankroll,
                p=1 - posterior,
                price=market.no_price,
                fraction=settings.fractional_kelly,
            )

        impact = impact_estimate(size, market.liquidity_depth)

        message = (
            f"Market: {market.question}\n"
            f"Posterior: {posterior:.4f}\n"
            f"Side: {signal['side']}\n"
            f"EV Yes: {signal['ev_yes']:.4f}\n"
            f"EV No: {signal['ev_no']:.4f}\n"
            f"Size: {size:.2f}\n"
            f"Impact: {impact:.4f}"
        )

        print(f"Market: {market.question}")
        print(f"Posterior: {posterior:.4f}")
        print(f"Side: {signal['side']}")
        print(f"EV Yes: {signal['ev_yes']:.4f}")
        print(f"EV No: {signal['ev_no']:.4f}")
        print(f"Size: {size:.2f}")
        print(f"Impact: {impact:.4f}")
        print()

        send_telegram_message(
            settings.telegram_bot_token,
            settings.telegram_chat_id,
            message,
        )


if __name__ == "__main__":
    main()
