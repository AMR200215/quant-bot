"""Signal generation helpers based on expected value."""


def expected_value_yes(
    model_p: float, yes_price: float, fee_buffer: float = 0.01
) -> float:
    """Compute the expected value of buying YES."""
    return model_p - yes_price - fee_buffer


def expected_value_no(
    model_p: float, no_price: float, fee_buffer: float = 0.01
) -> float:
    """Compute the expected value of buying NO."""
    return (1 - model_p) - no_price - fee_buffer


def choose_side(
    model_p: float, yes_price: float, no_price: float, fee_buffer: float = 0.01
) -> dict:
    """Choose the best side based on simple expected value."""
    ev_yes = expected_value_yes(model_p, yes_price, fee_buffer)
    ev_no = expected_value_no(model_p, no_price, fee_buffer)

    if ev_yes <= 0 and ev_no <= 0:
        return {"side": "hold", "ev_yes": ev_yes, "ev_no": ev_no}
    if ev_yes > ev_no:
        return {"side": "buy_yes", "ev_yes": ev_yes, "ev_no": ev_no}
    return {"side": "buy_no", "ev_yes": ev_yes, "ev_no": ev_no}


if __name__ == "__main__":
    model_p = 0.67
    yes_price = 0.55
    no_price = 0.45
    result = choose_side(model_p, yes_price, no_price)
    print(result)
