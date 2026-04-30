"""
Phase 2 — Trade Execution (stub).

This module will handle actual on-chain trade placement:
  - Solana: Jupiter aggregator API → swap SOL → token
  - BSC: PancakeSwap router via web3.py

Not yet active. All methods return NotImplementedError.
Wire this in after paper-trading validates signals.
"""


class MemeExecutor:
    def __init__(self, chain: str):
        self.chain = chain

    def buy(self, token_address: str, amount_usd: float) -> dict:
        raise NotImplementedError("Execution not yet enabled — still in paper-trade mode.")

    def sell(self, token_address: str, amount_pct: float = 1.0) -> dict:
        raise NotImplementedError("Execution not yet enabled — still in paper-trade mode.")

    def get_balance(self) -> dict:
        raise NotImplementedError("Execution not yet enabled.")
