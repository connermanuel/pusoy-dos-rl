from pusoy.train import play_round_async


def test_play_round_async(base_model_a2c):
    _ = play_round_async(
        [base_model_a2c] * 4,
        [0, 0, 0, 0],
    )
    pass
