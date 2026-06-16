import numpy as np

from backoff_control import backoff_control, priority_acb_backoff


def test_zero_load_allows_all_states():
    backoff = priority_acb_backoff(np.zeros(5), total_preambles=54)
    assert np.allclose(backoff, np.zeros(5))


def test_no_preambles_blocks_all_states():
    backoff = priority_acb_backoff(np.ones(5), total_preambles=0)
    assert np.allclose(backoff, np.ones(5))


def test_uniform_load_prioritizes_urgent_states():
    backoff = priority_acb_backoff(np.ones(5) * 100.0, total_preambles=54)
    assert np.all(backoff >= 0.0)
    assert np.all(backoff <= 1.0)
    assert np.all(np.diff(backoff) > 0.0)


def test_pacb_outputs_valid_state_vector():
    D = 5
    backoff, pi = backoff_control(
        N_tilde=1000.0,
        last_p_b=np.zeros(D),
        rho=0.1,
        D=D,
        p_d=np.ones(D) / D,
        p_s=0.9,
        K=2,
        Z=54,
        backoff_mode=3,
        Lambda=0.0,
    )

    assert backoff.shape == (D,)
    assert pi.shape == (D,)
    assert np.all(np.isfinite(backoff))
    assert np.all(np.isfinite(pi))
    assert np.all(backoff >= 0.0)
    assert np.all(backoff <= 1.0)
    assert backoff[0] < backoff[-1]


if __name__ == "__main__":
    test_zero_load_allows_all_states()
    test_no_preambles_blocks_all_states()
    test_uniform_load_prioritizes_urgent_states()
    test_pacb_outputs_valid_state_vector()
    print("priority_acb_test passed")
