from fallprecursor.training.callbacks import EarlyStopping


def test_does_not_stop_while_improving():
    stopper = EarlyStopping(patience=2)
    assert stopper.should_stop(1.0) is False
    assert stopper.should_stop(0.5) is False
    assert stopper.should_stop(0.4) is False


def test_stops_after_patience_exceeded():
    stopper = EarlyStopping(patience=2)
    stopper.should_stop(1.0)
    assert stopper.should_stop(1.1) is False
    assert stopper.should_stop(1.2) is False
    assert stopper.should_stop(1.3) is True


def test_improvement_resets_patience_counter():
    stopper = EarlyStopping(patience=1)
    stopper.should_stop(1.0)
    assert stopper.should_stop(1.1) is False
    assert stopper.should_stop(0.5) is False
    assert stopper.should_stop(0.6) is False
    assert stopper.should_stop(0.7) is True