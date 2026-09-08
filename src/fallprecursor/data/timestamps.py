def make_strictly_increasing(timestamps_ms: list[int]) -> list[int]:
    """Bump any non-increasing timestamp up to one more than its predecessor.

    MediaPipe's Tasks API requires strictly increasing timestamps in VIDEO
    mode; real-world capture timestamps can round to the same millisecond
    for two consecutive frames, which this corrects without reordering
    anything.
    """
    result = []
    previous = -1
    for timestamp in timestamps_ms:
        if timestamp <= previous:
            timestamp = previous + 1
        result.append(timestamp)
        previous = timestamp
    return result