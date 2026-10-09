"""Frame scheduling and response thresholds for streaming evaluation."""

def threshold_state(fixed=None):
    """Use the evaluation policy, or an explicit fixed threshold for a deployment."""
    if fixed is not None and not 0 <= fixed <= 1:
        raise ValueError('Response threshold must be between 0 and 1.')
    state = {'silent': 0, 'consecutive': 0, 'threshold': 0.5 if fixed is None else fixed}
    if fixed is not None:
        state['fixed'] = fixed
    return state


def update_threshold(state, responded):
    if 'fixed' in state:
        return state['fixed']
    if responded:
        state['silent'] = 0
        state['consecutive'] += 1
        if state['consecutive'] % 10 == 0:
            state['threshold'] += 0.1
    else:
        state['consecutive'] = 0
        state['silent'] += 1
        if state['silent'] >= 30:
            state['threshold'] = 0.5
    return state['threshold']


def process_frame(engine, timestamp, feed_frame, threshold_state):
    """Consume one frame, respond once, then apply the evaluation threshold policy."""
    if 'fixed' in threshold_state:
        engine.frame_token_interval_threshold = threshold_state['fixed']
    feed_frame(timestamp)
    query, response = engine(update=True)
    engine.frame_token_interval_threshold = update_threshold(threshold_state, bool(response))
    return query, response
