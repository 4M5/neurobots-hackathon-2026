const test = require('node:test');
const assert = require('node:assert/strict');
const { createExamLifecycle, createQuestionNavigation, calculateWpm, shouldRecommendManualReview, calculateBehaviorFeatures } = require('../exam-lifecycle.js');

function fakeClock(onTimeout = () => {}) {
    let tick;
    const cleared = new Set();
    const lifecycle = createExamLifecycle({
        setInterval(callback) { tick = callback; return 1; },
        clearInterval(id) { cleared.add(id); },
        onTimeout
    });
    return { lifecycle, advance() { tick(); }, cleared };
}

test('exam timer starts once and a manual submission stops it exactly once', () => {
    const clock = fakeClock();
    assert.equal(clock.lifecycle.status, 'setup');
    assert.equal(clock.lifecycle.start(3), true);
    assert.equal(clock.lifecycle.start(3), false);
    assert.equal(clock.lifecycle.status, 'active');
    clock.advance();
    assert.equal(clock.lifecycle.remaining, 2);
    assert.equal(clock.lifecycle.submit(), true);
    assert.equal(clock.lifecycle.submit(), false);
    assert.equal(clock.lifecycle.status, 'verification');
    assert.deepEqual([...clock.cleared], [1]);
    clock.advance();
    assert.equal(clock.lifecycle.remaining, 2);
});

test('timer expiry transitions once into verification and completion is terminal', () => {
    let timeouts = 0;
    const clock = fakeClock(() => {
        timeouts += 1;
        assert.equal(clock.lifecycle.submit(), true);
    });
    clock.lifecycle.start(2);
    clock.advance();
    assert.equal(clock.lifecycle.status, 'active');
    clock.advance();
    assert.equal(clock.lifecycle.remaining, 0);
    assert.equal(clock.lifecycle.status, 'verification');
    assert.equal(timeouts, 1);
    assert.equal(clock.lifecycle.complete(), true);
    assert.equal(clock.lifecycle.complete(), false);
    assert.equal(clock.lifecycle.status, 'completed');
});

test('warning threshold recommends manual review without forcing exam submission', () => {
    const clock = fakeClock();
    clock.lifecycle.start(10);
    assert.equal(shouldRecommendManualReview(4, 5), false);
    assert.equal(shouldRecommendManualReview(5, 5), true);
    assert.equal(shouldRecommendManualReview(6, 5), true);
    assert.equal(clock.lifecycle.status, 'active');
    assert.equal(clock.lifecycle.submit(), true);
});

test('question navigation moves through answers into review and supports return/jump', () => {
    const nav = createQuestionNavigation(3);
    assert.deepEqual(nav.next(), { index: 1, mode: 'answer' });
    assert.deepEqual(nav.next(), { index: 2, mode: 'answer' });
    assert.deepEqual(nav.next(), { index: 2, mode: 'review' });
    assert.deepEqual(nav.previous(), { index: 2, mode: 'answer' });
    assert.equal(nav.goTo(0), true);
    assert.deepEqual({ index: nav.index, mode: nav.mode }, { index: 0, mode: 'answer' });
    assert.equal(nav.goTo(3), false);
    assert.equal(nav.goTo(-1), false);
});

test('review can be opened directly and remains on the current answer when revisited', () => {
    const nav = createQuestionNavigation(2);
    nav.goTo(1);
    assert.deepEqual(nav.review(), { index: 1, mode: 'review' });
    assert.deepEqual(nav.previous(), { index: 1, mode: 'answer' });
});

test('typing speed uses the actual valid sample count and caps extreme values', () => {
    assert.equal(calculateWpm([]), 0);
    assert.equal(calculateWpm([400]), 30);
    assert.equal(calculateWpm([300, 300]), 40);
    assert.equal(calculateWpm([0, -1, 10000]), 0);
    assert.equal(calculateWpm([20]), 200);
});

test('behavior feature aggregation uses intervals in seconds and returns the model schema', () => {
    const features = calculateBehaviorFeatures({
        keyIntervalsMs: [200, 400, 0, 10000],
        backspaceCount: 2,
        keystrokeCount: 20,
        pasteCount: 1,
        tabSwitches: 2,
        focusLosses: 3,
        idleDurationsSeconds: [2, 8],
        editCount: 2,
        firstKeyDelaysSeconds: [4, 8],
        questionCount: 2,
        answerDurationSeconds: 125
    });
    assert.deepEqual(Object.keys(features), [
        'avg_key_interval', 'std_key_interval', 'typing_speed', 'backspace_rate',
        'paste_count', 'tab_switch', 'focus_loss', 'avg_idle', 'max_idle',
        'edit_count', 'time_to_first_key', 'answer_duration'
    ]);
    assert.equal(features.avg_key_interval, 0.3);
    assert.equal(features.std_key_interval, 0.1);
    assert.equal(features.typing_speed, 200);
    assert.equal(features.backspace_rate, 0.1);
    assert.equal(features.paste_count, 0.5);
    assert.equal(features.tab_switch, 1);
    assert.equal(features.focus_loss, 1.5);
    assert.equal(features.avg_idle, 5);
    assert.equal(features.max_idle, 8);
    assert.equal(features.time_to_first_key, 6);
    assert.equal(features.edit_count, 1);
    assert.equal(features.answer_duration, 62.5);
});
