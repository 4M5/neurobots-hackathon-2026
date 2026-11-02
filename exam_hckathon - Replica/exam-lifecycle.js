(function (root, factory) {
    const api = factory();
    if (typeof module === 'object' && module.exports) module.exports = api;
    if (root) root.ExamLifecycle = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
    function createExamLifecycle(options = {}) {
        const schedule = options.setInterval || setInterval;
        const cancel = options.clearInterval || clearInterval;
        const onTimeout = options.onTimeout || function () {};
        const onTick = options.onTick || function () {};
        let status = 'setup';
        let remaining = 0;
        let timer = null;

        function stopTimer() {
            if (timer !== null) cancel(timer);
            timer = null;
        }

        function submit() {
            if (status !== 'active') return false;
            status = 'verification';
            stopTimer();
            return true;
        }

        return {
            start(seconds) {
                if (status !== 'setup' || !Number.isFinite(seconds) || seconds <= 0) return false;
                status = 'active';
                remaining = Math.floor(seconds);
                timer = schedule(() => {
                    if (status !== 'active') return;
                    remaining = Math.max(0, remaining - 1);
                    onTick(remaining);
                    if (remaining === 0) {
                        stopTimer();
                        onTimeout();
                        if (status === 'active') submit();
                    }
                }, 1000);
                return true;
            },
            submit,
            complete() {
                if (status !== 'verification') return false;
                status = 'completed';
                stopTimer();
                return true;
            },
            get status() { return status; },
            get remaining() { return remaining; },
            dispose() { stopTimer(); }
        };
    }

    function createQuestionNavigation(total) {
        let index = 0;
        let mode = 'answer';
        if (!Number.isInteger(total) || total < 1) throw new TypeError('At least one question is required');
        return {
            next() {
                if (mode === 'review') return { index, mode };
                if (index < total - 1) index += 1;
                else mode = 'review';
                return { index, mode };
            },
            previous() {
                if (mode === 'review') mode = 'answer';
                else if (index > 0) index -= 1;
                return { index, mode };
            },
            goTo(questionIndex) {
                if (!Number.isInteger(questionIndex) || questionIndex < 0 || questionIndex >= total) return false;
                index = questionIndex;
                mode = 'answer';
                return true;
            },
            review() { mode = 'review'; return { index, mode }; },
            get index() { return index; },
            get mode() { return mode; },
            get total() { return total; }
        };
    }

    function calculateWpm(intervals) {
        const recent = intervals.filter(value => Number.isFinite(value) && value > 0 && value < 10000).slice(-20);
        if (!recent.length) return 0;
        const averageInterval = recent.reduce((sum, value) => sum + value, 0) / recent.length;
        return Math.min(200, Math.round(60000 / averageInterval / 5));
    }

    function shouldRecommendManualReview(warningCount, threshold) {
        return Number.isFinite(warningCount) && Number.isFinite(threshold) && threshold > 0 && warningCount >= threshold;
    }

    function calculateBehaviorFeatures(metrics) {
        const intervals = (metrics.keyIntervalsMs || []).filter(value => Number.isFinite(value) && value > 0 && value < 10000);
        const averageIntervalMs = intervals.length ? intervals.reduce((sum, value) => sum + value, 0) / intervals.length : 0;
        const averageIntervalSeconds = averageIntervalMs / 1000;
        const varianceSeconds = intervals.length
            ? intervals.reduce((sum, value) => sum + ((value / 1000 - averageIntervalSeconds) ** 2), 0) / intervals.length
            : 0;
        const idle = (metrics.idleDurationsSeconds || []).filter(value => Number.isFinite(value) && value >= 0);
        const delays = (metrics.firstKeyDelaysSeconds || []).filter(value => Number.isFinite(value) && value >= 0);
        // Training examples represent one answer, so exam-wide counts/duration
        // are expressed as a per-question mean before they reach the model.
        const questionCount = Number.isFinite(metrics.questionCount) && metrics.questionCount > 0 ? metrics.questionCount : 1;
        const backspaces = Number(metrics.backspaceCount) || 0;
        const keystrokes = Number(metrics.keystrokeCount) || 0;
        return {
            avg_key_interval: averageIntervalSeconds,
            std_key_interval: Math.sqrt(varianceSeconds),
            typing_speed: averageIntervalMs > 0 ? 60000 / averageIntervalMs : 0,
            backspace_rate: keystrokes > 0 ? Math.min(1, backspaces / keystrokes) : 0,
            paste_count: (Number(metrics.pasteCount) || 0) / questionCount,
            tab_switch: (Number(metrics.tabSwitches) || 0) / questionCount,
            focus_loss: (Number(metrics.focusLosses) || 0) / questionCount,
            avg_idle: idle.length ? idle.reduce((sum, value) => sum + value, 0) / idle.length : 0,
            max_idle: idle.length ? Math.max(...idle) : 0,
            edit_count: (Number(metrics.editCount) || 0) / questionCount,
            time_to_first_key: delays.length ? delays.reduce((sum, value) => sum + value, 0) / delays.length : 0,
            answer_duration: (Number(metrics.answerDurationSeconds) || 0) / questionCount
        };
    }

    return { createExamLifecycle, createQuestionNavigation, calculateWpm, shouldRecommendManualReview, calculateBehaviorFeatures };
});
