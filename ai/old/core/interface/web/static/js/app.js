$(document).ready(function() {
    const API_BASE_URL = 'http://localhost:8000';
    let isProcessing = false;

    // تابع نمایش/مخفی کردن لودینگ
    function toggleLoading(show) {
        isProcessing = show;
        if (show) {
            $('#loading-overlay').removeClass('hidden');
            disableControls();
        } else {
            $('#loading-overlay').addClass('hidden');
            enableControls();
        }
    }

    // غیرفعال کردن کنترل‌ها
    function disableControls() {
        $('#learn-btn, #ask-btn, #user-input, #language-select').prop('disabled', true);
    }

    // فعال کردن کنترل‌ها
    function enableControls() {
        $('#learn-btn, #ask-btn, #user-input, #language-select').prop('disabled', false);
    }

    // نمایش پیام در خروجی
    function showOutput(content, type = 'normal') {
        const timestamp = new Date().toLocaleTimeString('fa-IR');
        const messageClass = type === 'error' ? 'error-message' : 'normal-message';

        let formattedContent = content;
        if (typeof content === 'object') {
            formattedContent = JSON.stringify(content, null, 2);
        }

        const message = `
            <div class="message ${messageClass} fade-in">
                <div class="message-header">
                    <span class="timestamp">${timestamp}</span>
                </div>
                <pre class="message-content">${formattedContent}</pre>
            </div>
        `;

        $('#output-container').prepend(message);
    }

    // به‌روزرسانی متریک‌ها
    async function updateMetrics() {
        try {
            const response = await $.get(`${API_BASE_URL}/status`);
            $('#memory-usage').text(`${response.memory_usage.toFixed(1)}%`);
            $('#knowledge-count').text(response.knowledge_count);
            $('#active-learners').text(response.active_learners);
        } catch (error) {
            console.error('خطا در به‌روزرسانی متریک‌ها:', error);
        }
    }

    // پردازش درخواست یادگیری
    async function processLearning() {
        if (isProcessing) return;

        const inputText = $('#user-input').val().trim();
        const language = $('#language-select').val();

        if (!inputText) {
            showOutput('لطفاً متنی وارد کنید.', 'error');
            return;
        }

        toggleLoading(true);

        try {
            const response = await $.ajax({
                url: `${API_BASE_URL}/learn`,
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({
                    text: inputText,
                    language: language
                })
            });

            showOutput(response.result);
            await updateMetrics();
        } catch (error) {
            showOutput(error.responseJSON?.detail || 'خطا در پردازش درخواست', 'error');
        } finally {
            toggleLoading(false);
        }
    }

    // پردازش درخواست پرسش
    async function processQuery() {
        if (isProcessing) return;

        const inputText = $('#user-input').val().trim();
        const language = $('#language-select').val();

        if (!inputText) {
            showOutput('لطفاً سوال خود را وارد کنید.', 'error');
            return;
        }

        toggleLoading(true);

        try {
            const response = await $.ajax({
                url: `${API_BASE_URL}/query`,
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({
                    text: inputText,
                    language: language
                })
            });

            showOutput(response.result);
        } catch (error) {
            showOutput(error.responseJSON?.detail || 'خطا در پردازش درخواست', 'error');
        } finally {
            toggleLoading(false);
        }
    }

    // رویدادها
    $('#learn-btn').click(processLearning);
    $('#ask-btn').click(processQuery);

    $('#clear-output').click(() => {
        $('#output-container').empty();
    });

    $('#user-input').keydown(function(e) {
        if (e.ctrlKey && e.keyCode === 13) {
            processLearning();
        }
    });

    // به‌روزرسانی خودکار متریک‌ها
    setInterval(updateMetrics, 5000);
    updateMetrics();

async function checkServerStatus() {
        try {
            await $.get(`${API_BASE_URL}/status`);
            $('#status-indicator').text('فعال').addClass('status-active').removeClass('status-inactive');
        } catch (error) {
            $('#status-indicator').text('غیرفعال').addClass('status-inactive').removeClass('status-active');
        }
    }

    // بررسی دوره‌ای وضعیت سرور
    setInterval(checkServerStatus, 10000);
    checkServerStatus();

    // مدیریت رویدادهای کلیدی
    $(document).keydown(function(e) {
        // Ctrl/Cmd + L برای یادگیری
        if ((e.ctrlKey || e.metaKey) && e.keyCode === 76) {
            e.preventDefault();
            processLearning();
        }
        // Ctrl/Cmd + Q برای پرسش
        if ((e.ctrlKey || e.metaKey) && e.keyCode === 81) {
            e.preventDefault();
            processQuery();
        }
    });

    // مدیریت تغییر سایز پنجره
    $(window).resize(_.debounce(function() {
        const windowHeight = $(window).height();
        const headerHeight = $('header').outerHeight();
        const metricsHeight = $('.metrics-panel').outerHeight();
        const controlsHeight = $('.controls').outerHeight();
        const padding = 40;

        const availableHeight = windowHeight - (headerHeight + metricsHeight + controlsHeight + padding);

        $('.output-content').css('max-height', Math.max(200, availableHeight) + 'px');
    }, 150));
});