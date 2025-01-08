# ai/training/evaluator.py

async def _calculate_metrics(self, test_data: List[Dict[str, Any]]) -> TrainingMetrics:
    """
    محاسبه متریک‌های عملکرد مدل

    این متد متریک‌های مختلف عملکرد مانند دقت، صحت، فراخوانی و غیره را محاسبه می‌کند.
    همچنین متوسط زمان پاسخ و نرخ خطا را نیز محاسبه می‌کند.

    Args:
        test_data: لیستی از داده‌های تست به همراه نتایج مدل و GPT

    Returns:
        TrainingMetrics: متریک‌های محاسبه شده
    """
    predictions = []
    true_labels = []
    response_times = []
    error_count = 0
    total_count = len(test_data)

    # جمع‌آوری داده‌ها برای محاسبه متریک‌ها
    for sample in test_data:
        try:
            if 'model_result' in sample and 'gpt_result' in sample:
                model_result = sample['model_result']
                gpt_result = sample['gpt_result']

                # جمع‌آوری پیش‌بینی‌ها
                predictions.append(self._normalize_prediction(model_result))
                true_labels.append(self._normalize_prediction(gpt_result))

                # جمع‌آوری زمان پاسخ
                if 'response_time' in model_result:
                    response_times.append(model_result['response_time'])

            else:
                error_count += 1

        except Exception as e:
            logger.error(f"Error processing test sample: {str(e)}")
            error_count += 1

    # محاسبه متریک‌های اصلی
    accuracy = np.mean([p == t for p, t in zip(predictions, true_labels)]) if predictions else 0

    # محاسبه ماتریس درهم‌ریختگی
    confusion_matrix = self._calculate_confusion_matrix(predictions, true_labels)

    # محاسبه precision و recall
    precision = self._calculate_precision(confusion_matrix)
    recall = self._calculate_recall(confusion_matrix)

    # محاسبه F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # محاسبه متریک‌های پاسخ‌دهی
    avg_response_time = np.mean(response_times) if response_times else 0
    error_rate = error_count / total_count if total_count > 0 else 0

    return TrainingMetrics(
        loss=1.0 - accuracy,  # تبدیل دقت به loss
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        avg_response_time=avg_response_time,
        error_rate=error_rate
    )


def _normalize_prediction(self, result: Dict[str, Any]) -> Any:
    """
    نرمال‌سازی خروجی مدل برای مقایسه

    این متد نتایج خام مدل‌ها را به فرمت قابل مقایسه تبدیل می‌کند.
    """
    # این متد باید بر اساس نوع داده و خروجی مدل پیاده‌سازی شود
    if 'prediction' in result:
        return result['prediction']
    elif 'class' in result:
        return result['class']
    elif 'label' in result:
        return result['label']
    return result.get('output')


def _calculate_confusion_matrix(self, predictions: List[Any],
                                true_labels: List[Any]) -> np.ndarray:
    """
    محاسبه ماتریس درهم‌ریختگی

    Args:
        predictions: پیش‌بینی‌های مدل
        true_labels: برچسب‌های واقعی

    Returns:
        ماتریس درهم‌ریختگی 2x2
    """
    unique_labels = list(set(predictions + true_labels))
    matrix_size = len(unique_labels)
    matrix = np.zeros((matrix_size, matrix_size))

    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    for pred, true in zip(predictions, true_labels):
        pred_idx = label_to_idx[pred]
        true_idx = label_to_idx[true]
        matrix[true_idx][pred_idx] += 1

    return matrix


def _calculate_precision(self, confusion_matrix: np.ndarray) -> float:
    """
    محاسبه precision از ماتریس درهم‌ریختگی

    Args:
        confusion_matrix: ماتریس درهم‌ریختگی

    Returns:
        مقدار precision
    """
    true_positives = np.diag(confusion_matrix)
    false_positives = np.sum(confusion_matrix, axis=0) - true_positives

    precision = np.mean(
        true_positives / (true_positives + false_positives + 1e-10)
    )
    return float(precision)


def _calculate_recall(self, confusion_matrix: np.ndarray) -> float:
    """
    محاسبه recall از ماتریس درهم‌ریختگی

    Args:
        confusion_matrix: ماتریس درهم‌ریختگی

    Returns:
        مقدار recall
    """
    true_positives = np.diag(confusion_matrix)
    false_negatives = np.sum(confusion_matrix, axis=1) - true_positives

    recall = np.mean(
        true_positives / (true_positives + false_negatives + 1e-10)
    )
    return float(recall)


def _calculate_performance_score(self, metrics: TrainingMetrics) -> float:
    """
    محاسبه امتیاز کلی عملکرد مدل

    این متد یک امتیاز بین 0 تا 1 برای عملکرد کلی مدل محاسبه می‌کند
    که شامل ترکیبی از متریک‌های مختلف است.

    Args:
        metrics: متریک‌های محاسبه شده

    Returns:
        امتیاز کلی عملکرد
    """
    # وزن‌های مختلف برای متریک‌های مختلف
    weights = {
        'accuracy': 0.3,
        'f1_score': 0.3,
        'response_time': 0.2,
        'error_rate': 0.2
    }

    # نرمال‌سازی زمان پاسخ (کمتر بهتر)
    normalized_response_time = 1.0 - min(metrics.avg_response_time / 5.0, 1.0)

    # محاسبه امتیاز نهایی
    score = (
            weights['accuracy'] * metrics.accuracy +
            weights['f1_score'] * metrics.f1_score +
            weights['response_time'] * normalized_response_time +
            weights['error_rate'] * (1.0 - metrics.error_rate)
    )

    return float(score)


async def _detect_performance_drift(self) -> bool:
    """
    تشخیص انحراف در عملکرد مدل

    این متد تغییرات ناگهانی یا تدریجی در عملکرد مدل را شناسایی می‌کند.

    Returns:
        True اگر انحراف عملکرد تشخیص داده شود
    """
    if len(self.metrics_history) < 2:
        return False

    # بررسی روند متریک‌های اخیر
    recent_metrics = self.metrics_history[-5:]  # 5 نمونه آخر
    accuracy_trend = [m.accuracy for m in recent_metrics]
    f1_trend = [m.f1_score for m in recent_metrics]

    # تشخیص انحراف با استفاده از انحراف معیار
    accuracy_std = np.std(accuracy_trend)
    f1_std = np.std(f1_trend)

    return (accuracy_std > self.config.drift_threshold or
            f1_std > self.config.drift_threshold)


async def _generate_recommendations(self, metrics: TrainingMetrics,
                                    performance_score: float,
                                    drift_detected: bool) -> List[str]:
    """
    تولید پیشنهادات برای بهبود عملکرد

    بر اساس متریک‌ها و وضعیت مدل، پیشنهاداتی برای بهبود ارائه می‌دهد.

    Args:
        metrics: متریک‌های فعلی
        performance_score: امتیاز کلی عملکرد
        drift_detected: آیا انحراف عملکرد تشخیص داده شده

    Returns:
        لیست پیشنهادات
    """
    recommendations = []

    # بررسی دقت مدل
    if metrics.accuracy < self.config.performance_threshold:
        recommendations.append(
            "دقت مدل پایین‌تر از حد انتظار است. پیشنهاد می‌شود:"
            "\n- داده‌های آموزشی بیشتری جمع‌آوری شود"
            "\n- پارامترهای مدل بهینه‌سازی شوند"
            "\n- از تکنیک‌های تنظیم مدل استفاده شود"
        )

    # بررسی زمان پاسخ
    if metrics.avg_response_time > 2.0:  # بیش از 2 ثانیه
        recommendations.append(
            "زمان پاسخ‌دهی بالاست. پیشنهاد می‌شود:"
            "\n- از تکنیک‌های بهینه‌سازی مدل استفاده شود"
            "\n- سیستم کش‌گذاری بهبود یابد"
            "\n- منابع محاسباتی افزایش یابد"
        )

    # بررسی نرخ خطا
    if metrics.error_rate > 0.1:  # بیش از 10%
        recommendations.append(
            "نرخ خطای سیستم بالاست. پیشنهاد می‌شود:"
            "\n- مکانیزم‌های مدیریت خطا بهبود یابد"
            "\n- پیش‌پردازش داده‌ها بازبینی شود"
            "\n- محدودیت‌های ورودی بررسی شود"
        )

    # بررسی انحراف عملکرد
    if drift_detected:
        recommendations.append(
            "انحراف در عملکرد مدل مشاهده شده است. پیشنهاد می‌شود:"
            "\n- داده‌های جدید بررسی و تحلیل شوند"
            "\n- آموزش مجدد مدل با داده‌های جدید انجام شود"
            "\n- تغییرات احتمالی در توزیع داده‌ها بررسی شود"
        )

    return recommendations


async def _save_evaluation_result(self, model_id: str,
                                  result: EvaluationResult):
    """
    ذخیره نتایج ارزیابی

    نتایج ارزیابی را برای تحلیل‌های بعدی ذخیره می‌کند.

    Args:
        model_id: شناسه مدل
        result: نتایج ارزیابی
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = self.results_dir / f"{model_id}_{timestamp}.json"

    try:
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'model_id': model_id,
                'timestamp': result.timestamp.isoformat(),
                'metrics': {
                    'accuracy': result.metrics.accuracy,
                    'precision': result.metrics.precision,
                    'recall': result.metrics.recall,
                    'f1_score': result.metrics.f1_score,
                    'avg_response_time': result.metrics.avg_response_time,
                    'error_rate': result.metrics.error_rate
                },
                'performance_score': result.performance_score,
                'drift_detected': result.drift_detected,
                'recommendations': result.recommendations
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved evaluation results to {result_file}")

    except Exception as e:
        logger.error(f"Failed to save evaluation results: {str(e)}")
        # ادامه اجرا حتی در صورت خطا در ذخیره‌سازی