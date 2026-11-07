"""Hash-bound target provenance; audit metadata never enters model features."""

DISPUTED_TRAIN_SHA256 = '239efa4b3aa667b3681f6e1062d86ca5b0ba441364b8f2b894b4c428e272ba81'


def target_status(manifest):
    digest = manifest.get('inputs', {}).get('train', {}).get('sha256')
    known = digest == DISPUTED_TRAIN_SHA256
    return {
        'status': 'unresolved_target_semantics' if known else 'not_reviewed',
        'train_sha256': digest,
        'dataset_version': manifest.get('dataset_version'),
        'disputed_rows': 81 if known else None,
        'organizer_resolution': 'pending',
        'message': (
            'В train 81 из 320 меток SI = 10000 имеют неопределённый смысл. '
            'Эти значения уже есть в исходном архиве; все 81 строки имеют пустой IC50. '
            'Версия с нулями описана коллегой, но здесь не проверена. '
            'Прогнозы и MSE относятся только к сохранённой версии меток; нужен ответ организаторов.'
            if known else
            'Смысл меток этой версии данных не проверен. Прогнозы и MSE относятся только к её меткам.'
        ),
        'comparison_note': 'MSE разных версий меток нельзя напрямую сравнивать как улучшение модели.',
        'report_url': '/download/target_version_investigation.md' if known else None,
    }


def require_same_dataset(current, reference):
    """Fail closed before reusing another run's OOF scores or declaring improvement."""
    if not current.get('dataset_version') or current.get('dataset_version') != reference.get('dataset_version'):
        raise ValueError('Dataset version mismatch: cross-version MSE comparison is not allowed')
    for name in ('train', 'test', 'sample'):
        left = current.get('inputs', {}).get(name, {}).get('sha256')
        right = reference.get('inputs', {}).get(name, {}).get('sha256')
        if not left or left != right:
            raise ValueError(f'Dataset hash mismatch ({name}): cross-version MSE comparison is not allowed')
