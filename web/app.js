const $ = id => document.getElementById(id);
const number = (value, digits = 0) => new Intl.NumberFormat('ru-RU', {maximumFractionDigits: digits}).format(value);
let summary = null;
let example = 0;
let controller = null;
function view() {
  const current = ['#quality','#batch'].includes(location.hash) ? location.hash.slice(1) : 'predict';
  for (const name of ['predict','batch','quality']) {
    $(`${name}-view`).hidden = name !== current;
    $(`nav-${name}`).removeAttribute('aria-current');
  }
  $(`nav-${current}`).setAttribute('aria-current','page');
  document.title = 'VACV SI — ' + {predict:'проверка молекулы',batch:'список молекул',quality:'проверка качества'}[current];
}
addEventListener('hashchange', view);
view();
function row(values, className = '') {
  const tr = document.createElement('tr'); tr.className = className;
  values.forEach(value => { const td = document.createElement('td'); td.textContent = value; tr.append(td); });
  return tr;
}
async function initialize() {
  try {
    const response = await fetch('/api/summary');
    if (!response.ok) throw new Error('Сведения о модели недоступны. Перезапустите локальное приложение.');
    summary = await response.json();
    $('run-label').textContent = `${summary.run_name} · ${summary.audit.train_rows} строк train`;
    $('dataset-description').textContent = `${summary.audit.train_rows} наблюдений из исходного набора организаторов. Нулевых меток: ${summary.audit.zero_targets}; меток 10000: ${summary.audit.targets_10000}. Все сохранены без исправлений. Внешние данные не используются.`;
    const referenceLabel = summary.reference_label || 'Исходный Ridge';
    $('decision-title').textContent = summary.promoted ? 'Новая модель прошла критерии выпуска' : `Для прогнозов сохранена предыдущая модель`;
    const baseline = summary.leaderboard.find(r => r.model === summary.reference_strategy);
    const candidateId = summary.comparison_candidate || 'tuned_ridge';
    const tuned = summary.leaderboard.find(r => r.model === candidateId);
    const checks = summary.promotion_checks || [];
    let decision = summary.promoted ? 'Процедура подбора прошла критерии выпуска. Прогнозы рассчитаны новой моделью.' : `Подбор не прошёл все критерии выпуска. Для прогнозов используется: ${referenceLabel}.`;
    if (checks.length) {
      decision += ` Сравнение с «${referenceLabel}»: ` + checks.map(c => `${c.phase === 'historical' ? 'прежние' : 'дополнительные'} разбиения, ${c.scheme === 'structure' ? 'структуры' : 'каркасы'}: ${c.improvement_percent >= 0 ? 'снижение' : 'рост'} MSE ${number(Math.abs(c.improvement_percent),2)}%`).join('; ') + '.';
    } else if (tuned && baseline) {
      decision += ` MSE подбора по структурам: ${number(tuned.structure_mse_mean)}, по каркасам: ${number(tuned.scaffold_mse)}; исходный Ridge по каркасам: ${number(baseline.scaffold_mse)}.`;
    }
    $('decision-description').textContent = decision;
    const names = {baseline_ridge:'Исходный Ridge',tuned_ridge:'Полная процедура подбора',mean:'Среднее train',tuned_r2_binary:'Отдельно: Morgan r2',tuned_r3_binary:'Отдельно: Morgan r3',tuned_r2_count:'Отдельно: счётчики Morgan',tuned_r2_descriptors:'Отдельно: Morgan + дескрипторы',...summary.model_labels};
    for (const item of summary.leaderboard) $('score-rows').append(row([(names[item.model] || item.model) + (item.model === summary.strategy ? ' · используется' : ''), number(item.structure_mse_mean), number(item.scaffold_mse), number(item.structure_mae), number(item.scaffold_mae)], item.model === summary.strategy ? 'active' : item.model === candidateId ? 'search' : ''));
    if (checks.length) $('score-note').textContent = 'В таблице — прежние разбиения: среднее трёх по структурам и одно по каркасам. Результаты дополнительных разбиений приведены выше и в отчёте. Все оценки получены на одном train.';
    const extreme = summary.diagnostics.largest_target_review;
    $('extreme-description').textContent = `Строка ${extreme.row_id}: истинный SI ${number(extreme.target,3)}, отложенный прогноз ${number(extreme.primary_oof_prediction,2)}. На первом разбиении она даёт ${number(extreme.primary_error_share*100,1)}% суммарной квадратичной ошибки. Метка сохранена; первичная публикация ещё не проверена.`;
    $('duplicates-description').textContent = `Из ${summary.diagnostics.duplicate_groups} групп повторяющихся структур в ${summary.diagnostics.conflicting_groups} есть существенные расхождения меток, ещё в ${summary.diagnostics.rounding_groups} — различия масштаба округления. Все строки сохранены и группируются при валидации.`;
    $('validation-note').textContent = summary.validation_note;
    if (summary.reliability) {
      $('reliability-section').hidden = false;
      $('reliability-caption').textContent = `Новая процедура против «${referenceLabel}». Положительная разница означает снижение ошибки; если интервал включает ноль, преимущество этими данными не установлено.`;
      const selected = summary.reliability.rows.filter(r => r.candidate === candidateId && r.metric === 'mse');
      for (const r of selected) $('reliability-rows').append(row([`${r.phase === 'historical' ? 'Прежние' : 'Дополнительные'} / ${r.scheme === 'structure' ? 'структуры' : 'каркасы'}`,number(r.improvement),`${number(r.ci_low)} … ${number(r.ci_high)}`,r.ci_low <= 0 && r.ci_high >= 0 ? 'Да' : 'Нет']));
      $('reliability-note').textContent = summary.reliability.note;
    }
    if (summary.range_comparison?.length) {
      $('ranges-section').hidden = false;
      for (const r of summary.range_comparison.filter(r => r.phase === 'historical')) $('range-rows').append(row([`${r.scheme === 'structure' ? 'Структуры' : 'Каркасы'} / ${r.target_bin}`,r.n,number(r.mse_before),number(r.mse_after),number(r.mae_before),number(r.mae_after)]));
    }
    if (summary.medical_review) {
      $('medical-section').hidden = false;
      const m = summary.medical_review;
      $('medical-summary').textContent = `${m.rows} записей; первоочередных: ${m.priority_counts.P1 || 0}. Причины могут пересекаться: экстремальное SI, значения 10000 и 0, разные метки одинаковых структур.`;
    }
  } catch (error) { $('connection-error').textContent = error.message; $('connection-error').hidden = false; $('predict-button').disabled = true; $('example-button').disabled = true; }
}
function invalidatePrediction() {
  $('prediction-result').hidden = true;
  $('neighbors').hidden = true;
  $('empty-state').hidden = false;
  $('result-state').textContent = 'Нужен новый расчёт';
  $('form-error').hidden = true;
  $('smiles').removeAttribute('aria-invalid');
}
$('example-button').addEventListener('click', () => {
  if (!summary) return;
  const selected = summary.examples[example++ % summary.examples.length];
  $('smiles').value = selected.smiles;
  invalidatePrediction();
  $('example-source').textContent = selected.label + ' · истинная метка неизвестна';
  $('example-source').hidden = false;
  $('form-error').hidden = true;
  $('smiles').focus();
});
$('smiles').addEventListener('input', () => {
  $('example-source').hidden = true;
  invalidatePrediction();
});
$('predict-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (controller) controller.abort();
  controller = new AbortController();
  const thisController = controller;
  const smiles = $('smiles').value.trim();
  if (!smiles) { $('form-error').textContent = 'Введите SMILES или загрузите пример.'; $('form-error').hidden = false; $('smiles').setAttribute('aria-invalid','true'); $('smiles').focus(); return; }
  $('form-error').hidden = true; $('smiles').removeAttribute('aria-invalid');
  $('predict-button').disabled = true; $('example-button').disabled = true; $('smiles').readOnly = true;
  $('result-state').textContent = 'Рассчитываем…'; document.querySelector('.result-area').setAttribute('aria-busy','true');
  const timeout = setTimeout(() => thisController.abort(), 20000);
  try {
    const response = await fetch('/api/predict', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({smiles}),signal:thisController.signal});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Не удалось получить прогноз.');
    $('molecule-image').src = result.molecule_image;
    $('si-value').textContent = number(result.predicted_si,2);
    $('similarity-value').textContent = number(result.max_similarity,3);
    $('canonical-smiles').textContent = result.canonical_smiles;
    let note = result.clipped_to_zero ? 'Отрицательный ответ модели ограничен нулём. Это не означает подтверждённую неактивность.' : 'Модельная оценка в исходной шкале SI.';
    if (result.training_matches) note += ` Такая структура встречается в train (${result.training_matches} наблюд.). Прогноз вычислен моделью, метка не скопирована.`;
    if (result.max_similarity < .3) note += ' Близких структур в train мало; для этого случая особенно важна экспериментальная проверка.';
    $('prediction-note').textContent = note;
    $('neighbor-rows').replaceChildren();
    result.neighbors.forEach(n => $('neighbor-rows').append(row([n.row_id,number(n.similarity,3),number(n.observed_si,3)])));
    $('neighbors').hidden = false; $('empty-state').hidden = true; $('prediction-result').hidden = false;
    $('result-state').textContent = 'Расчёт завершён';
  } catch (error) {
    $('form-error').textContent = error.name === 'AbortError' ? 'Расчёт занял слишком много времени. Попробуйте ещё раз.' : error.message;
    $('form-error').hidden = false; $('smiles').setAttribute('aria-invalid','true');
    $('result-state').textContent = $('prediction-result').hidden ? 'Нет результата' : 'Результат предыдущего расчёта';
  } finally { clearTimeout(timeout); $('predict-button').disabled = false; $('example-button').disabled = false; $('smiles').readOnly = false; document.querySelector('.result-area').setAttribute('aria-busy','false'); }
});
initialize();

let batchResult = null;
let batchPage = 0;
let batchUrl = null;
function clearBatch() {
  batchResult = null; batchPage = 0;
  $('batch-results').hidden = true; $('batch-error').hidden = true;
  $('batch-file').removeAttribute('aria-invalid');
  if (batchUrl) URL.revokeObjectURL(batchUrl);
  batchUrl = null;
  $('batch-download').removeAttribute('href');
}
function renderBatch() {
  if (!batchResult) return;
  const filter = $('batch-filter').value;
  const rows = batchResult.rows.filter(r => filter === 'all' || (filter === 'invalid' ? r.status !== 'ok' : r.status === 'ok' && r.max_similarity < .5));
  const start = batchPage * 50;
  $('batch-rows').replaceChildren();
  for (const r of rows.slice(start,start+50)) $('batch-rows').append(row([`${r.row_number}${r.index ? ' / '+r.index : ''}`,r.SMILES || '(пусто)',r.predicted_si == null ? 'Нет прогноза' : number(r.predicted_si,2),r.max_similarity == null ? '—' : number(r.max_similarity,3),r.status === 'ok' ? (r.training_matches ? 'Есть в train' : r.max_similarity < .3 ? 'Мало близких аналогов' : 'Рассчитано') : r.error],r.status === 'ok' ? '' : 'invalid-row'));
  $('batch-page').textContent = rows.length ? `Строки ${start+1}–${Math.min(start+50,rows.length)} из ${rows.length}` : '0 строк';
  $('batch-no-rows').hidden = rows.length > 0;
  $('batch-prev').disabled = batchPage === 0;
  $('batch-next').disabled = start+50 >= rows.length;
}
$('batch-file').addEventListener('change', () => {clearBatch(); $('batch-state').textContent = 'Файл изменён. Запустите новый расчёт.';});
$('batch-filter').addEventListener('change', () => {batchPage=0;renderBatch();});
$('batch-prev').addEventListener('click', () => {batchPage--;renderBatch();});
$('batch-next').addEventListener('click', () => {batchPage++;renderBatch();});
$('batch-template').addEventListener('click', () => {
  const url = URL.createObjectURL(new Blob(['\ufeffindex,SMILES\r\nexample_1,CCO\r\nexample_2,CCN\r\n'],{type:'text/csv;charset=utf-8'}));
  const a = document.createElement('a'); a.href=url;a.download='smiles-example.csv';a.click();setTimeout(() => URL.revokeObjectURL(url),1000);
});
$('batch-form').addEventListener('submit', async event => {
  event.preventDefault(); clearBatch();
  const file = $('batch-file').files[0];
  if (!file || file.size > 2*1024*1024) {
    $('batch-error').textContent = file ? 'Файл превышает 2 МБ. Разделите список на несколько CSV.' : 'Выберите CSV со столбцом SMILES.';
    $('batch-error').hidden = false; $('batch-file').setAttribute('aria-invalid','true'); return;
  }
  $('batch-button').disabled=true; $('batch-file').disabled=true;
  $('batch-form').setAttribute('aria-busy','true'); $('batch-state').textContent='Рассчитываем список…';
  const abort = new AbortController(); const timeout=setTimeout(() => abort.abort(),120000);
  try {
    let csv;
    try {csv = new TextDecoder('utf-8',{fatal:true}).decode(await file.arrayBuffer());}
    catch {throw new Error('Не удалось прочитать UTF-8. Сохраните файл как CSV UTF-8 и повторите.');}
    const response=await fetch('/api/batch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({csv}),signal:abort.signal});
    const result=await response.json();
    if (!response.ok) throw new Error(result.error || 'Не удалось рассчитать список.');
    batchResult=result;
    $('batch-summary').textContent=`Всего ${result.total}. Рассчитано: ${result.valid}; ошибок: ${result.invalid}. Сходство ниже 0,5: ${result.below_0_5}, из них ниже 0,3: ${result.below_0_3}.`;
    $('batch-model').textContent=`Модель: ${result.rows[0].model_name}. Запуск: ${result.rows[0].run_name}. Полный SHA-256 модели включён в CSV.`;
    $('batch-note').textContent=result.note + (result.ignored_columns.length ? ` Игнорируемые столбцы: ${result.ignored_columns.join(', ')}.` : '');
    batchUrl=URL.createObjectURL(new Blob([result.csv],{type:'text/csv;charset=utf-8'}));
    $('batch-download').href=batchUrl; $('batch-results').hidden=false;
    $('batch-filter').value='all'; renderBatch();
    $('batch-state').textContent='Расчёт завершён. Результаты готовы к скачиванию.';
  } catch (error) {
    $('batch-error').textContent=error.name==='AbortError' ? 'Расчёт превысил две минуты. Попробуйте файл меньшего размера.' : error.message;
    $('batch-error').hidden=false; $('batch-file').setAttribute('aria-invalid','true'); $('batch-state').textContent='Расчёт не завершён.';
  } finally {clearTimeout(timeout);$('batch-button').disabled=false;$('batch-file').disabled=false;$('batch-form').setAttribute('aria-busy','false');}
});
