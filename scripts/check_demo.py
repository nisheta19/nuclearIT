"""Browser integration check for our local application; requires Edge and requirements-dev."""
import json
import argparse
import csv
import io
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".impeccable" / "review"
OUT.mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8765')
parser.add_argument('--report', type=Path, default=ROOT/'artifacts'/'browser_verification.json')
args = parser.parse_args()

with sync_playwright() as runner:
    browser = runner.chromium.launch(channel="msedge", headless=True)
    results = []
    for name, width, height in [("desktop", 1440, 1000), ("mobile", 390, 844)]:
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url, wait_until="networkidle")
        summary = page.request.get(args.url + '/api/summary').json()
        review = summary['target_review']
        assert review['status'] == 'unresolved_target_semantics' and review['disputed_rows'] == 81
        expect(page.locator('#target-review-notice')).to_be_visible()
        expect(page.locator('#target-review-message')).to_contain_text('81 из 320')
        page.locator('#target-review-notice summary').click()
        expect(page.locator('#target-review-question')).to_be_visible()
        with page.expect_download() as event:
            page.locator('a[href="/download/labels_for_clarification.csv"]').click()
        labels=list(csv.DictReader(io.StringIO(Path(event.value.path()).read_text(encoding='utf-8-sig'))))
        assert len(labels)==81 and {r['train_sha256'] for r in labels}=={review['train_sha256']}
        assert page.request.get(args.url+review['report_url']).status==200
        page.locator('#target-review-notice summary').click()
        page.evaluate("() => document.fonts.ready")
        assert page.locator("#connection-error").is_hidden()
        assert page.locator("#empty-state").is_visible()
        page.locator("#predict-button").click()
        assert "Введите SMILES" in page.locator("#form-error").inner_text()
        page.locator("#smiles").fill("invalid_smiles")
        page.locator("#predict-button").click()
        expect(page.locator("#form-error")).to_be_visible()
        assert "Не удалось разобрать" in page.locator("#form-error").inner_text()
        page.locator("#example-button").click()
        assert page.locator("#smiles").input_value()
        page.locator("#predict-button").click()
        expect(page.locator("#result-state")).to_have_text("Расчёт завершён")
        assert page.locator("#neighbors").is_visible()
        assert page.locator("#neighbor-rows tr").count() == 3
        assert page.locator("#molecule-image").evaluate("image => image.complete && image.naturalWidth > 0")
        page.locator("#example-button").click()
        assert page.locator("#prediction-result").is_hidden()
        assert page.locator("#neighbors").is_hidden()
        expect(page.locator("#result-state")).to_have_text("Нужен новый расчёт")
        page.locator("#predict-button").click()
        expect(page.locator("#result-state")).to_have_text("Расчёт завершён")
        page.locator("#smiles").fill("CCO")
        assert page.locator("#prediction-result").is_hidden()
        page.locator("#example-button").click()
        page.locator("#predict-button").click()
        expect(page.locator("#result-state")).to_have_text("Расчёт завершён")
        assert page.locator("#run-label").is_visible()
        if name == "mobile":
            assert page.locator(".result-area").bounding_box()["y"] < page.locator(".context-area").bounding_box()["y"]
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
        page.locator("#nav-quality").click()
        assert page.locator("#quality-view").is_visible()
        assert page.locator("#score-rows tr").count() == len(summary['leaderboard'])
        assert page.locator("#score-rows tr.active").count() == 1
        assert page.locator('#decision-title').inner_text() == 'Историческое сравнение на одной версии меток'
        if name == "mobile":
            assert page.locator("#quality-view .scroll-hint").first.is_visible()
            assert page.locator(".chart-scroll[tabindex='0']").count() == 2
        page.locator(".chart img").first.scroll_into_view_if_needed()
        page.locator(".chart img").last.scroll_into_view_if_needed()
        page.wait_for_load_state("networkidle")
        assert page.locator(".chart img").evaluate_all("images => images.every(img => img.complete && img.naturalWidth > 0)")
        page.evaluate("scrollTo(0,0)")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=str(OUT / f"{name}-quality.png"), full_page=True)
        assert page.locator('#reliability-rows tr').count() == 4
        assert '89 записей' in page.locator('#medical-summary').inner_text()
        page.locator('#ranges-section summary').click()
        assert page.locator('#range-rows tr').count() == 14
        assert page.request.get(args.url+'/download/medical_review_queue.csv').status == 200
        page.locator('#nav-batch').click()
        page.locator('#batch-button').click()
        expect(page.locator('#batch-error')).to_contain_text('Выберите CSV')
        page.locator('#batch-file').set_input_files({'name':'wrong.csv','mimeType':'text/csv','buffer':b'index\n1\n'})
        page.locator('#batch-button').click()
        expect(page.locator('#batch-error')).to_contain_text('нужен столбец SMILES')
        csv_text='index,SMILES,CC50\n'+'\n'.join(f'{i:03d},{"invalid" if i in (4,8) else "CCO"},999999' for i in range(75))+'\n'
        page.locator('#batch-file').set_input_files({'name':'list.csv','mimeType':'text/csv','buffer':csv_text.encode()})
        page.locator('#batch-button').click()
        expect(page.locator('#batch-state')).to_contain_text('Расчёт завершён',timeout=30000)
        assert page.locator('#batch-rows tr').count() == 50
        page.locator('#batch-next').click()
        assert page.locator('#batch-rows tr').count() == 25
        page.locator('#batch-filter').select_option('invalid')
        assert page.locator('#batch-rows tr').count() == 2
        assert 'Нет прогноза' in page.locator('#batch-rows').inner_text()
        with page.expect_download() as event:
            page.locator('#batch-download').click()
        downloaded=list(csv.DictReader(io.StringIO(Path(event.value.path()).read_text(encoding='utf-8-sig'))))
        assert len(downloaded)==75 and downloaded[0]['index']=='000'
        assert downloaded[4]['predicted_si']=='' and downloaded[0]['run_name']==summary['run_name']
        assert downloaded[0]['model_sha256']==summary['model_sha256']
        assert downloaded[0]['train_sha256']==review['train_sha256']
        assert downloaded[0]['target_review_status']=='unresolved_target_semantics'
        page.locator('#batch-file').set_input_files({'name':'utf8.csv','mimeType':'text/csv','buffer':b'SMILES\n\xff\n'})
        assert page.locator('#batch-results').is_hidden()
        assert page.locator('#batch-download').get_attribute('href') is None
        page.locator('#batch-button').click()
        expect(page.locator('#batch-error')).to_contain_text('UTF-8')
        example='index,SMILES\n001,CCO\n002,c1ccncc1\n003,invalid\n004,CC(=O)O\n005,C1CCCCC1\n'
        page.locator('#batch-file').set_input_files({'name':'example.csv','mimeType':'text/csv','buffer':example.encode()})
        page.locator('#batch-button').click()
        expect(page.locator('#batch-state')).to_contain_text('Расчёт завершён',timeout=30000)
        assert page.locator('#batch-rows tr').count() == 5
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.evaluate('scrollTo(0,0)')
        page.screenshot(path=str(OUT/f'{name}-batch.png'),full_page=True)
        assert not errors, errors
        results.append({"run":summary['run_name'], "viewport": name, "width": width, "height": height, "empty_invalid_valid_inputs": True, "stale_result_cleared_on_edit_and_example": True, "run_caption_visible": True, "molecule_rendered": True, "neighbors": 3, "quality_rows": len(summary['leaderboard']), "batch_csv_upload_export_pagination_filters_errors_stale_state":True, "reliability_rows":4, "range_rows":14, "medical_queue_download":True, "horizontal_page_overflow": False, "js_errors": errors})
        page.close()
    browser.close()
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps({'target_notice_and_downloads_checked':True,'viewports':results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
