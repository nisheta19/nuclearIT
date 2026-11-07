import json

import numpy as np
import pandas as pd
import pytest
from sklearn.kernel_ridge import KernelRidge

from .test_pipeline import dataset
from vacv.chemistry import featurize
from vacv.kernel import TanimotoRidge, tanimoto_kernel
from vacv.pipeline import predict
from vacv.robust_experiment import promotion_check, robust_run
from vacv.robust_search import candidate_grid


def test_tanimoto_matches_hand_calculation_and_rejects_invalid():
    x = np.array([[1, 1, 0], [1, 0, 1], [0, 0, 0]])
    kernel = tanimoto_kernel(x, x)
    np.testing.assert_allclose(kernel, [[1, 1/3, 0], [1/3, 1, 0], [0, 0, 1]])
    for bad in (np.array([[np.nan, 0, 1]]), np.array([[-1, 0, 1]])):
        with pytest.raises(ValueError):
            tanimoto_kernel(bad, x)


def test_kernel_regression_offset_and_cached_matrix():
    x = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    q = np.array([[1, 0, 0], [0, 0, 1]])
    y = np.array([0., 20., 200.])
    gram = tanimoto_kernel(x, x)
    model = TanimotoRidge(alpha=.3).fit(x, y)
    reference = KernelRidge(alpha=.3, kernel="precomputed").fit(gram, y-y.mean())
    np.testing.assert_allclose(model.predict(q), reference.predict(tanimoto_kernel(q,x))+y.mean())
    cached = TanimotoRidge(alpha=.3).fit(x, y, gram=gram)
    np.testing.assert_array_equal(cached.predict(q), model.predict(q))
    original = model.target_offset_
    model.predict(q)
    assert model.target_offset_ == original == y.mean()


def test_feature_morgan_preserves_groups_but_changes_features():
    spec = {"radius":2,"bits":128,"include_chirality":True}
    normal, x = featurize(["CCO", "OCC", "c1ccncc1"], spec)
    feature, z = featurize(["CCO", "OCC", "c1ccncc1"], {**spec,"atom_invariants":"features"})
    pd.testing.assert_frame_equal(normal, feature)
    assert not np.array_equal(x,z)
    np.testing.assert_array_equal(z[0],z[1])


def test_promotion_requires_improvement_in_all_four_checks():
    records = [{"phase": phase,"scheme": scheme,"model": model,"mse": mse} for phase in ("historical","additional") for scheme in ("structure","scaffold") for model,mse in (("baseline_ridge",100.),("robust_search",90.))]
    table = pd.DataFrame(records)
    assert promotion_check(table)[0]
    table.loc[7,"mse"] = 100.
    assert not promotion_check(table)[0]


def test_kernel_run_nested_groups_saved_predictions_and_reload(dataset, tmp_path):
    frame = pd.read_csv(dataset["train"])
    frame.loc[2:5,"SMILES"] = ["c1ccoc1","c1ccsc1","C1COCC1","C1CCNC1"]
    frame.to_csv(dataset["train"],index=False)
    dataset.update(structure_seeds=[42,19],scaffold_seeds=[-1,19],historical_seeds={"structure":[42],"scaffold":[-1]},promotion_rule="strict improvement on both schemes and phases")
    dataset["robust_search"] = {"inner_folds":2,"inner_seed":2903,"ridge_alphas":[10.],"kernel_alphas":[1.],"kernel_weights":[1.],"blend_linear_alpha":10.,"feature_sets":[
        {**dataset["fingerprint"],"name":"r2_binary","atom_invariants":"connectivity"},
        {**dataset["fingerprint"],"name":"r2_features","atom_invariants":"features"}]}
    assert len(candidate_grid(dataset)) == 4
    output=tmp_path/"robust"
    result=robust_run(dataset,output)
    outer=pd.read_csv(output/"folds.csv")
    inner=pd.read_csv(output/"inner_folds.csv")
    structures=pd.read_csv(output/"train_structures.csv").set_index("train_row_id")
    for (scheme,seed,fold,inner_scheme),part in inner.groupby(["scheme","seed","outer_fold","inner_scheme"]):
        validation=outer[(outer.scheme==scheme)&(outer.seed==seed)&(outer.fold==fold)]
        assert not set(part.train_row_id)&set(validation.train_row_id)
        assert part.groupby("group_id").inner_fold.nunique().eq(1).all()
        # A scaffold outer fold has no shared scaffolds even when its inner split uses structures.
        if scheme=="scaffold":
            assert not set(structures.loc[part.train_row_id,"scaffold_id"]) & set(structures.loc[validation.train_row_id,"scaffold_id"])
    ranking=pd.read_csv(output/"inner_scores.csv")
    choices=pd.read_csv(output/"outer_choices.csv")
    for choice in choices.itertuples():
        group=ranking[(ranking.scheme==choice.scheme)&(ranking.seed==choice.seed)&(ranking.outer_fold==choice.outer_fold)]
        assert choice.objective==pytest.approx(group.objective.min())
    result2=predict(str(output/"model.joblib"),dataset["test"],dataset["sample"],str(tmp_path/"again.csv"))
    assert result2["valid_format"]
    assert (tmp_path/"again.csv").read_bytes()==(output/"submission.csv").read_bytes()
    assert len(result["promotion_checks"])==4
    assert "SYNTHETIC_TEST_FIXTURE_ONLY" in (output/"report.md").read_text(encoding="utf-8")
