from pathlib import Path

from scripts.run_full_corpus_evaluation import FINALISTS, build_manifest, build_plan


def test_full_corpus_evaluation_uses_only_permitted_gates(tmp_path):
    manifest = build_manifest(tmp_path)
    assert manifest['forbidden_evaluations'] == ['pipeline', 'shadow']

    for finalist in FINALISTS:
        stages = build_plan(finalist, tmp_path)
        assert [stage.name for stage in stages] == [
            'corpus test loss',
            'release development evaluation',
            'release development analysis',
            'retention evaluation',
            'retention analysis',
        ]
        commands = [' '.join(stage.command) for stage in stages]
        assert all('pipeline-compatibility' not in command for command in commands)
        assert all('semantic-shadow' not in command for command in commands)
        assert str(finalist.checkpoint) in commands[0]
        assert '--split test' in commands[0]


def test_full_corpus_evaluation_outputs_are_isolated_by_finalist(tmp_path):
    output_sets = []
    for finalist in FINALISTS:
        outputs = {output for stage in build_plan(finalist, tmp_path) for output in stage.outputs}
        assert all(Path(tmp_path) in output.parents for output in outputs)
        output_sets.append(outputs)

    assert output_sets[0].isdisjoint(output_sets[1])
