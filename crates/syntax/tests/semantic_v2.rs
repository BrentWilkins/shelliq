use shelliq_syntax::SyntaxDocumentV1;
use shelliq_syntax::semantic::{RedirectOperatorV2, SemanticDocumentV1, SemanticDocumentV2, StatementV2};

fn lower_v2(source: &str) -> SemanticDocumentV2 {
    let syntax = SyntaxDocumentV1::parse(source).unwrap();
    SemanticDocumentV2::lower(&syntax).unwrap()
}

#[test]
fn v1_wire_contract_remains_available() {
    let syntax = SyntaxDocumentV1::parse("print ok").unwrap();
    let document = SemanticDocumentV1::lower(&syntax).unwrap();
    assert_eq!(document.version(), 1);
    assert_eq!(serde_json::to_value(document).unwrap()["v"], 1);
}

#[test]
fn v2_lowers_and_validates_here_string_redirects() {
    let document = lower_v2("jq . <<< \"$json\"");
    let StatementV2::Pipeline { stages, .. } = &document.statements()[0] else {
        panic!("expected pipeline statement");
    };
    assert_eq!(stages[0].redirects()[0].operator(), RedirectOperatorV2::HereString);
    assert_eq!(stages[0].redirects()[0].target().source(), "\"$json\"");
    assert_eq!(document.render(), "jq . <<<\"$json\"");
    document.validate().unwrap();
}

#[test]
fn v2_lowers_scalar_and_array_declarations() {
    let document = lower_v2("typeset -i count=0; typeset -A hosts=(staging stage.example.com prod prod.example.com)");
    let StatementV2::Declaration {
        utility,
        options,
        assignments,
    } = &document.statements()[0]
    else {
        panic!("expected declaration statement");
    };
    assert_eq!(utility.source(), "typeset");
    assert_eq!(options[0].source(), "-i");
    assert_eq!(assignments[0].name(), "count");
    assert_eq!(assignments[0].value().unwrap().source(), "0");

    let StatementV2::Declaration { assignments, .. } = &document.statements()[1] else {
        panic!("expected declaration statement");
    };
    assert_eq!(assignments[0].name(), "hosts");
    assert_eq!(
        assignments[0].value().unwrap().source(),
        "(staging stage.example.com prod prod.example.com)"
    );
    document.validate().unwrap();
}

#[test]
fn v2_serde_round_trip_preserves_semantic_contract() {
    let document = lower_v2("typeset -i count=0");
    assert_eq!(document.version(), 2);
    let json = serde_json::to_string(&document).unwrap();
    let decoded: SemanticDocumentV2 = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, document);
    decoded.validate().unwrap();
}
