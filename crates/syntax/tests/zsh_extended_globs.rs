use shelliq_syntax::SyntaxDocumentV1;
use shelliq_syntax::semantic::SemanticDocumentV2;

const EXTENDED_GLOB_COMMANDS: &[&str] = &[
    "ls -l *(om[1])",
    "ls -l *(Om[1,3])",
    "ls -l (#i)readme*",
    "ls -lh **/*(.OL[1,5])",
    "ls -l **/*(.W)",
    "ls -l ~/Downloads/*(om[1])",
];

#[test]
fn native_zsh_extended_globs_are_lossless_and_semantic() {
    for source in EXTENDED_GLOB_COMMANDS {
        let syntax = SyntaxDocumentV1::parse(source).unwrap_or_else(|error| panic!("failed to parse {source:?}: {error}"));
        assert_eq!(syntax.render(), *source);
        syntax.validate().unwrap();

        let semantic = SemanticDocumentV2::lower(&syntax).unwrap();
        assert_eq!(semantic.render(), *source);
        semantic.validate().unwrap();
    }
}

#[test]
fn malformed_glob_subscripts_are_not_masked_from_the_grammar() {
    for source in ["ls *(om[1,])", "ls *(om[x])", "ls *(om[1,2,3])"] {
        assert!(
            SyntaxDocumentV1::parse(source).is_err(),
            "malformed qualifier unexpectedly parsed: {source}"
        );
    }
}
