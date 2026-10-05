use rust_mlx::chat::ChatTemplate;
#[test]
fn checkpoint_chat_matches_jinja_oracle() {
    let path = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/chat"));
    let template = ChatTemplate::load(path).unwrap();
    let cases: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(path.join("oracle.json")).unwrap()).unwrap();
    for c in cases {
        assert_eq!(
            template
                .render(
                    c["messages"].as_array().unwrap(),
                    c["enable_thinking"].as_bool().unwrap(),
                    c["reasoning_effort"].as_str().unwrap()
                )
                .unwrap(),
            c["expected"].as_str().unwrap()
        );
    }
    assert!(template.render(&[], true, "low").is_err());
    assert!(
        template
            .render(
                &[serde_json::json!({"role":"user","content":[{"image_url":"invalid"}]})],
                true,
                "low"
            )
            .is_err()
    );
    assert!(
        template
            .render(
                &[serde_json::json!({"role":"user","content":"test"})],
                true,
                "high"
            )
            .is_err()
    );
}
