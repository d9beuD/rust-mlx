//! Execute the checkpoint's own text chat template, including thinking controls.
use anyhow::{Context, Result, ensure};
use minijinja::{Environment, Error, ErrorKind, context};
use serde_json::Value;
use std::path::Path;

pub struct ChatTemplate {
    source: String,
}
pub fn encode_prompt(
    tokenizer: &tokenizers::Tokenizer,
    path: &Path,
    prompt: &str,
    chat: bool,
    thinking: bool,
    effort: &str,
) -> Result<Vec<u32>> {
    let rendered = if chat {
        ChatTemplate::load(path)?.render(
            &[serde_json::json!({"role":"user","content":prompt})],
            thinking,
            effort,
        )?
    } else {
        prompt.into()
    };
    Ok(tokenizer
        .encode(rendered, false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec())
}
impl ChatTemplate {
    pub fn load(path: &Path) -> Result<Self> {
        let config: Value =
            serde_json::from_slice(&std::fs::read(path.join("tokenizer_config.json"))?)?;
        Ok(Self {
            source: config["chat_template"]
                .as_str()
                .context("missing text chat_template")?
                .into(),
        })
    }
    pub fn render(&self, messages: &[Value], thinking: bool, effort: &str) -> Result<String> {
        ensure!(!messages.is_empty(), "messages must not be empty");
        ensure!(
            ["low", "medium", "xhigh"].contains(&effort),
            "unsupported reasoning_effort"
        );
        for message in messages {
            ensure!(
                matches!(
                    message["role"].as_str(),
                    Some("system" | "user" | "assistant")
                ),
                "only text system/user/assistant messages are supported"
            );
            ensure!(
                message["content"].is_string() && message.get("tool_calls").is_none(),
                "only text messages without tools are supported"
            );
        }
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |message: String| -> std::result::Result<String, Error> {
                Err(Error::new(ErrorKind::InvalidOperation, message))
            },
        );
        env.add_template("chat", &self.source)?;
        Ok(env.get_template("chat")?.render(context!(messages => messages, add_generation_prompt => true, enable_thinking => thinking, reasoning_effort => effort, tools => Vec::<Value>::new()))?)
    }
}
