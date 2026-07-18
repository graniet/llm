use axum::http::StatusCode;

pub type ApiResult<T> = Result<T, (StatusCode, String)>;

pub fn bad_request(msg: impl Into<String>) -> (StatusCode, String) {
    (StatusCode::BAD_REQUEST, msg.into())
}

pub fn unauthorized(msg: impl Into<String>) -> (StatusCode, String) {
    (StatusCode::UNAUTHORIZED, msg.into())
}

pub fn internal_error(msg: impl Into<String>) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, msg.into())
}

pub fn parse_model(model: &str) -> ApiResult<(String, String)> {
    let (provider_id, model_name) = model
        .split_once(':')
        .ok_or_else(|| bad_request("Invalid model format"))?;
    if provider_id.trim().is_empty() || model_name.trim().is_empty() {
        return Err(bad_request("Invalid model format"));
    }
    Ok((provider_id.to_string(), model_name.to_string()))
}

pub fn transform_response(resp: String, transform: &str) -> String {
    match transform {
        "extract_think" => resp
            .lines()
            .skip_while(|line| !line.contains("<think>"))
            .take_while(|line| !line.contains("</think>"))
            .map(|line| line.replace("<think>", "").trim().to_string())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        "trim_whitespace" => resp.trim().to_string(),
        "extract_json" => {
            // Extract the contents of a ```json ... ``` fenced block.
            // Searching for a bare ``` to mark the end would match the first
            // fence anywhere in the response (e.g. a preceding ```python
            // block), which can put the closing index before the opening one
            // and panic on the slice. Anchor after the opening ```json marker,
            // and only accept a closing fence that is alone on its line so a
            // ``` inside a JSON string value does not truncate early.
            match resp.find("```json") {
                Some(start) => {
                    let after = start + "```json".len();
                    let inner = &resp[after..];
                    let end = bare_closing_fence_offset(inner).unwrap_or(inner.len());
                    inner[..end].trim().to_string()
                }
                None => resp.trim().to_string(),
            }
        }
        _ => resp,
    }
}

/// Index of a line that is only ``` (optional surrounding whitespace).
fn bare_closing_fence_offset(s: &str) -> Option<usize> {
    let mut offset = 0;
    for line in s.split_inclusive('\n') {
        if line.trim() == "```" {
            let indent = line.len() - line.trim_start().len();
            return Some(offset + indent);
        }
        offset += line.len();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::transform_response;

    // A code fence preceding the ```json block used to make the closing-fence
    // index land before the opening one, panicking on `&resp[start..end]`.
    #[test]
    fn extract_json_with_preceding_code_fence_does_not_panic() {
        let resp =
            "Example:\n```python\nprint(1)\n```\nResult:\n```json\n{\"a\":1}\n```".to_string();
        let out = transform_response(resp, "extract_json");
        assert_eq!(out, "{\"a\":1}");
    }

    #[test]
    fn extract_json_returns_fenced_object() {
        let resp = "```json\n{\"ok\": true}\n```".to_string();
        assert_eq!(transform_response(resp, "extract_json"), "{\"ok\": true}");
    }

    #[test]
    fn extract_json_without_fence_returns_trimmed_response() {
        let resp = "  {\"x\":2}  ".to_string();
        assert_eq!(transform_response(resp, "extract_json"), "{\"x\":2}");
    }

    #[test]
    fn extract_json_open_fence_without_close_returns_remainder() {
        let resp = "```json\n{\"partial\": 1}".to_string();
        assert_eq!(transform_response(resp, "extract_json"), "{\"partial\": 1}");
    }

    #[test]
    fn extract_json_ignores_embedded_fence_inside_json_string() {
        let resp =
            "```json\n{\"answer\": \"```python\\nprint(1)\\n```\"}\n```".to_string();
        assert_eq!(
            transform_response(resp, "extract_json"),
            "{\"answer\": \"```python\\nprint(1)\\n```\"}"
        );
    }
}
