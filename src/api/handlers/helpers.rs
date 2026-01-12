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
            let json_start = resp.find("```json").unwrap_or(0);
            let json_end = resp.find("```").unwrap_or(resp.len());
            let json_str = &resp[json_start..json_end];
            serde_json::from_str::<String>(json_str)
                .unwrap_or_else(|_| "Invalid JSON response".to_string())
        }
        _ => resp,
    }
}
