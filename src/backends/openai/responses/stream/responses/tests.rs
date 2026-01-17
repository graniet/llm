use super::*;
use bytes::Bytes;
use futures::stream::StreamExt;

#[tokio::test]
async fn responses_stream_responses_emits_text_and_usage() {
    let text_event = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}\n\n";
    let done_event = "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n";
    let data = format!("{text_event}{done_event}");

    let chunks: Vec<Result<Bytes, reqwest::Error>> = vec![Ok(Bytes::from(data))];
    let response = create_mock_response(chunks);
    let mut stream = create_responses_stream_responses(response, false);

    let mut text = String::new();
    let mut usage_seen = false;
    while let Some(result) = stream.next().await {
        let result = result.unwrap();
        if let Some(choice) = result.choices.first() {
            if let Some(content) = &choice.delta.content {
                text.push_str(content);
            }
        }
        if result.usage.is_some() {
            usage_seen = true;
        }
    }

    assert_eq!(text, "Hello");
    assert!(usage_seen);
}

#[tokio::test]
async fn responses_stream_responses_normalizes_tool_calls() {
    let added = "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"get_weather\",\"arguments\":\"\"}}\n\n";
    let delta = "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"output_index\":0,\"delta\":\"{\\\"city\\\":\\\"Paris\\\"\"}\n\n";
    let done = "data: {\"type\":\"response.function_call_arguments.done\",\"item_id\":\"fc_1\",\"output_index\":0,\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}\n\n";
    let completed = "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n";
    let data = format!("{added}{delta}{done}{completed}");

    let chunks: Vec<Result<Bytes, reqwest::Error>> = vec![Ok(Bytes::from(data))];
    let response = create_mock_response(chunks);
    let mut stream = create_responses_stream_responses(response, true);

    let mut tool_calls = Vec::new();
    while let Some(result) = stream.next().await {
        let result = result.unwrap();
        if let Some(choice) = result.choices.first() {
            if let Some(calls) = &choice.delta.tool_calls {
                tool_calls.extend(calls.clone());
            }
        }
    }

    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].id, "call_1");
    assert_eq!(tool_calls[0].function.name, "get_weather");
    assert_eq!(tool_calls[0].function.arguments, "{\"city\":\"Paris\"}");
}

fn create_mock_response(chunks: Vec<Result<Bytes, reqwest::Error>>) -> reqwest::Response {
    use http_body_util::StreamBody;
    use reqwest::Body;

    let frame_stream = futures::stream::iter(
        chunks
            .into_iter()
            .map(|chunk| chunk.map(hyper::body::Frame::data)),
    );

    let body = StreamBody::new(frame_stream);
    let body = Body::wrap(body);

    let http_response = http::Response::builder().status(200).body(body).unwrap();

    http_response.into()
}
