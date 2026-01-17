use super::*;
use crate::chat::StreamChunk;
use bytes::Bytes;
use futures::stream::StreamExt;

#[tokio::test]
async fn responses_stream_chunks_emits_text_and_done() {
    let text_event = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}\n\n";
    let done_event = "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n";
    let data = format!("{text_event}{done_event}");

    let chunks: Vec<Result<Bytes, reqwest::Error>> = vec![Ok(Bytes::from(data))];
    let response = create_mock_response(chunks);
    let mut stream = create_responses_stream_chunks(response);

    let mut results = Vec::new();
    while let Some(result) = stream.next().await {
        results.push(result.unwrap());
    }

    assert_eq!(results.len(), 2);
    assert!(matches!(&results[0], StreamChunk::Text(text) if text == "Hello"));
    assert!(matches!(&results[1], StreamChunk::Done { stop_reason } if stop_reason == "end_turn"));
}

#[tokio::test]
async fn responses_stream_chunks_emits_tool_use_flow() {
    let added = "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"get_weather\",\"arguments\":\"\"}}\n\n";
    let delta = "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"output_index\":0,\"delta\":\"{\\\"city\\\":\\\"Paris\\\"\"}\n\n";
    let done = "data: {\"type\":\"response.function_call_arguments.done\",\"item_id\":\"fc_1\",\"output_index\":0,\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}\n\n";
    let completed = "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n";
    let payload = format!("{added}{delta}{done}{completed}");
    let split = added.len() + 10;

    let chunks: Vec<Result<Bytes, reqwest::Error>> = vec![
        Ok(Bytes::from(payload[..split].to_string())),
        Ok(Bytes::from(payload[split..].to_string())),
    ];
    let response = create_mock_response(chunks);
    let mut stream = create_responses_stream_chunks(response);

    let mut results = Vec::new();
    while let Some(result) = stream.next().await {
        results.push(result.unwrap());
    }

    assert!(
        matches!(&results[0], StreamChunk::ToolUseStart { id, name, .. } if id == "call_1" && name == "get_weather")
    );
    assert!(
        matches!(&results[1], StreamChunk::ToolUseInputDelta { partial_json, .. } if partial_json.contains("Paris"))
    );
    assert!(
        matches!(&results[2], StreamChunk::ToolUseComplete { tool_call, .. } if tool_call.function.arguments == "{\"city\":\"Paris\"}")
    );
    assert!(matches!(&results[3], StreamChunk::Done { stop_reason } if stop_reason == "tool_use"));
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
