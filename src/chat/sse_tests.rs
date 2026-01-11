use bytes::Bytes;
use futures::stream::StreamExt;

use super::create_sse_stream;
use crate::error::LLMError;

#[tokio::test]
async fn test_create_sse_stream_handles_split_utf8() {
    let test_data = "data: Positive reactions\n\n".as_bytes();

    let chunks: Vec<Result<Bytes, reqwest::Error>> = vec![
        Ok(Bytes::from(&test_data[..10])),
        Ok(Bytes::from(&test_data[10..])),
    ];

    let mock_response = create_mock_response(chunks);

    let parser = |event: &str| -> Result<Option<String>, LLMError> {
        if let Some(content) = event.strip_prefix("data: ") {
            let content = content.trim();
            if content.is_empty() {
                return Ok(None);
            }
            return Ok(Some(content.to_string()));
        }
        Ok(None)
    };

    let mut stream = create_sse_stream(mock_response, parser);

    let mut results = Vec::new();
    while let Some(result) = stream.next().await {
        results.push(result);
    }

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].as_ref().unwrap(), "Positive reactions");
}

#[tokio::test]
async fn test_create_sse_stream_handles_split_sse_events() {
    let event1 = "data: First event\n\n";
    let event2 = "data: Second event\n\n";
    let combined = format!("{event1}{event2}");
    let test_data = combined.as_bytes().to_vec();

    let split_point = event1.len() + 5;
    let chunks: Vec<Result<Bytes, reqwest::Error>> = vec![
        Ok(Bytes::from(test_data[..split_point].to_vec())),
        Ok(Bytes::from(test_data[split_point..].to_vec())),
    ];

    let mock_response = create_mock_response(chunks);

    let parser = |event: &str| -> Result<Option<String>, LLMError> {
        if let Some(content) = event.strip_prefix("data: ") {
            let content = content.trim();
            if content.is_empty() {
                return Ok(None);
            }
            return Ok(Some(content.to_string()));
        }
        Ok(None)
    };

    let mut stream = create_sse_stream(mock_response, parser);

    let mut results = Vec::new();
    while let Some(result) = stream.next().await {
        results.push(result);
    }

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].as_ref().unwrap(), "First event");
    assert_eq!(results[1].as_ref().unwrap(), "Second event");
}

#[tokio::test]
async fn test_create_sse_stream_handles_multibyte_utf8_split() {
    let multibyte_char = "✨";
    let event = format!("data: Star {multibyte_char}\n\n");
    let test_data = event.as_bytes().to_vec();

    let emoji_start = event.find(multibyte_char).unwrap();
    let split_in_emoji = emoji_start + 1;

    let chunks: Vec<Result<Bytes, reqwest::Error>> = vec![
        Ok(Bytes::from(test_data[..split_in_emoji].to_vec())),
        Ok(Bytes::from(test_data[split_in_emoji..].to_vec())),
    ];

    let mock_response = create_mock_response(chunks);

    let parser = |event: &str| -> Result<Option<String>, LLMError> {
        if let Some(content) = event.strip_prefix("data: ") {
            let content = content.trim();
            if content.is_empty() {
                return Ok(None);
            }
            return Ok(Some(content.to_string()));
        }
        Ok(None)
    };

    let mut stream = create_sse_stream(mock_response, parser);

    let mut results = Vec::new();
    while let Some(result) = stream.next().await {
        results.push(result);
    }

    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].as_ref().unwrap(),
        &format!("Star {multibyte_char}")
    );
}

fn create_mock_response(chunks: Vec<Result<Bytes, reqwest::Error>>) -> reqwest::Response {
    use http_body_util::StreamBody;
    use reqwest::Body;

    let frame_stream = futures::stream::iter(
        chunks
            .into_iter()
            .map(|chunk| chunk.map(|bytes| hyper::body::Frame::data(bytes))),
    );

    let body = StreamBody::new(frame_stream);
    let body = Body::wrap(body);

    let http_response = http::Response::builder().status(200).body(body).unwrap();

    http_response.into()
}
